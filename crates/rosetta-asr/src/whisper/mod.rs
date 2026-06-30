//! Motor ASR Whisper (large-v3-turbo) sobre ONNX Runtime, genérico y
//! cross-platform (Bloque D). Reusa la cascada de EP (`ep::build_session`) y los
//! tipos de `rosetta-core`, replicando la estructura de [`crate::ParakeetEngine`].
//!
//! Flujo: PCM 16k → mel log de 128 bandas (`mel`) → encoder (`input_features`
//! `[1,128,3000]` → `last_hidden_state` `[1,1500,1280]`) → detección de idioma →
//! decode greedy autoregresivo sobre el **decoder merged** con KV-cache de dos
//! ramas (`use_cache_branch`) → tokens → texto (`tokenizer`, byte-level sin C).
//!
//! v1: timestamps a nivel de segmento (un bloque = un `Segment`); `words` vacío
//! (los word-timestamps por DTW se difieren). Contrato I/O verificado con el
//! `.onnx` real (ver `tests/whisper_io_dump.rs`).

mod mel;
mod tokenizer;

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Array3, Array4, Axis};
use ort::session::SessionInputValue;
use ort::value::{DynValue, Value};
use serde::Deserialize;

use rosetta_accel::{Device, EpKind, HwProfile, ep};
use rosetta_core::{DecodeCtx, ModelInfo, Result, RosettaError, Segment, SourceInfo, Transcript};

use crate::Engine;
use crate::ort_util::{argmax, asr_err, extract_3d, extract_4d, find_first};
use mel::MelFrontend;
use tokenizer::WhisperTokenizer;

/// Frecuencia que espera el frontend de Whisper.
const EXPECTED_SR: u32 = 16_000;

/// Subconjunto de `config.json` que necesita el motor.
#[derive(Debug, Deserialize)]
struct WhisperConfig {
    num_mel_bins: usize,
    decoder_layers: usize,
    decoder_attention_heads: usize,
    d_model: usize,
    eos_token_id: i64,
    decoder_start_token_id: i64,
    #[serde(default = "default_max_target")]
    max_target_positions: usize,
}

fn default_max_target() -> usize {
    448
}

fn default_prev_sot() -> i64 {
    50362 // `<|startofprev|>` de Whisper large-v3
}

/// Subconjunto de `generation_config.json`.
#[derive(Debug, Deserialize)]
struct GenConfig {
    no_timestamps_token_id: i64,
    #[serde(default)]
    suppress_tokens: Vec<i64>,
    #[serde(default)]
    begin_suppress_tokens: Vec<i64>,
    task_to_id: HashMap<String, i64>,
    lang_to_id: HashMap<String, i64>,
    #[serde(default = "default_prev_sot")]
    prev_sot_token_id: i64,
}

// La caché KV se gestiona como estado persistente del decode (opt #2): la cross-attn
// (constante tras el prefill) y `encoder_hidden_states` se construyen UNA sola vez
// como `Value` y se alimentan por VISTA (`SessionInputValue::View`) en cada paso;
// solo la self-attn del decoder crece y se reconstruye. Ver `prefill` / `gen_step`.

/// Motor Whisper sobre dos sesiones ONNX (encoder + decoder merged).
pub struct WhisperEngine {
    encoder: ort::session::Session,
    decoder: ort::session::Session,
    mel: MelFrontend,
    tok: WhisperTokenizer,
    cfg: WhisperConfig,
    gcfg: GenConfig,
    head_dim: usize,
    transcribe_id: i64,
    /// Ids de los tokens de idioma (para la detección por argmax).
    lang_ids: Vec<i64>,
    /// id de token de idioma → código ISO ("<|es|>" → "es").
    lang_code: HashMap<i64, String>,
    device_label: String,
    /// Parámetros para reconstruir las sesiones en CPU si el acelerador falla.
    dir: PathBuf,
    hw: HwProfile,
    threads: usize,
    /// `true` si las sesiones actuales ya corren en CPU (no reintentar fallback).
    on_cpu: bool,
}

impl WhisperEngine {
    /// Carga el motor desde un directorio con los `.onnx`, `config.json`,
    /// `generation_config.json` y `vocab.json`.
    pub fn from_dir(dir: &Path, hw: &HwProfile, device: Device, threads: usize) -> Result<Self> {
        // El decoder MERGED int8 de Whisper CUELGA DirectML en ALGUNAS GPUs (GPU
        // device hung, 887A0007: el `If`/use_cache_branch sobre ops int8 no es
        // estable en ciertos drivers/GPU). NO es universal: en otras Windows/GPU
        // puede funcionar. Estrategia (ver `transcribe`): intentar el device pedido
        // y, si la inferencia falla en runtime, reconstruir en CPU + anotar un
        // marcador LOCAL para arrancar directo en CPU las próximas veces en ESTA
        // máquina. Aquí solo aplicamos el atajo si ese marcador ya existe.
        let eff_device = if matches!(device, Device::Auto) && dml_marker(dir).exists() {
            tracing::debug!("marcador DirectML-roto presente: Whisper arranca en CPU");
            Device::Cpu
        } else {
            device
        };
        let (encoder, decoder, primary) = build_sessions(dir, hw, eff_device, threads)?;

        let cfg: WhisperConfig = read_json(&dir.join("config.json"))?;
        let gcfg: GenConfig = read_json(&dir.join("generation_config.json"))?;
        let tok = WhisperTokenizer::from_files(&dir.join("vocab.json"), &dir.join("merges.txt"))?;
        let mel = MelFrontend::new(cfg.num_mel_bins);
        let head_dim = cfg.d_model / cfg.decoder_attention_heads;

        let transcribe_id = *gcfg
            .task_to_id
            .get("transcribe")
            .ok_or_else(|| RosettaError::Asr("generation_config sin task 'transcribe'".into()))?;
        let mut lang_ids: Vec<i64> = gcfg.lang_to_id.values().copied().collect();
        lang_ids.sort_unstable();
        let lang_code: HashMap<i64, String> = gcfg
            .lang_to_id
            .iter()
            .map(|(k, &v)| {
                (
                    v,
                    k.trim_start_matches("<|")
                        .trim_end_matches("|>")
                        .to_string(),
                )
            })
            .collect();

        Ok(Self {
            encoder,
            decoder,
            mel,
            tok,
            cfg,
            gcfg,
            head_dim,
            transcribe_id,
            lang_ids,
            lang_code,
            device_label: primary.label().to_string(),
            dir: dir.to_path_buf(),
            hw: hw.clone(),
            threads,
            on_cpu: matches!(primary, EpKind::Cpu | EpKind::Xnnpack),
        })
    }

    /// PCM mono 16k → `last_hidden_state` `[1, 1500, 1280]`.
    fn run_encoder(&mut self, pcm: &[f32]) -> Result<Array3<f32>> {
        let feats = self.mel.log_mel(pcm).insert_axis(Axis(0)); // [1, n_mels, 3000]
        let outputs = self
            .encoder
            .run(ort::inputs!("input_features" => Value::from_array(feats).map_err(asr_err)?))
            .map_err(|e| RosettaError::Asr(format!("encoder run: {e}")))?;
        extract_3d(&outputs["last_hidden_state"], "last_hidden_state")
    }

    /// Prefill (paso 1, `use_cache_branch=false`): alimenta `input_ids`, el encoder
    /// (por VISTA) y KV dummy (seq=0) en ambas ramas. Devuelve los logits de la última
    /// posición, la self-KV del decoder y la **cross-KV** ya como `Value` (constante el
    /// resto del decode → se reusa por referencia, opt #2).
    #[allow(clippy::type_complexity)]
    fn prefill(
        &mut self,
        input_ids: &[i64],
        enc: &DynValue,
    ) -> Result<(
        Vec<f32>,
        Vec<(Array4<f32>, Array4<f32>)>,
        Vec<(DynValue, DynValue)>,
    )> {
        let layers = self.cfg.decoder_layers;
        let h = self.cfg.decoder_attention_heads;
        let hd = self.head_dim;
        let seq = input_ids.len();

        let ids = Array2::<i64>::from_shape_vec((1, seq), input_ids.to_vec())
            .map_err(|e| RosettaError::Asr(format!("input_ids: {e}")))?;
        let flag = Array1::<bool>::from_vec(vec![false]);
        let empty = Array4::<f32>::zeros((1, h, 0, hd));

        let mut feeds: Vec<(Cow<'static, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(3 + layers * 4);
        feeds.push((
            Cow::Borrowed("input_ids"),
            Value::from_array(ids).map_err(asr_err)?.into(),
        ));
        // encoder_hidden_states por VISTA: no se clona el tensor [1,1500,1280].
        feeds.push((Cow::Borrowed("encoder_hidden_states"), enc.into()));
        feeds.push((
            Cow::Borrowed("use_cache_branch"),
            Value::from_array(flag).map_err(asr_err)?.into(),
        ));
        // Las 4 KV dummy (seq=0) por capa se alimentan POR VALOR (clon de `empty`):
        // en el prefill no hay asimetría owned/borrowed; ambas ramas son tensores
        // vacíos propios. La asimetría opt#2 aparece en `gen_step`.
        for l in 0..layers {
            for part in [
                "decoder.key",
                "decoder.value",
                "encoder.key",
                "encoder.value",
            ] {
                feeds.push((
                    Cow::Owned(past_name(l, part)),
                    Value::from_array(empty.clone()).map_err(asr_err)?.into(),
                ));
            }
        }

        let outputs = self
            .decoder
            .run(feeds)
            .map_err(|e| RosettaError::Asr(format!("decoder run: {e}")))?;

        let logits_last = last_logits(&outputs)?;
        let dec = collect_present_decoder(&outputs, layers)?;

        // Cross-KV → `Value` persistente (se alimentará por VISTA en cada paso, opt#2).
        let mut cross = Vec::with_capacity(layers);
        for l in 0..layers {
            let ek = extract_4d(&outputs[present_name(l, "encoder", "key")], "present enc k")?;
            let ev = extract_4d(
                &outputs[present_name(l, "encoder", "value")],
                "present enc v",
            )?;
            cross.push((
                Value::from_array(ek).map_err(asr_err)?.into_dyn(),
                Value::from_array(ev).map_err(asr_err)?.into_dyn(),
            ));
        }
        Ok((logits_last, dec, cross))
    }

    /// Paso de generación (`use_cache_branch=true`): `encoder_hidden_states` y la
    /// cross-KV (constantes) se alimentan por VISTA; solo la self-KV del decoder se
    /// clona (crece, O(seq)). Devuelve los logits de la última posición y la self-KV
    /// crecida.
    #[allow(clippy::type_complexity)]
    fn gen_step(
        &mut self,
        input_ids: &[i64],
        enc: &DynValue,
        cross: &[(DynValue, DynValue)],
        dec: &[(Array4<f32>, Array4<f32>)],
    ) -> Result<(Vec<f32>, Vec<(Array4<f32>, Array4<f32>)>)> {
        let layers = self.cfg.decoder_layers;
        let seq = input_ids.len();

        let ids = Array2::<i64>::from_shape_vec((1, seq), input_ids.to_vec())
            .map_err(|e| RosettaError::Asr(format!("input_ids: {e}")))?;
        let flag = Array1::<bool>::from_vec(vec![true]);

        let mut feeds: Vec<(Cow<'static, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(3 + layers * 4);
        feeds.push((
            Cow::Borrowed("input_ids"),
            Value::from_array(ids).map_err(asr_err)?.into(),
        ));
        feeds.push((Cow::Borrowed("encoder_hidden_states"), enc.into()));
        feeds.push((
            Cow::Borrowed("use_cache_branch"),
            Value::from_array(flag).map_err(asr_err)?.into(),
        ));
        // ASIMETRÍA opt#2 (INVARIANTE CRÍTICO — no colapsar en un solo modo de paso):
        //   · decoder.{key,value} → POR VALOR (`.clone()`): la self-KV CRECE cada paso.
        //   · encoder.{key,value} → POR VISTA (`(&cross[l].X).into()`): la cross-KV es
        //     CONSTANTE (~61 MB); clonarla por paso regresaría opt#2 (RSS/tiempo).
        // El bucle comparte solo el nombre (`past_name`); cada slot conserva su modo.
        for l in 0..layers {
            let (dk, dv) = &dec[l];
            // decoder → OWNED (clon, crece):
            feeds.push((
                Cow::Owned(past_name(l, "decoder.key")),
                Value::from_array(dk.clone()).map_err(asr_err)?.into(),
            ));
            feeds.push((
                Cow::Owned(past_name(l, "decoder.value")),
                Value::from_array(dv.clone()).map_err(asr_err)?.into(),
            ));
            // encoder → BORROWED (vista, constante): evita clonar ~61 MB por paso.
            feeds.push((
                Cow::Owned(past_name(l, "encoder.key")),
                (&cross[l].0).into(),
            ));
            feeds.push((
                Cow::Owned(past_name(l, "encoder.value")),
                (&cross[l].1).into(),
            ));
        }

        let outputs = self
            .decoder
            .run(feeds)
            .map_err(|e| RosettaError::Asr(format!("decoder run: {e}")))?;

        let logits_last = last_logits(&outputs)?;
        let new_dec = collect_present_decoder(&outputs, layers)?;
        Ok((logits_last, new_dec))
    }

    /// Idioma por argmax de los logits de `[SOT]` sobre los ids de token de idioma.
    /// Aplica `is_finite` y `>` ESTRICTO (mismo tie-break que `argmax`: gana el
    /// PRIMER máximo; NO usar `max_by`+`total_cmp`, que devolvería el ÚLTIMO en
    /// empate y rompería la semántica). Si ningún logit de idioma es finito,
    /// cae al primer id de idioma (no hay forma mejor de elegir y `lang_ids` no
    /// está vacío por construcción del catálogo).
    fn argmax_lang(&self, logits: &[f32]) -> i64 {
        let mut best = self.lang_ids[0];
        let mut best_v = f32::NEG_INFINITY;
        for &id in &self.lang_ids {
            let v = logits[id as usize];
            if v.is_finite() && v > best_v {
                best_v = v;
                best = id;
            }
        }
        best
    }

    /// Resuelve un código de idioma (`ctx.language`) al id de su token de idioma.
    /// `"auto"` o cadena vacía → `Ok(None)` (autodetección). Un código ISO válido →
    /// `Ok(Some(id))`. Un código desconocido → error claro (NO se cae a auto en
    /// silencio: el usuario lo pidió explícito y debe enterarse de que es inválido).
    /// Acepta tanto `"es"` como `"<|es|>"`.
    fn resolve_lang(&self, code: &str) -> Result<Option<i64>> {
        let code = code.trim();
        if code.is_empty() || code.eq_ignore_ascii_case("auto") {
            return Ok(None);
        }
        let norm = code
            .trim_start_matches("<|")
            .trim_end_matches("|>")
            .to_ascii_lowercase();
        // `lang_code` es id → código; recorrerlo para el inverso (la tabla es pequeña,
        // ~99 entradas, y esto solo corre una vez por transcripción).
        for (&id, c) in &self.lang_code {
            if *c == norm {
                return Ok(Some(id));
            }
        }
        Err(RosettaError::Asr(format!(
            "idioma '{code}' desconocido para Whisper (usa un código ISO como 'es', 'en', 'fr', o 'auto')"
        )))
    }

    /// Decode greedy con KV-cache. Funde la detección de idioma en el prefill
    /// (opt #5): un único forward de `[SOT]` da el idioma y la cross-KV, y el resto del
    /// prompt `[idioma, transcribe, no_ts]` continúa con `use_cache_branch=true`
    /// reutilizando esa cross-KV. Si `forced_lang` es `Some`, se SALTA la detección por
    /// argmax y se usa ese idioma en el prefill. Devuelve el id de idioma realmente
    /// usado y los tokens de texto (sin eos).
    fn decode(
        &mut self,
        enc_arr: Array3<f32>,
        prompt: &[i64],
        forced_lang: Option<i64>,
    ) -> Result<(i64, Vec<i64>)> {
        // `encoder_hidden_states` se construye UNA vez (movimiento, sin copia en
        // layout contiguo) y se alimenta por vista en todos los pasos (opt #2).
        let enc: DynValue = Value::from_array(enc_arr).map_err(asr_err)?.into_dyn();

        // Prefill (use_cache=false): `[<|startofprev|>, ...prompt..., SOT]` si hay
        // prompt de condicionamiento (E4); si no, solo `[SOT]`. El idioma se lee de
        // los logits de la última posición (SOT) → cross-KV constante poblada.
        let sot = self.cfg.decoder_start_token_id;
        let mut prefill_ids: Vec<i64> = Vec::with_capacity(prompt.len() + 2);
        if !prompt.is_empty() {
            prefill_ids.push(self.gcfg.prev_sot_token_id);
            prefill_ids.extend_from_slice(prompt);
        }
        prefill_ids.push(sot);
        let (logits_sot, dec0, cross) = self.prefill(&prefill_ids, &enc)?;
        // Con idioma forzado se SALTA la detección por argmax (los logits de SOT se
        // siguen calculando porque el prefill también puebla la cross-KV; solo se
        // ignora su argmax de idioma). En "auto" se autodetecta como siempre.
        let lang_id = match forced_lang {
            Some(id) => id,
            None => self.argmax_lang(&logits_sot),
        };

        // Resto del prompt con la cross-KV ya poblada (use_cache_branch=true).
        let no_ts = self.gcfg.no_timestamps_token_id;
        let prompt_rest = [lang_id, self.transcribe_id, no_ts];
        let (mut logits, mut dec) = self.gen_step(&prompt_rest, &enc, &cross, &dec0)?;

        let eos = self.cfg.eos_token_id;
        let max_len = self.cfg.max_target_positions;
        let prompt_len = prefill_ids.len() + prompt_rest.len();

        let mut out: Vec<i64> = Vec::new();
        loop {
            self.suppress(&mut logits, out.is_empty());
            // Si tras suprimir no queda ningún logit finito (todo -inf/NaN), ABORTAR
            // el bloque en vez de seguir generando desde el token espurio 0 (bugs-2).
            let next = argmax(logits.iter().copied()).ok_or_else(|| {
                RosettaError::Asr(
                    "decode: logits sin máximo finito tras suppress (NaN/-inf)".into(),
                )
            })? as i64;
            if next == eos {
                break;
            }
            out.push(next);
            if prompt_len + out.len() >= max_len {
                break;
            }
            let (l, d) = self.gen_step(&[next], &enc, &cross, &dec)?;
            logits = l;
            dec = d;
        }
        Ok((lang_id, out))
    }

    /// Pone a -inf los tokens prohibidos: `suppress_tokens` (siempre),
    /// `begin_suppress_tokens` (solo el primero), y TODOS los de timestamp
    /// (`> no_timestamps_token_id`), ya que v1 transcribe sin timestamps.
    fn suppress(&self, logits: &mut [f32], is_first: bool) {
        for &t in &self.gcfg.suppress_tokens {
            if let Some(x) = logits.get_mut(t as usize) {
                *x = f32::NEG_INFINITY;
            }
        }
        if is_first {
            for &t in &self.gcfg.begin_suppress_tokens {
                if let Some(x) = logits.get_mut(t as usize) {
                    *x = f32::NEG_INFINITY;
                }
            }
        }
        let ts_start = (self.gcfg.no_timestamps_token_id + 1) as usize;
        for x in logits.iter_mut().skip(ts_start) {
            *x = f32::NEG_INFINITY;
        }
    }

    /// Núcleo de la transcripción (mel → encoder → idioma → decode). Separado del
    /// trait para poder reintentarlo tras un fallback de device. `forced_lang` =
    /// idioma a forzar (`None` = autodetectar).
    fn transcribe_inner(
        &mut self,
        pcm: &[f32],
        prompt: &[i64],
        forced_lang: Option<i64>,
    ) -> Result<Transcript> {
        let enc = self.run_encoder(pcm)?;
        let (lang_id, tokens) = self.decode(enc, prompt, forced_lang)?;
        let text = self.tok.decode(&tokens).trim().to_string();

        let duration = pcm.len() as f32 / EXPECTED_SR as f32;
        let language = self
            .lang_code
            .get(&lang_id)
            .cloned()
            .unwrap_or_else(|| "auto".into());

        tracing::debug!(idioma = %language, tokens = tokens.len(), "whisper: bloque transcrito");

        let segment = Segment {
            id: 0,
            start: 0.0,
            end: duration,
            text: text.clone(),
            speaker: None,
            speakers: Vec::new(),
            overlap: false,
            words: Vec::new(),
        };
        Ok(Transcript {
            version: "1.0".into(),
            source: SourceInfo {
                file: String::new(),
                duration_s: duration,
                language,
            },
            model: ModelInfo {
                name: self.name().to_string(),
                device: self.device_label.clone(),
            },
            segments: vec![segment],
            text,
        })
    }

    /// Construye los tokens de prompt (init-prompt + texto previo) para condicionar
    /// el decode (E4), truncados a los últimos `PROMPT_MAX_TOKENS`. Vacío si no hay
    /// nada que inyectar.
    fn build_prompt(&self, ctx: &DecodeCtx) -> Vec<i64> {
        let text = match (ctx.init_prompt.trim(), ctx.prev_text.trim()) {
            ("", "") => return Vec::new(),
            (ip, "") => ip.to_string(),
            ("", pt) => pt.to_string(),
            (ip, pt) => format!("{ip} {pt}"),
        };
        let mut ids = self.tok.encode(&text);
        const PROMPT_MAX_TOKENS: usize = 200;
        if ids.len() > PROMPT_MAX_TOKENS {
            ids = ids.split_off(ids.len() - PROMPT_MAX_TOKENS);
        }
        ids
    }

    /// Transcribe con un prompt de condicionamiento (vacío = sin prompt) y un idioma
    /// opcional a forzar (`None` = autodetectar), con el fallback a CPU si el
    /// acelerador falla en inferencia.
    fn run_with_prompt(
        &mut self,
        pcm: &[f32],
        sample_rate: u32,
        prompt: &[i64],
        forced_lang: Option<i64>,
    ) -> Result<Transcript> {
        if sample_rate != EXPECTED_SR {
            return Err(RosettaError::Asr(format!(
                "Whisper espera {EXPECTED_SR} Hz, recibido {sample_rate}"
            )));
        }
        if pcm.is_empty() {
            return Err(RosettaError::Asr("audio vacío".into()));
        }

        match self.transcribe_inner(pcm, prompt, forced_lang) {
            Ok(t) => Ok(t),
            // El acelerador (p. ej. DirectML) falló en inferencia (el decoder int8
            // cuelga en algunas GPUs). Reconstruir en CPU, anotar el marcador local
            // y reintentar una vez. Si ya estábamos en CPU, propagar el error.
            Err(e) if !self.on_cpu => {
                let was_dml = self.device_label.contains("DirectML");
                tracing::warn!(
                    error = %e,
                    device = %self.device_label,
                    "inferencia Whisper falló en el acelerador; reconstruyendo en CPU y reintentando"
                );
                if was_dml {
                    let _ = std::fs::write(
                        dml_marker(&self.dir),
                        "el decoder int8 de Whisper cuelga DirectML en esta maquina",
                    );
                }
                let (encoder, decoder, primary) =
                    build_sessions(&self.dir, &self.hw, Device::Cpu, self.threads)?;
                self.encoder = encoder;
                self.decoder = decoder;
                self.device_label = primary.label().to_string();
                self.on_cpu = true;
                self.transcribe_inner(pcm, prompt, forced_lang)
            }
            Err(e) => Err(e),
        }
    }
}

impl Engine for WhisperEngine {
    fn name(&self) -> &str {
        "whisper-large-v3-turbo"
    }

    fn transcribe(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Transcript> {
        self.run_with_prompt(pcm, sample_rate, &[], None)
    }

    fn transcribe_ctx(
        &mut self,
        pcm: &[f32],
        sample_rate: u32,
        ctx: &DecodeCtx,
    ) -> Result<Transcript> {
        let prompt = self.build_prompt(ctx);
        // Resuelve el idioma forzado del contexto (error si es un código desconocido;
        // `"auto"`/vacío → `None` → autodetección).
        let forced_lang = self.resolve_lang(&ctx.language)?;
        self.run_with_prompt(pcm, sample_rate, &prompt, forced_lang)
    }
}

/// Nombre del feed de KV de entrada de la capa `l`. `part` ∈ {"decoder.key",
/// "decoder.value", "encoder.key", "encoder.value"} (simplificar-4).
fn past_name(l: usize, part: &str) -> String {
    format!("past_key_values.{l}.{part}")
}

/// Nombre de la salida present-KV de la capa `l`. `side` ∈ {"decoder","encoder"},
/// `part` ∈ {"key","value"} (simplificar-4).
fn present_name(l: usize, side: &str, part: &str) -> String {
    format!("present.{l}.{side}.{part}")
}

/// Logits de la ÚLTIMA posición de la secuencia (`[1, seq, vocab]` → `[vocab]`).
/// Bloque idéntico de `prefill` y `gen_step` (simplificar-2).
fn last_logits(outputs: &ort::session::SessionOutputs) -> Result<Vec<f32>> {
    let (shape, data) = outputs["logits"]
        .try_extract_tensor::<f32>()
        .map_err(|e| RosettaError::Asr(format!("logits: {e}")))?;
    let d = shape.as_ref();
    let (s, vocab) = (d[1] as usize, d[2] as usize);
    Ok(data[(s - 1) * vocab..s * vocab].to_vec())
}

/// Recoge la self-KV del decoder (`present.{l}.decoder.{key,value}`) de las
/// `n_layers` capas. Bloque idéntico de `prefill` y `gen_step` (simplificar-2).
fn collect_present_decoder(
    outputs: &ort::session::SessionOutputs,
    n_layers: usize,
) -> Result<Vec<(Array4<f32>, Array4<f32>)>> {
    let mut dec = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let dk = extract_4d(&outputs[present_name(l, "decoder", "key")], "present dec k")?;
        let dv = extract_4d(
            &outputs[present_name(l, "decoder", "value")],
            "present dec v",
        )?;
        dec.push((dk, dv));
    }
    Ok(dec)
}

/// Construye las sesiones encoder + decoder en `device`. Devuelve también el EP
/// primario del encoder (para la etiqueta de device y decidir el fallback).
fn build_sessions(
    dir: &Path,
    hw: &HwProfile,
    device: Device,
    threads: usize,
) -> Result<(ort::session::Session, ort::session::Session, EpKind)> {
    let (encoder, primary) = ep::build_session(
        &find_first(dir, &["encoder_model_int8.onnx", "encoder_model.onnx"])?,
        hw,
        device,
        threads,
    )?;
    let (decoder, _) = ep::build_session(
        &find_first(
            dir,
            &[
                "decoder_model_merged_int8.onnx",
                "decoder_model_merged.onnx",
            ],
        )?,
        hw,
        device,
        threads,
    )?;
    Ok((encoder, decoder, primary))
}

/// Marcador LOCAL (en el dir del modelo, por máquina): el decoder int8 de Whisper
/// cuelga DirectML en esta GPU → arrancar directo en CPU las próximas veces.
fn dml_marker(dir: &Path) -> PathBuf {
    dir.join(".whisper-dml-decoder-broken")
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let txt = fs::read_to_string(path)
        .map_err(|e| RosettaError::Asr(format!("leer {}: {e}", path.display())))?;
    serde_json::from_str(&txt)
        .map_err(|e| RosettaError::Asr(format!("parsear {}: {e}", path.display())))
}
