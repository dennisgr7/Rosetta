//! Orquestador de transcripción de audio largo (F5).
//!
//! Parakeet procesa bien tramos de hasta ~30 s; en audios más largos se trocea
//! por voz con el VAD (Silero), se agrupan los tramos de voz en bloques de
//! ≤ `MAX_BLOCK_S` cortando en silencios, se transcribe cada bloque y se
//! recosen reubicando los timestamps a tiempo absoluto. Como los cortes caen en
//! silencio (no parten palabras), los bloques son disjuntos y no hace falta
//! deduplicar solapes.

use rosetta_asr::Engine;
use rosetta_audio::{AudioPcm, SileroVad, SpeechSegment};
use rosetta_core::{DecodeCtx, ModelInfo, Result, SourceInfo, Transcript};

/// Umbral para transcribir de una sola pasada (sin VAD). Es también el punto a
/// partir del cual el CLI carga el VAD, así que se expone para no duplicarlo.
pub const SINGLE_PASS_MAX_S: f32 = 28.0;
/// Duración máxima de un bloque troceado.
const MAX_BLOCK_S: f32 = 28.0;

/// Transcribe un audio, troceándolo por voz si es largo. `vad` se usa solo
/// cuando el audio supera el umbral; si es `None`, siempre va de una pasada.
///
/// Devuelve también los segmentos de voz del VAD (`Some` solo en el camino de
/// audio largo) para que la diarización los reutilice sin re-ejecutar el VAD
/// (opt #3); en el camino de una sola pasada son `None`.
pub fn transcribe(
    engine: &mut dyn Engine,
    audio: &AudioPcm,
    vad: Option<&mut SileroVad>,
    file: String,
    language: String,
    init_prompt: String,
) -> Result<(Transcript, Option<Vec<SpeechSegment>>)> {
    let sr = audio.sample_rate;

    let vad = match vad {
        Some(v) if audio.duration_s() > SINGLE_PASS_MAX_S => v,
        _ => {
            let ctx = DecodeCtx {
                init_prompt,
                prev_text: String::new(),
                language,
            };
            let mut t = engine.transcribe_ctx(&audio.samples, sr, &ctx)?;
            t.source.file = file;
            // METADATO HONESTO: NO se sobrescribe `source.language` con el flag. El
            // engine ya lo fija al idioma REALMENTE usado (el detectado en "auto" o
            // el forzado en Whisper; Parakeet refleja "auto" porque lo ignora).
            return Ok((t, None));
        }
    };

    let speech = vad.detect(audio)?;
    let blocks = chunk_blocks(&speech, audio.duration_s(), MAX_BLOCK_S);
    tracing::info!(bloques = blocks.len(), "audio largo: troceado por voz");

    let mut segments = Vec::new();
    let mut full = String::new();
    let mut device = String::new();
    // Idioma REALMENTE usado, reportado por el engine en el primer bloque (Whisper:
    // el detectado en "auto" o el forzado; Parakeet: "auto"). Arranca con el flag
    // como respaldo por si no hubiese ningún bloque transcrito.
    let mut used_language = language.clone();
    let mut got_language = false;
    let mut next_id = 0;
    // Texto del bloque anterior, para condicionar el siguiente (E4; solo Whisper lo
    // usa, Parakeet ignora el contexto). Se reemplaza por bloque (no se acumula).
    let mut prev_text = String::new();

    for &(bs, be) in &blocks {
        let s0 = (bs * sr as f32) as usize;
        let s1 = ((be * sr as f32) as usize).min(audio.samples.len());
        if s1 <= s0 {
            continue;
        }
        let ctx = DecodeCtx {
            init_prompt: init_prompt.clone(),
            prev_text: std::mem::take(&mut prev_text),
            language: language.clone(),
        };
        let t = engine.transcribe_ctx(&audio.samples[s0..s1], sr, &ctx)?;
        device = t.model.device.clone();
        // El idioma del primer bloque transcrito (Whisper autodetecta una vez y
        // los bloques siguientes lo reflejan igual; con idioma forzado es constante).
        if !got_language {
            used_language = t.source.language.clone();
            got_language = true;
        }
        prev_text = t.text.clone();
        for mut seg in t.segments {
            if seg.text.trim().is_empty() {
                continue;
            }
            seg.id = next_id;
            next_id += 1;
            seg.start += bs;
            seg.end += bs;
            for w in &mut seg.words {
                w.start += bs;
                w.end += bs;
            }
            if !full.is_empty() {
                full.push(' ');
            }
            full.push_str(seg.text.trim());
            segments.push(seg);
        }
    }

    let transcript = Transcript {
        version: "1.0".into(),
        source: SourceInfo {
            file,
            duration_s: audio.duration_s(),
            // METADATO HONESTO: el idioma realmente usado por el engine, no el flag.
            language: used_language,
        },
        model: ModelInfo {
            name: engine.name().to_string(),
            device,
        },
        segments,
        text: full,
    };
    Ok((transcript, Some(speech)))
}

/// Agrupa los tramos de voz en bloques de ≤ `max_block_s`, cortando en los
/// silencios entre tramos. Un tramo individual más largo que el máximo se parte.
fn chunk_blocks(speech: &[SpeechSegment], total_s: f32, max_block_s: f32) -> Vec<(f32, f32)> {
    if speech.is_empty() {
        return vec![(0.0, total_s)];
    }
    let mut grouped: Vec<(f32, f32)> = Vec::new();
    let mut bstart = speech[0].start_s;
    let mut bend = speech[0].end_s;
    for seg in &speech[1..] {
        if seg.end_s - bstart <= max_block_s {
            bend = seg.end_s;
        } else {
            grouped.push((bstart, bend));
            bstart = seg.start_s;
            bend = seg.end_s;
        }
    }
    grouped.push((bstart, bend));

    // Partir cualquier bloque que exceda el máximo (voz continua sin pausas):
    // ningún bloque enviado al encoder debe superar su ventana de diseño.
    let mut out = Vec::new();
    for (s, e) in grouped {
        if e - s <= max_block_s {
            out.push((s, e));
        } else {
            let mut cs = s;
            while cs < e {
                let ce = (cs + max_block_s).min(e);
                out.push((cs, ce));
                cs = ce;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(a: f32, b: f32) -> SpeechSegment {
        SpeechSegment {
            start_s: a,
            end_s: b,
        }
    }

    #[test]
    fn agrupa_en_bloques_por_silencio() {
        let speech = vec![seg(0.0, 10.0), seg(12.0, 20.0), seg(35.0, 50.0)];
        let blocks = chunk_blocks(&speech, 55.0, 28.0);
        // [0-10]+[12-20] caben en un bloque (<=28); [35-50] en otro.
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0], (0.0, 20.0));
        assert_eq!(blocks[1], (35.0, 50.0));
    }

    #[test]
    fn parte_tramo_continuo_muy_largo() {
        let speech = vec![seg(0.0, 60.0)];
        let blocks = chunk_blocks(&speech, 60.0, 28.0);
        assert!(blocks.len() >= 2, "un tramo de 60s debe partirse");
        assert!(blocks.iter().all(|(s, e)| e - s <= 28.0 + 0.01));
    }

    #[test]
    fn parte_bloque_en_ventana_28_42() {
        // Regresión: un tramo continuo de 40 s (28 < 40 <= 42) se colaba ENTERO
        // con el antiguo margen *1.5; ahora ningún bloque supera max_block_s.
        let speech = vec![seg(0.0, 40.0)];
        let blocks = chunk_blocks(&speech, 40.0, 28.0);
        assert!(blocks.len() >= 2, "un tramo de 40s debe partirse");
        assert!(blocks.iter().all(|(s, e)| e - s <= 28.0 + 0.01));
    }

    #[test]
    fn sin_voz_devuelve_bloque_entero() {
        let blocks = chunk_blocks(&[], 40.0, 28.0);
        assert_eq!(blocks, vec![(0.0, 40.0)]);
    }
}
