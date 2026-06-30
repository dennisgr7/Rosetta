//! Detección de actividad de voz (VAD) con Silero (ONNX), corrido por
//! `ep::build_session` en CPU (modelo de ~2 MB; ventanas de 512 muestras).
//!
//! Silero es recurrente: por cada ventana de 512 muestras a 16 kHz toma
//! `input[1,512]` + `state[2,1,128]` + `sr` y devuelve la probabilidad de voz y
//! el estado actualizado. El estado se realimenta y se resetea por archivo. Sobre
//! la secuencia de probabilidades se aplica una histéresis para obtener los
//! intervalos de voz.

use std::path::Path;

use ndarray::{Array2, ArrayD, IxDyn, arr0};
use ort::session::Session;
use ort::value::Value;

use rosetta_accel::{Device, HwProfile};
use rosetta_core::{Result, RosettaError};

use crate::AudioPcm;

const WINDOW: usize = 512;
const CONTEXT: usize = 64;
const SR: i64 = 16_000;

/// Intervalo de voz detectado (segundos).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeechSegment {
    pub start_s: f32,
    pub end_s: f32,
}

/// Parámetros de la histéresis del VAD.
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    pub threshold: f32,
    pub neg_threshold: f32,
    pub min_speech_ms: u32,
    pub min_silence_ms: u32,
    pub speech_pad_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            neg_threshold: 0.35,
            min_speech_ms: 250,
            min_silence_ms: 100,
            speech_pad_ms: 30,
        }
    }
}

/// VAD Silero cargado en una sesión `ort`.
pub struct SileroVad {
    session: Session,
    cfg: VadConfig,
}

impl SileroVad {
    /// Carga `silero_vad.onnx` (forzado a CPU: las ventanas de 512 no rentan GPU).
    pub fn from_file(model: &Path, hw: &HwProfile, threads: usize) -> Result<Self> {
        let (session, _ep) = rosetta_accel::ep::build_session(model, hw, Device::Cpu, threads)?;
        Ok(Self {
            session,
            cfg: VadConfig::default(),
        })
    }

    /// Cambia la configuración de histéresis.
    pub fn with_config(mut self, cfg: VadConfig) -> Self {
        self.cfg = cfg;
        self
    }

    /// Probabilidad de voz por ventana de 512 muestras (estado reseteado).
    fn speech_probs(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        let mut state = ArrayD::<f32>::zeros(IxDyn(&[2, 1, 128]));
        let mut context = vec![0.0f32; CONTEXT];
        let mut probs = Vec::with_capacity(pcm.len() / WINDOW + 1);
        let mut pos = 0;
        while pos < pcm.len() {
            let mut chunk = vec![0.0f32; WINDOW];
            let n = (pcm.len() - pos).min(WINDOW);
            chunk[..n].copy_from_slice(&pcm[pos..pos + n]);
            // Silero v5/v6 procesa contexto(64) + ventana(512) = 576 muestras.
            let mut input_vec = Vec::with_capacity(CONTEXT + WINDOW);
            input_vec.extend_from_slice(&context);
            input_vec.extend_from_slice(&chunk);
            context = chunk[WINDOW - CONTEXT..].to_vec();
            let input =
                Array2::<f32>::from_shape_vec((1, CONTEXT + WINDOW), input_vec).map_err(vad_err)?;

            let outputs = self
                .session
                .run(ort::inputs!(
                    "input" => Value::from_array(input).map_err(vad_err)?,
                    "state" => Value::from_array(state.clone()).map_err(vad_err)?,
                    "sr" => Value::from_array(arr0(SR)).map_err(vad_err)?
                ))
                .map_err(vad_err)?;

            let out = outputs
                .get("output")
                .ok_or_else(|| vad_err("salida 'output' ausente"))?;
            let (_s, prob) = out.try_extract_tensor::<f32>().map_err(vad_err)?;
            probs.push(prob.first().copied().unwrap_or(0.0));

            let state_out = outputs
                .get("stateN")
                .ok_or_else(|| vad_err("salida 'stateN' ausente"))?;
            let (ss, sd) = state_out.try_extract_tensor::<f32>().map_err(vad_err)?;
            let dims: Vec<usize> = ss.as_ref().iter().map(|&d| d as usize).collect();
            state = ArrayD::from_shape_vec(IxDyn(&dims), sd.to_vec()).map_err(vad_err)?;

            pos += WINDOW;
        }
        Ok(probs)
    }

    /// Detecta los intervalos de voz del audio (mono 16 kHz).
    pub fn detect(&mut self, audio: &AudioPcm) -> Result<Vec<SpeechSegment>> {
        let probs = self.speech_probs(&audio.samples)?;
        Ok(self.hysteresis(&probs, audio.sample_rate))
    }

    fn hysteresis(&self, probs: &[f32], sr: u32) -> Vec<SpeechSegment> {
        hysteresis_probs(&self.cfg, probs, sr)
    }
}

/// Histéresis pura sobre la secuencia de probabilidades (sin sesión ONNX, testeable).
///
/// Convierte la secuencia de probabilidades por ventana en intervalos de voz
/// aplicando los umbrales y los mínimos de `cfg`. Aislada de `SileroVad` para
/// poder testear la lógica con probabilidades sintéticas.
fn hysteresis_probs(cfg: &VadConfig, probs: &[f32], sr: u32) -> Vec<SpeechSegment> {
    let win_s = WINDOW as f32 / sr as f32;
    let pad = cfg.speech_pad_ms as f32 / 1000.0;
    let min_speech = cfg.min_speech_ms as f32 / 1000.0;
    let min_silence = cfg.min_silence_ms as f32 / 1000.0;

    // Fin del audio en segundos: cota superior para el padding de cierre.
    let audio_end = probs.len() as f32 * win_s;

    let mut segments = Vec::new();
    let mut triggered = false;
    let mut start = 0.0f32;
    // Inicio del silencio en curso. `None` = no hay silencio en curso
    // (centinela explícito; el antiguo 0.0 colisionaba con un silencio en t=0).
    let mut temp_end: Option<f32> = None;

    for (i, &p) in probs.iter().enumerate() {
        let t = i as f32 * win_s;
        if p >= cfg.threshold {
            if !triggered {
                triggered = true;
                start = t;
            }
            temp_end = None;
        } else if triggered && p < cfg.neg_threshold {
            let silence_start = *temp_end.get_or_insert(t);
            if t - silence_start >= min_silence {
                if silence_start - start >= min_speech {
                    segments.push(SpeechSegment {
                        start_s: (start - pad).max(0.0),
                        end_s: silence_start + pad,
                    });
                }
                triggered = false;
                temp_end = None;
            }
        }
    }
    if triggered {
        // Si el audio acaba con un silencio en curso que no alcanzó
        // `min_silence`, cierra el segmento en el inicio de ese silencio
        // (`temp_end`), no al final del audio: así no se arrastra hasta
        // ~min_silence+speech_pad de silencio de cola. Sin silencio
        // pendiente (habla hasta el final), se cierra en el fin del audio.
        let end = temp_end.unwrap_or(audio_end);
        if end - start >= min_speech {
            // El padding de cierre no debe rebasar el fin del audio.
            segments.push(SpeechSegment {
                start_s: (start - pad).max(0.0),
                end_s: (end + pad).min(audio_end),
            });
        }
    }
    segments
}

fn vad_err<E: std::fmt::Display>(e: E) -> RosettaError {
    RosettaError::Audio(format!("VAD Silero: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 16_000;
    // 512/16000 = 0.032 s por ventana.
    const WIN_S: f32 = WINDOW as f32 / SR as f32;

    /// Construye un vector de probabilidades: `voz` ventanas a 0.9 (voz clara)
    /// seguidas de `silencio` ventanas a 0.0 (silencio claro).
    fn probs(voz: usize, silencio: usize) -> Vec<f32> {
        let mut v = vec![0.9f32; voz];
        v.extend(std::iter::repeat_n(0.0f32, silencio));
        v
    }

    /// (a) El audio empieza con silencio en t=0: con el antiguo centinela 0.0,
    /// un silencio en curso en la ventana 0 quedaba indistinguible de "ninguno".
    /// Aquí: silencio inicial + voz larga hasta el final. El segmento debe
    /// empezar donde arranca la voz (no romperse por el centinela), y como la
    /// voz llega hasta el final no hay cola de silencio que recortar.
    #[test]
    fn silencio_en_t0_no_rompe_por_centinela() {
        let cfg = VadConfig::default();
        // 5 ventanas de silencio (0.16 s) + 10 de voz (0.32 s > min_speech 0.25).
        let v = probs(0, 5)
            .into_iter()
            .chain(probs(10, 0))
            .collect::<Vec<f32>>();
        let segs = hysteresis_probs(&cfg, &v, SR);
        assert_eq!(segs.len(), 1, "debe emitir exactamente un segmento de voz");
        // La voz arranca en la ventana 5 → t = 5*WIN_S; menos pad (0.03), clamp a 0.
        let start_voz = 5.0 * WIN_S;
        let pad = 0.03;
        assert!(
            (segs[0].start_s - (start_voz - pad).max(0.0)).abs() < 1e-4,
            "el segmento debe empezar en el inicio de la voz, no en t=0 espurio: {:?}",
            segs[0]
        );
        // Voz hasta el final del audio: fin = audio_end (clamp), sin cola.
        let audio_end = v.len() as f32 * WIN_S;
        assert!(
            (segs[0].end_s - audio_end).abs() < 1e-4,
            "el fin debe quedar acotado al fin del audio: {:?}",
            segs[0]
        );
    }

    /// (b) La voz termina con un silencio en curso que NO alcanza min_silence
    /// antes de que se acabe el audio: el último segmento NO debe incluir la
    /// cola de silencio extra. Antes el cierre usaba `probs.len()*win_s` (fin
    /// del audio), arrastrando hasta ~min_silence+pad de silencio.
    #[test]
    fn cola_de_silencio_al_cierre_se_recorta() {
        let cfg = VadConfig::default();
        // 10 ventanas de voz (0.32 s) + 2 de silencio (0.064 s < min_silence 0.1).
        let mut v = vec![0.9f32; 10];
        v.extend(vec![0.0f32; 2]);
        let segs = hysteresis_probs(&cfg, &v, SR);
        assert_eq!(segs.len(), 1, "debe emitir un único segmento");

        let pad = 0.03;
        // El silencio en curso empieza en la ventana 10 → temp_end = 10*WIN_S.
        let silence_start = 10.0 * WIN_S;
        let esperado_fin = silence_start + pad;
        assert!(
            (segs[0].end_s - esperado_fin).abs() < 1e-4,
            "el fin debe ser el inicio del silencio + pad ({esperado_fin}), no el fin del audio: {:?}",
            segs[0]
        );

        // Comprobación explícita de regresión: el comportamiento ANTIGUO habría
        // cerrado en audio_end + pad. Verificamos que recortamos ~la cola.
        let fin_antiguo = v.len() as f32 * WIN_S + pad;
        assert!(
            segs[0].end_s < fin_antiguo - 1e-4,
            "el nuevo fin debe ser estrictamente menor que el antiguo (cola recortada): nuevo={}, antiguo={}",
            segs[0].end_s,
            fin_antiguo
        );
    }

    /// La voz que llega hasta el final (sin silencio pendiente) conserva la
    /// semántica: se cierra en el fin del audio (con pad acotado al audio).
    #[test]
    fn voz_hasta_el_final_cierra_en_fin_de_audio() {
        let cfg = VadConfig::default();
        let v = vec![0.9f32; 10]; // toda voz, sin silencio de cola.
        let segs = hysteresis_probs(&cfg, &v, SR);
        assert_eq!(segs.len(), 1);
        let audio_end = v.len() as f32 * WIN_S;
        assert!(
            (segs[0].end_s - audio_end).abs() < 1e-4,
            "sin silencio pendiente el cierre es el fin del audio: {:?}",
            segs[0]
        );
    }

    /// Caso común ya gestionado antes del cierre: voz + silencio largo
    /// (>min_silence) + más audio. El segmento se cierra en el inicio del
    /// silencio, no se ve afectado por el bloque de cierre final.
    #[test]
    fn silencio_largo_intermedio_cierra_en_temp_end() {
        let cfg = VadConfig::default();
        // 10 voz + 5 silencio (0.16 s > min_silence) + 10 voz.
        let mut v = vec![0.9f32; 10];
        v.extend(vec![0.0f32; 5]);
        v.extend(vec![0.9f32; 10]);
        let segs = hysteresis_probs(&cfg, &v, SR);
        assert_eq!(
            segs.len(),
            2,
            "dos segmentos separados por el silencio largo"
        );
        let pad = 0.03;
        let primer_fin = 10.0 * WIN_S + pad; // inicio del silencio + pad
        assert!(
            (segs[0].end_s - primer_fin).abs() < 1e-4,
            "el primer segmento cierra en el inicio del silencio: {:?}",
            segs[0]
        );
    }
}
