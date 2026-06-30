//! Motor ASR Parakeet TDT 0.6B v3 sobre ONNX Runtime (CPU en F2).
//!
//! Pipeline: PCM f32 16 kHz mono → `nemo128.onnx` (preproc) → features `[1,128,T]`
//! → encoder → `[1,1024,T']` → decodificación greedy TDT → tokens → texto.
//! Interfaz de los ONNX verificada con `istupakov/parakeet-tdt-0.6b-v3-onnx`.
//! El bucle TDT está portado de parakeet-rs (MIT/Apache-2.0).

use std::path::Path;

use ndarray::{Array1, Array2, Array3, s};
use ort::value::Value;
use rosetta_accel::{Device, HwProfile, ep};

use rosetta_core::{ModelInfo, Result, RosettaError, Segment, SourceInfo, Transcript, Word};

use crate::Engine;
use crate::ort_util::{argmax, asr_err, extract_3d, find_first};
use crate::tokenizer::Vocabulary;

/// Frecuencia esperada por el preprocesador NeMo.
const EXPECTED_SR: u32 = 16_000;
/// Segundos por frame del encoder (subsampling 8 × hop 10 ms = 80 ms).
const FRAME_SECONDS: f32 = 0.08;
/// Dimensión oculta de la prediction network (estados LSTM `[2,1,640]`).
const PRED_HIDDEN: usize = 640;
/// Máximo de símbolos emitidos por frame (anti-bucle).
const MAX_SYMBOLS_PER_STEP: usize = 10;

/// Motor Parakeet TDT.
pub struct ParakeetEngine {
    preproc: ort::session::Session,
    encoder: ort::session::Session,
    decoder: ort::session::Session,
    vocab: Vocabulary,
    vocab_size: usize,
    blank_id: usize,
    device_label: String,
}

impl ParakeetEngine {
    /// Carga el motor desde un directorio con `nemo128.onnx`, encoder,
    /// decoder_joint y `vocab.txt`.
    pub fn from_dir(dir: &Path, hw: &HwProfile, device: Device, threads: usize) -> Result<Self> {
        let (preproc, _) = ep::build_session(&dir.join("nemo128.onnx"), hw, device, threads)?;
        let (encoder, primary) = ep::build_session(
            &find_first(
                dir,
                &[
                    "encoder-model.int8.onnx",
                    "encoder-model.onnx",
                    "encoder.onnx",
                ],
            )?,
            hw,
            device,
            threads,
        )?;
        let (decoder, _) = ep::build_session(
            &find_first(
                dir,
                &[
                    "decoder_joint-model.int8.onnx",
                    "decoder_joint-model.onnx",
                    "decoder_joint.onnx",
                ],
            )?,
            hw,
            device,
            threads,
        )?;
        let vocab = Vocabulary::from_file(&dir.join("vocab.txt"))?;
        let vocab_size = vocab.size();
        let blank_id = vocab.blank_id;
        Ok(Self {
            preproc,
            encoder,
            decoder,
            vocab,
            vocab_size,
            blank_id,
            device_label: primary.label().to_string(),
        })
    }

    /// PCM mono → features `[1,128,T]` vía `nemo128.onnx`.
    fn preprocess(&mut self, pcm: &[f32]) -> Result<Array3<f32>> {
        let n = pcm.len();
        let waveforms = Array2::<f32>::from_shape_vec((1, n), pcm.to_vec())
            .map_err(|e| RosettaError::Asr(format!("waveforms: {e}")))?;
        let lens = Array1::<i64>::from_vec(vec![n as i64]);
        let outputs = self
            .preproc
            .run(ort::inputs!(
                "waveforms" => Value::from_array(waveforms).map_err(asr_err)?,
                "waveforms_lens" => Value::from_array(lens).map_err(asr_err)?
            ))
            .map_err(|e| RosettaError::Asr(format!("preproc run: {e}")))?;
        extract_3d(&outputs["features"], "features")
    }

    /// features `[1,128,T]` → salidas del encoder `[1,1024,T']`.
    fn run_encoder(&mut self, features: Array3<f32>) -> Result<Array3<f32>> {
        let t = features.shape()[2] as i64;
        let length = Array1::<i64>::from_vec(vec![t]);
        let outputs = self
            .encoder
            .run(ort::inputs!(
                "audio_signal" => Value::from_array(features).map_err(asr_err)?,
                "length" => Value::from_array(length).map_err(asr_err)?
            ))
            .map_err(|e| RosettaError::Asr(format!("encoder run: {e}")))?;
        extract_3d(&outputs["outputs"], "encoder outputs")
    }

    /// Decodificación greedy TDT. Devuelve (token_ids, frame_idx por token).
    fn greedy_decode(&mut self, enc: &Array3<f32>) -> Result<(Vec<usize>, Vec<usize>)> {
        let encoder_dim = enc.shape()[1];
        let time_steps = enc.shape()[2];
        let blank_id = self.blank_id;
        let vocab_size = self.vocab_size;

        let mut state_h = Array3::<f32>::zeros((2, 1, PRED_HIDDEN));
        let mut state_c = Array3::<f32>::zeros((2, 1, PRED_HIDDEN));

        let mut tokens = Vec::new();
        let mut frames = Vec::new();
        let mut t = 0usize;
        let mut emitted = 0usize;
        let mut last_token = blank_id as i32;

        while t < time_steps {
            let frame = enc
                .slice(s![0, .., t])
                .to_shape((1, encoder_dim, 1))
                .map_err(|e| RosettaError::Asr(format!("reshape frame: {e}")))?
                .to_owned();
            let targets = Array2::<i32>::from_shape_vec((1, 1), vec![last_token])
                .map_err(|e| RosettaError::Asr(format!("targets: {e}")))?;

            let outputs = self
                .decoder
                .run(ort::inputs!(
                    "encoder_outputs" => Value::from_array(frame).map_err(asr_err)?,
                    "targets" => Value::from_array(targets).map_err(asr_err)?,
                    "target_length" => Value::from_array(Array1::<i32>::from_vec(vec![1])).map_err(asr_err)?,
                    "input_states_1" => Value::from_array(state_h.clone()).map_err(asr_err)?,
                    "input_states_2" => Value::from_array(state_c.clone()).map_err(asr_err)?
                ))
                .map_err(|e| RosettaError::Asr(format!("decoder run: {e}")))?;

            let (_, logits) = outputs["outputs"]
                .try_extract_tensor::<f32>()
                .map_err(|e| RosettaError::Asr(format!("logits: {e}")))?;

            // Las logits del joint TDT nunca son degeneradas en operación normal;
            // pero si el argmax no halla máximo finito (slice vacío o todo-NaN/-inf)
            // abortamos el paso en vez de emitir el token espurio 0 (bugs-2).
            let token_id = argmax(logits.iter().take(vocab_size).copied()).ok_or_else(|| {
                RosettaError::Asr("argmax token: logits sin máximo finito (NaN/-inf)".into())
            })?;
            let duration_step =
                argmax(logits.iter().skip(vocab_size).copied()).ok_or_else(|| {
                    RosettaError::Asr("argmax duración: logits sin máximo finito (NaN/-inf)".into())
                })?;

            if token_id != blank_id {
                state_h = extract_3d(&outputs["output_states_1"], "state_h")?;
                state_c = extract_3d(&outputs["output_states_2"], "state_c")?;
                tokens.push(token_id);
                frames.push(t);
                last_token = token_id as i32;
                emitted += 1;
            }

            if duration_step > 0 {
                t += duration_step;
                emitted = 0;
            } else if token_id == blank_id || emitted >= MAX_SYMBOLS_PER_STEP {
                t += 1;
                emitted = 0;
            }
        }

        Ok((tokens, frames))
    }
}

impl Engine for ParakeetEngine {
    fn name(&self) -> &str {
        "parakeet-tdt-0.6b-v3"
    }

    fn transcribe(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Transcript> {
        if sample_rate != EXPECTED_SR {
            return Err(RosettaError::Asr(format!(
                "Parakeet espera {EXPECTED_SR} Hz, recibido {sample_rate}"
            )));
        }
        if pcm.is_empty() {
            return Err(RosettaError::Asr("audio vacío".into()));
        }

        let features = self.preprocess(pcm)?;
        let enc = self.run_encoder(features)?;
        let (tokens, frames) = self.greedy_decode(&enc)?;

        let text = self.vocab.decode(&tokens);
        let duration = pcm.len() as f32 / sample_rate as f32;
        let start = frames.first().map_or(0.0, |&f| f as f32 * FRAME_SECONDS);
        let end = frames
            .last()
            .map_or(duration, |&f| (f as f32 * FRAME_SECONDS).min(duration))
            .max(start);

        let words: Vec<Word> = self
            .vocab
            .decode_words(&tokens, &frames)
            .into_iter()
            .map(|(text, f0, f1)| Word {
                text,
                start: f0 as f32 * FRAME_SECONDS,
                end: ((f1 + 1) as f32 * FRAME_SECONDS).min(duration),
                confidence: None,
            })
            .collect();

        let segment = Segment {
            id: 0,
            start,
            end,
            text: text.clone(),
            speaker: None,
            speakers: Vec::new(),
            overlap: false,
            words,
        };

        Ok(Transcript {
            version: "1.0".into(),
            source: SourceInfo {
                file: String::new(),
                duration_s: duration,
                language: "auto".into(),
            },
            model: ModelInfo {
                name: self.name().to_string(),
                device: self.device_label.clone(),
            },
            segments: vec![segment],
            text,
        })
    }
}
