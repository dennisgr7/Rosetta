//! Motores de reconocimiento de voz (ASR).
//!
//! Implementación en F2+: motor Parakeet TDT 0.6B v3 sobre `ort` (decoder TDT
//! portado de parakeet-rs). En fase 2, motor Whisper-large-v3-turbo.

use std::path::Path;

use rosetta_accel::{Device, HwProfile};
use rosetta_core::{DecodeCtx, Result, RosettaError, Transcript};

mod ort_util;
mod parakeet;
mod tokenizer;
mod whisper;

pub use parakeet::ParakeetEngine;
pub use whisper::WhisperEngine;

/// Interfaz común a todos los motores ASR (Parakeet, Whisper, ...).
pub trait Engine {
    /// Nombre legible del motor.
    fn name(&self) -> &str;

    /// Transcribe PCM mono `f32` a la frecuencia dada y devuelve el transcript.
    fn transcribe(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Transcript>;

    /// Transcribe condicionando al contexto (`init_prompt` + `prev_text`). Por
    /// defecto IGNORA el contexto y delega en [`Engine::transcribe`] (Parakeet);
    /// Whisper lo sobreescribe para inyectar el prompt en el decode (E4).
    fn transcribe_ctx(
        &mut self,
        pcm: &[f32],
        sample_rate: u32,
        _ctx: &DecodeCtx,
    ) -> Result<Transcript> {
        self.transcribe(pcm, sample_rate)
    }
}

/// Construye el motor ASR adecuado para `model` cargando sus pesos desde `dir`.
///
/// Despacha por el nombre del modelo; hoy solo Parakeet, pero centraliza la
/// selección para que añadir Whisper (u otro) no obligue a tocar el binario.
pub fn build_engine(
    model: &str,
    dir: &Path,
    hw: &HwProfile,
    device: Device,
    threads: usize,
) -> Result<Box<dyn Engine>> {
    let id = model.to_ascii_lowercase();
    if id.starts_with("parakeet") {
        Ok(Box::new(ParakeetEngine::from_dir(
            dir, hw, device, threads,
        )?))
    } else if id.starts_with("whisper") {
        Ok(Box::new(WhisperEngine::from_dir(dir, hw, device, threads)?))
    } else {
        Err(RosettaError::Model(format!(
            "motor ASR desconocido: {model}"
        )))
    }
}

#[cfg(test)]
mod tests {
    /// Verifica que `ort` puede cargar el runtime de ONNX Runtime (modo
    /// `load-dynamic`) en esta plataforma. Requiere `ORT_DYLIB_PATH`, definido en
    /// `.cargo/config.toml`.
    /// Carga el preprocesador `nemo128.onnx` real (modelo descargado) para validar
    /// que ort + onnxruntime (load-dynamic) funcionan en esta plataforma. Se ignora
    /// si el modelo aún no está descargado.
    #[test]
    fn ort_carga_modelo_nemo() {
        let models = std::env::var("ROSETTA_MODELS_DIR")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../models").to_string());
        let path = std::path::Path::new(&models)
            .join("parakeet-tdt-0.6b-v3")
            .join("nemo128.onnx");
        if !path.exists() {
            eprintln!("modelo no presente, test omitido");
            return;
        }
        let mut b = ort::session::Session::builder().expect("SessionBuilder");
        b.commit_from_file(path)
            .expect("cargar nemo128.onnx con onnxruntime");
    }
}
