//! Decodificación, resampleo y preprocesado de audio.
//!
//! F1: cualquier formato de audio o vídeo → PCM `f32` mono a 16 kHz. `symphonia`
//! es el decodificador primario (puro Rust) y `ffmpeg-sidecar` el fallback para
//! códecs/contenedores no soportados. VAD (F5) y denoise/realce (F7) se añaden
//! sobre este crate.

mod decode;
mod denoise;
mod resample;
mod stft;
mod vad;

pub use decode::decode_file;
pub use denoise::{Denoiser, GtcrnDenoiser};
pub use resample::resample_mono;
pub use vad::{SileroVad, SpeechSegment, VadConfig};

use rosetta_core::Result;
use std::path::Path;

/// Frecuencia de muestreo objetivo para los modelos ASR.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Audio PCM mono en coma flotante, normalizado a `[-1.0, 1.0]`.
#[derive(Debug, Clone)]
pub struct AudioPcm {
    /// Muestras mono.
    pub samples: Vec<f32>,
    /// Frecuencia de muestreo (Hz).
    pub sample_rate: u32,
}

impl AudioPcm {
    /// Duración en segundos.
    pub fn duration_s(&self) -> f32 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f32 / self.sample_rate as f32
        }
    }
}

/// Decodifica un archivo (audio o vídeo) a PCM mono 16 kHz, probando primero
/// symphonia y cayendo a ffmpeg-sidecar si el formato/códec no está soportado.
pub fn load_audio_16k_mono(path: impl AsRef<Path>) -> Result<AudioPcm> {
    decode_file(path.as_ref(), TARGET_SAMPLE_RATE)
}
