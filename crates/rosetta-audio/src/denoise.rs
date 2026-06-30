//! Realce/denoise de voz pre-ASR con GTCRN (ONNX) sobre `ep::build_session`.
//!
//! GTCRN es streaming frame-a-frame: STFT (`n_fft=512`, `hop=256`, raíz-de-Hann)
//! → por cada frame `mix[1,257,1,2]` + 3 caches recurrentes (a cero al empezar)
//! → espectro realzado `enh[1,257,1,2]` + caches actualizados → ISTFT. Es una
//! sola pasada secuencial sobre todo el audio (modo "offline").

use std::path::Path;

use ndarray::{Array, ArrayD, IxDyn};
use ort::session::Session;
use ort::value::Value;
use realfft::num_complex::Complex;

use rosetta_accel::{Device, HwProfile};
use rosetta_core::{Result, RosettaError};

use crate::AudioPcm;
use crate::stft::Stft;

const N_FFT: usize = 512;
const HOP: usize = 256;
const BINS: usize = 257;

/// Limpieza/realce de voz aplicado antes del ASR.
pub trait Denoiser {
    /// Procesa audio mono 16 kHz y devuelve el audio realzado (mismo `sample_rate`).
    fn process(&mut self, pcm: &AudioPcm) -> Result<AudioPcm>;
}

/// Denoiser GTCRN (ONNX), ejecutado por la cascada DirectML→CPU.
pub struct GtcrnDenoiser {
    session: Session,
    stft: Stft,
}

impl GtcrnDenoiser {
    /// Carga `gtcrn_simple.onnx` desde disco.
    pub fn from_file(model: &Path, hw: &HwProfile, device: Device, threads: usize) -> Result<Self> {
        let (session, _ep) = rosetta_accel::ep::build_session(model, hw, device, threads)?;
        Ok(Self {
            session,
            stft: Stft::new(N_FFT, HOP),
        })
    }
}

fn den_err<E: std::fmt::Display>(e: E) -> RosettaError {
    RosettaError::Audio(format!("denoise GTCRN: {e}"))
}

fn to_arrayd(shape: &[i64], data: &[f32]) -> Result<ArrayD<f32>> {
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec()).map_err(den_err)
}

impl Denoiser for GtcrnDenoiser {
    fn process(&mut self, pcm: &AudioPcm) -> Result<AudioPcm> {
        let frames = self.stft.forward(&pcm.samples);
        let mut conv = ArrayD::<f32>::zeros(IxDyn(&[2, 1, 16, 16, 33]));
        let mut tra = ArrayD::<f32>::zeros(IxDyn(&[2, 3, 1, 1, 16]));
        let mut inter = ArrayD::<f32>::zeros(IxDyn(&[2, 1, 33, 16]));
        let mut enhanced: Vec<Vec<Complex<f32>>> = Vec::with_capacity(frames.len());

        for frame in &frames {
            let mut mix = Array::<f32, _>::zeros((1, BINS, 1, 2));
            for (b, c) in frame.iter().enumerate() {
                mix[[0, b, 0, 0]] = c.re;
                mix[[0, b, 0, 1]] = c.im;
            }

            let outputs = self
                .session
                .run(ort::inputs!(
                    "mix" => Value::from_array(mix).map_err(den_err)?,
                    "conv_cache" => Value::from_array(conv.clone()).map_err(den_err)?,
                    "tra_cache" => Value::from_array(tra.clone()).map_err(den_err)?,
                    "inter_cache" => Value::from_array(inter.clone()).map_err(den_err)?
                ))
                .map_err(den_err)?;

            let enh_v = outputs
                .get("enh")
                .ok_or_else(|| den_err("salida 'enh' ausente"))?;
            let (_s, enh) = enh_v.try_extract_tensor::<f32>().map_err(den_err)?;
            if enh.len() < BINS * 2 {
                return Err(den_err(format!(
                    "salida 'enh' demasiado corta: {} (esperado >= {})",
                    enh.len(),
                    BINS * 2
                )));
            }
            let ef: Vec<Complex<f32>> = enh
                .chunks_exact(2)
                .take(BINS)
                .map(|c| Complex::new(c[0], c[1]))
                .collect();
            enhanced.push(ef);

            // Extracción común de los 3 caches de salida (mismo patrón triplicado):
            // try_extract_tensor + to_arrayd. Un closure evita la repetición sin
            // cambiar el comportamiento numérico ni el orden de los caches.
            let take = |name: &str| -> Result<ArrayD<f32>> {
                let (shape, data) = outputs[name].try_extract_tensor::<f32>().map_err(den_err)?;
                to_arrayd(shape.as_ref(), data)
            };
            conv = take("conv_cache_out")?;
            tra = take("tra_cache_out")?;
            inter = take("inter_cache_out")?;
        }

        let samples = self.stft.inverse(&enhanced, pcm.samples.len());
        Ok(AudioPcm {
            samples,
            sample_rate: pcm.sample_rate,
        })
    }
}
