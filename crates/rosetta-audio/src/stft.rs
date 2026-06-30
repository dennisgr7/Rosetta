//! STFT / ISTFT con ventana raíz-de-Hann para los modelos de realce de voz
//! (GTCRN: `n_fft=512`, `hop=256`, `center=True`). Rust puro sobre `realfft`.
//!
//! Análisis y síntesis usan la misma ventana raíz-de-Hann; la reconstrucción es
//! WOLA (weighted overlap-add) normalizada por la suma de ventanas², lo que da
//! reconstrucción perfecta con solape del 50 %.

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// Transformada tiempo-frecuencia configurable.
pub struct Stft {
    n_fft: usize,
    hop: usize,
    window: Vec<f32>,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
}

impl Stft {
    /// Crea una STFT con `n_fft`/`hop` dados y ventana raíz-de-Hann periódica.
    pub fn new(n_fft: usize, hop: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(n_fft);
        let inv = planner.plan_fft_inverse(n_fft);
        Self {
            n_fft,
            hop,
            window: sqrt_hann(n_fft),
            fwd,
            inv,
        }
    }

    /// Nº de bins de frecuencia (`n_fft/2 + 1`).
    pub fn num_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// STFT con `center=True` (pad reflect de `n_fft/2`). Devuelve un frame por
    /// columna: `frames[t][bin]` complejo.
    pub fn forward(&self, signal: &[f32]) -> Vec<Vec<Complex<f32>>> {
        let pad = self.n_fft / 2;
        let padded = reflect_pad(signal, pad);
        let mut input = self.fwd.make_input_vec();
        let mut spectrum = self.fwd.make_output_vec();
        let mut frames = Vec::new();
        let mut pos = 0;
        while pos + self.n_fft <= padded.len() {
            for (i, s) in input.iter_mut().enumerate() {
                *s = padded[pos + i] * self.window[i];
            }
            self.fwd
                .process(&mut input, &mut spectrum)
                .expect("rfft forward");
            frames.push(spectrum.clone());
            pos += self.hop;
        }
        frames
    }

    /// ISTFT (WOLA) de los frames complejos; recorta el padding de `center` y
    /// devuelve exactamente `original_len` muestras.
    pub fn inverse(&self, frames: &[Vec<Complex<f32>>], original_len: usize) -> Vec<f32> {
        let pad = self.n_fft / 2;
        let out_len = frames.len().saturating_sub(1) * self.hop + self.n_fft;
        let mut y = vec![0.0f32; out_len];
        let mut norm = vec![0.0f32; out_len];
        let mut spectrum = self.inv.make_input_vec();
        let mut output = self.inv.make_output_vec();
        let last = self.num_bins() - 1;
        let mut pos = 0;
        for frame in frames {
            spectrum.copy_from_slice(frame);
            // `realfft` exige parte imaginaria nula en DC y Nyquist.
            spectrum[0].im = 0.0;
            spectrum[last].im = 0.0;
            self.inv
                .process(&mut spectrum, &mut output)
                .expect("rfft inverse");
            for i in 0..self.n_fft {
                let v = output[i] / self.n_fft as f32 * self.window[i];
                y[pos + i] += v;
                norm[pos + i] += self.window[i] * self.window[i];
            }
            pos += self.hop;
        }
        for (yi, ni) in y.iter_mut().zip(norm.iter()) {
            if *ni > 1e-8 {
                *yi /= *ni;
            }
        }
        let start = pad.min(y.len());
        let end = (start + original_len).min(y.len());
        let mut out = y[start..end].to_vec();
        out.resize(original_len, 0.0);
        out
    }
}

/// Ventana raíz-de-Hann periódica (`sqrt(hann)`), como `torch.hann_window(n)**0.5`.
fn sqrt_hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let hann = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos();
            hann.sqrt()
        })
        .collect()
}

/// Padding por reflexión (sin repetir el borde), como `torch.stft(center=True)`.
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return vec![0.0; 2 * pad];
    }
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in 0..pad {
        // x[pad], x[pad-1], ..., x[1]  (con clamp para señales cortas)
        let idx = (pad - i).min(n - 1);
        out.push(x[idx]);
    }
    out.extend_from_slice(x);
    for i in 0..pad {
        // x[n-2], x[n-3], ..., x[n-1-pad]
        let idx = (n.saturating_sub(2 + i)).min(n - 1);
        out.push(x[idx]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_roundtrip_recovers_signal() {
        // Señal de prueba: suma de senos.
        let sr = 16_000.0f32;
        let n = 8000;
        let sig: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                0.5 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
            })
            .collect();

        let stft = Stft::new(512, 256);
        let frames = stft.forward(&sig);
        let rec = stft.inverse(&frames, sig.len());

        assert_eq!(rec.len(), sig.len());
        // Error medio cuadrático pequeño (la zona central debe reconstruir casi exacto).
        let mut err = 0.0f64;
        let mut energy = 0.0f64;
        for (a, b) in sig.iter().zip(rec.iter()).skip(512).take(n - 1024) {
            err += ((a - b) as f64).powi(2);
            energy += (*a as f64).powi(2);
        }
        let rel = (err / energy).sqrt();
        assert!(rel < 1e-3, "reconstrucción STFT/ISTFT imprecisa: rel={rel}");
    }
}
