//! Frontend log-mel de Whisper (Bloque D), Rust puro (sin C).
//!
//! Réplica de `transformers.WhisperFeatureExtractor`: STFT con `n_fft=400`,
//! `hop=160`, ventana Hann periódica y `center=True` (reflect-pad de `n_fft/2`);
//! espectrograma de potencia (`|X|^2`); banco mel **Slaney** (`htk=false`,
//! `fmin=0`, `fmax=8000`, normalización Slaney); `log10`, recorte al **máximo
//! global − 8** y normalización `(x+4)/4`. Salida `[n_mels, 3000]` (audio
//! recortado/rellenado a 30 s a 16 kHz). Validado contra el contrato del encoder
//! ONNX (`input_features [1,128,3000]`).

use ndarray::Array2;
use realfft::RealToComplex;
use std::sync::Arc;

const N_FFT: usize = 400;
const HOP: usize = 160;
/// Frecuencia de muestreo que espera Whisper.
pub const SAMPLE_RATE: usize = 16_000;
/// Muestras en una ventana de 30 s (chunk de Whisper).
pub const N_SAMPLES: usize = SAMPLE_RATE * 30; // 480000
/// Frames de salida del encoder (30 s / hop).
pub const N_FRAMES: usize = N_SAMPLES / HOP; // 3000

/// Extractor de features log-mel. Construye una vez el banco de filtros, la
/// ventana y el plan FFT; reutilizable para todos los bloques de audio.
pub struct MelFrontend {
    n_mels: usize,
    /// Banco mel en orden por bin de frecuencia: `filters[f * n_mels + m]`.
    filters: Vec<f32>,
    /// Ventana Hann periódica de `N_FFT` muestras.
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl MelFrontend {
    /// Construye el frontend para `n_mels` bandas (128 en large-v3/turbo).
    pub fn new(n_mels: usize) -> Self {
        let window = hann_periodic(N_FFT);
        let filters = mel_filterbank_slaney(N_FFT / 2 + 1, n_mels, SAMPLE_RATE as f64);
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        Self {
            n_mels,
            filters,
            window,
            fft,
        }
    }

    /// PCM mono a 16 kHz → log-mel `[n_mels, 3000]`.
    pub fn log_mel(&self, pcm: &[f32]) -> Array2<f32> {
        let n_freq = N_FFT / 2 + 1;
        let pad = N_FFT / 2;

        // 1) Recortar/rellenar a 30 s, luego reflect-pad de `pad` a cada lado
        //    (equivale a `torch.stft(center=True, pad_mode='reflect')`).
        let mut sig = vec![0.0f32; N_SAMPLES];
        let n = pcm.len().min(N_SAMPLES);
        sig[..n].copy_from_slice(&pcm[..n]);
        let padded = reflect_pad(&sig, pad);

        // 2) STFT + potencia + proyección mel. Tomamos exactamente N_FRAMES
        //    (3000) frames, que equivale a descartar el último frame (3001 -> 3000)
        //    como hace el feature extractor de HF.
        let mut mel = Array2::<f32>::zeros((self.n_mels, N_FRAMES));
        let mut input = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let mut power = vec![0.0f32; n_freq];

        for t in 0..N_FRAMES {
            let start = t * HOP;
            for i in 0..N_FFT {
                input[i] = padded[start + i] * self.window[i];
            }
            self.fft
                .process_with_scratch(&mut input, &mut spectrum, &mut scratch)
                .expect("FFT real de tamaño fijo");
            for (f, c) in spectrum.iter().enumerate() {
                power[f] = c.re * c.re + c.im * c.im; // power=2.0
            }
            for m in 0..self.n_mels {
                let mut acc = 0.0f32;
                for (f, &p) in power.iter().enumerate() {
                    acc += self.filters[f * self.n_mels + m] * p;
                }
                mel[[m, t]] = acc;
            }
        }

        // 3) log10 con suelo 1e-10, recorte al máximo GLOBAL − 8, y (x+4)/4.
        let mut max_val = f32::NEG_INFINITY;
        for v in mel.iter_mut() {
            *v = v.max(1e-10).log10();
            if *v > max_val {
                max_val = *v;
            }
        }
        let floor = max_val - 8.0;
        for v in mel.iter_mut() {
            *v = (v.max(floor) + 4.0) / 4.0;
        }
        mel
    }
}

/// Reflect-pad de `pad` muestras a cada lado (sin repetir el borde), como
/// `torch.nn.functional.pad(mode='reflect')`.
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in 0..pad {
        out.push(x[pad - i]); // refleja alrededor de x[0]
    }
    out.extend_from_slice(x);
    for i in 0..pad {
        out.push(x[n - 2 - i]); // refleja alrededor de x[n-1]
    }
    out
}

/// Ventana Hann periódica de `n` muestras: `0.5 - 0.5*cos(2πi/n)`.
fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos();
            w as f32
        })
        .collect()
}

/// Hz → mel en la escala Slaney (htk=false), como librosa/transformers.
fn hz_to_mel_slaney(freq: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp; // 15.0
    let logstep = (6.4f64).ln() / 27.0;
    if freq >= min_log_hz {
        min_log_mel + (freq / min_log_hz).ln() / logstep
    } else {
        freq / f_sp
    }
}

/// mel → Hz en la escala Slaney (htk=false).
fn mel_to_hz_slaney(mel: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp; // 15.0
    let logstep = (6.4f64).ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_sp * mel
    }
}

/// Banco de filtros mel triangular (Slaney) en orden `[f * n_mels + m]`.
/// `n_freq = N_FFT/2+1` bins de frecuencia, `[0, sr/2]`.
fn mel_filterbank_slaney(n_freq: usize, n_mels: usize, sr: f64) -> Vec<f32> {
    // Frecuencias de cada bin FFT: linspace(0, sr/2, n_freq).
    let nyq = sr / 2.0;
    let fft_freqs: Vec<f64> = (0..n_freq)
        .map(|f| f as f64 * nyq / (n_freq - 1) as f64)
        .collect();

    // Puntos mel equiespaciados (n_mels+2) y su conversión a Hz.
    let mel_min = hz_to_mel_slaney(0.0);
    let mel_max = hz_to_mel_slaney(nyq);
    let filter_freqs: Vec<f64> = (0..n_mels + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64;
            mel_to_hz_slaney(mel)
        })
        .collect();

    let mut filters = vec![0.0f32; n_freq * n_mels];
    for f in 0..n_freq {
        for m in 0..n_mels {
            let left = filter_freqs[m];
            let center = filter_freqs[m + 1];
            let right = filter_freqs[m + 2];
            let down = (fft_freqs[f] - left) / (center - left);
            let up = (right - fft_freqs[f]) / (right - center);
            let mut w = down.min(up).max(0.0);
            // Normalización Slaney: área constante por filtro.
            let enorm = 2.0 / (right - left);
            w *= enorm;
            filters[f * n_mels + m] = w as f32;
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forma_y_rango() {
        let mel = MelFrontend::new(128);
        // 2 s de tono a 440 Hz.
        let pcm: Vec<f32> = (0..32_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin() * 0.5)
            .collect();
        let out = mel.log_mel(&pcm);
        assert_eq!(out.shape(), &[128, 3000]);
        // Tras (x+4)/4 con recorte a max-8: valores en ~[-1, max/4+1].
        let mn = out.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(mn >= -1.001, "min mel {mn}");
        assert!(mx <= 1.5, "max mel {mx}");
        // El recorte fija el suelo en (max-8+4)/4 = (max-4)/4 → diferencia 2.0.
        assert!((mx - mn - 2.0).abs() < 0.5, "rango dinámico {} ", mx - mn);
    }

    #[test]
    fn tono_concentra_energia() {
        // Un tono de 1 kHz debe activar más una banda mel concreta que el silencio.
        let mel = MelFrontend::new(128);
        let tono: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16_000.0).sin() * 0.5)
            .collect();
        let silencio = vec![0.0f32; 16_000];
        let a = mel.log_mel(&tono);
        let b = mel.log_mel(&silencio);
        // Energía total del tono > energía del silencio.
        let sa: f32 = a.iter().sum();
        let sb: f32 = b.iter().sum();
        assert!(sa > sb, "tono {sa} debería superar silencio {sb}");
    }

    #[test]
    fn filtros_slaney_no_negativos_y_picos() {
        let f = mel_filterbank_slaney(201, 128, 16_000.0);
        assert_eq!(f.len(), 201 * 128);
        assert!(f.iter().all(|&w| w >= 0.0));
        // Cada filtro (columna m) debe tener al menos un peso > 0.
        for m in 0..128 {
            let any = (0..201).any(|fr| f[fr * 128 + m] > 0.0);
            assert!(any, "filtro mel {m} vacío");
        }
    }
}
