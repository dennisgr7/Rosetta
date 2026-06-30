//! Resampleo de señal mono con rubato 3 (resampler FFT, alta calidad).

use audioadapter_buffers::direct::InterleavedSlice;
use rosetta_core::{Result, RosettaError};
use rubato::{Fft, FixedSync, Resampler};

/// Resamplea una señal mono de `sr_in` a `sr_out`. Si coinciden, copia tal cual.
pub fn resample_mono(input: &[f32], sr_in: u32, sr_out: u32) -> Result<Vec<f32>> {
    if sr_in == sr_out || input.is_empty() {
        return Ok(input.to_vec());
    }

    let channels = 1usize;
    let chunk = 1024usize;
    let sub_chunks = 2usize;

    let mut resampler = Fft::<f32>::new(
        sr_in as usize,
        sr_out as usize,
        chunk,
        sub_chunks,
        channels,
        FixedSync::Input,
    )
    .map_err(|e| RosettaError::Audio(format!("init resampler: {e}")))?;

    let in_frames = input.len();
    let ratio = sr_out as f64 / sr_in as f64;
    let out_cap = (in_frames as f64 * ratio) as usize + 2 * chunk;
    let mut out = vec![0.0f32; out_cap];

    let input_adapter = InterleavedSlice::new(input, channels, in_frames)
        .map_err(|e| RosettaError::Audio(format!("input adapter: {e}")))?;
    let mut output_adapter = InterleavedSlice::new_mut(&mut out, channels, out_cap)
        .map_err(|e| RosettaError::Audio(format!("output adapter: {e}")))?;

    // Procesa toda la señal de una vez; devuelve (frames_in, frames_out).
    let (_in_done, out_done) = resampler
        .process_all_into_buffer(&input_adapter, &mut output_adapter, in_frames, None)
        .map_err(|e| RosettaError::Audio(format!("resample: {e}")))?;

    out.truncate(out_done * channels);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Genera un seno de `freq` Hz, `len` muestras, a `sr` Hz de muestreo.
    fn seno(len: usize, freq: f32, sr: u32) -> Vec<f32> {
        let mut v = Vec::with_capacity(len);
        for n in 0..len {
            let t = n as f32 / sr as f32;
            v.push((2.0 * std::f32::consts::PI * freq * t).sin());
        }
        v
    }

    #[test]
    fn identidad_cuando_sr_iguales() {
        let x = seno(4096, 440.0, 16_000);
        let y = resample_mono(&x, 16_000, 16_000).unwrap();
        assert_eq!(x, y, "sr_in==sr_out debe copiar tal cual");
    }

    #[test]
    fn vacio_devuelve_vacio() {
        let y = resample_mono(&[], 44_100, 16_000).unwrap();
        assert!(y.is_empty());
    }

    #[test]
    fn downsample_44100_a_16000() {
        let in_len = 44_100; // 1 s a 44,1 kHz
        let x = seno(in_len, 440.0, 44_100);
        let y = resample_mono(&x, 44_100, 16_000).unwrap();
        let esperado = in_len as f64 * (16_000.0 / 44_100.0);
        // Tolerancia ~3 % por latencia/relleno del resampler FFT.
        let tol = esperado * 0.03 + 2048.0;
        let diff = (y.len() as f64 - esperado).abs();
        assert!(
            diff < tol,
            "downsample: len={} esperado≈{esperado:.0} (diff={diff:.0} > tol={tol:.0})",
            y.len()
        );
        // Acotada: nunca más larga que la entrada al bajar de sr.
        assert!(y.len() < in_len);
    }

    #[test]
    fn upsample_8000_a_16000() {
        let in_len = 8_000; // 1 s a 8 kHz
        let x = seno(in_len, 300.0, 8_000);
        let y = resample_mono(&x, 8_000, 16_000).unwrap();
        let esperado = in_len as f64 * (16_000.0 / 8_000.0); // ×2
        let tol = esperado * 0.03 + 2048.0;
        let diff = (y.len() as f64 - esperado).abs();
        assert!(
            diff < tol,
            "upsample: len={} esperado≈{esperado:.0} (diff={diff:.0} > tol={tol:.0})",
            y.len()
        );
        // Acotada: más larga que la entrada al subir de sr.
        assert!(y.len() > in_len);
    }
}
