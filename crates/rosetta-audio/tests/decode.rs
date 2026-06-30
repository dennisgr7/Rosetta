//! Tests de decodificación (F1). Generan WAVs sintéticos con `hound` y verifican
//! la conversión a PCM mono 16 kHz.

use std::path::PathBuf;

use rosetta_audio::load_audio_16k_mono;

fn tmp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("rosetta_{}_{}", std::process::id(), name))
}

fn write_sine_wav(path: &std::path::Path, sr: u32, channels: u16, secs: f32, freq: f32) {
    let spec = hound::WavSpec {
        channels,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    let n = (sr as f32 * secs) as u32;
    for i in 0..n {
        let t = i as f32 / sr as f32;
        let s = (2.0 * std::f32::consts::PI * freq * t).sin();
        let v = (s * i16::MAX as f32 * 0.8) as i16;
        for _ in 0..channels {
            w.write_sample(v).unwrap();
        }
    }
    w.finalize().unwrap();
}

#[test]
fn decode_wav_estereo_44k_resamplea_a_16k_mono() {
    let p = tmp_path("44k_stereo.wav");
    write_sine_wav(&p, 44_100, 2, 1.0, 440.0);

    let pcm = load_audio_16k_mono(&p).expect("decodificación");
    assert_eq!(pcm.sample_rate, 16_000, "debe resamplear a 16 kHz");
    // ~1 segundo a 16 kHz (tolerancia por el resampler).
    let len = pcm.samples.len() as i64;
    assert!((len - 16_000).abs() < 2_000, "longitud inesperada: {len}");
    // Señal no trivial (no todo ceros).
    let energia: f32 = pcm.samples.iter().map(|x| x.abs()).sum();
    assert!(energia > 0.0, "señal vacía");

    let _ = std::fs::remove_file(&p);
}

#[test]
fn decode_wav_mono_16k_passthrough() {
    let p = tmp_path("16k_mono.wav");
    write_sine_wav(&p, 16_000, 1, 0.5, 440.0);

    let pcm = load_audio_16k_mono(&p).expect("decodificación");
    assert_eq!(pcm.sample_rate, 16_000);
    let len = pcm.samples.len() as i64;
    assert!((len - 8_000).abs() < 500, "longitud inesperada: {len}");

    let _ = std::fs::remove_file(&p);
}
