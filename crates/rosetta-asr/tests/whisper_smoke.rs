//! Smoke end-to-end del motor Whisper (Bloque D): mel + encoder + decode loop con
//! KV-cache sobre un audio real. Ignorado por defecto (requiere los modelos
//! Whisper descargados y `ORT_DYLIB_PATH`).
//!
//!   cargo test -p rosetta-asr --test whisper_smoke -- --ignored --nocapture

use std::path::Path;

use rosetta_accel::Device;
use rosetta_asr::{Engine, WhisperEngine};

#[test]
#[ignore = "requiere modelos Whisper + ORT_DYLIB_PATH"]
fn whisper_transcribe_es() {
    let models = std::env::var("ROSETTA_MODELS_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../models").to_string());
    let model_dir = Path::new(&models).join("whisper-large-v3-turbo");
    let wav = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/es_prueba.wav"
    ));
    if !model_dir.exists() || !wav.exists() {
        eprintln!("modelos o fixture ausentes; test omitido");
        return;
    }
    let audio = rosetta_audio::load_audio_16k_mono(wav).expect("cargar wav");
    let hw = rosetta_accel::detect_hw();
    // CPU como referencia (evita posibles caídas silenciosas de int8 en DirectML).
    let mut eng = WhisperEngine::from_dir(&model_dir, &hw, Device::Cpu, 4).expect("cargar whisper");
    let t = eng
        .transcribe(&audio.samples, audio.sample_rate)
        .expect("transcribir");
    println!("IDIOMA detectado: {}", t.source.language);
    println!("TEXTO: {}", t.text);
    assert!(!t.text.trim().is_empty(), "texto vacío");
    assert_eq!(t.source.language, "es", "idioma esperado es");
    assert!(
        t.text.to_lowercase().contains("prueba"),
        "esperaba 'prueba' en: {}",
        t.text
    );
}
