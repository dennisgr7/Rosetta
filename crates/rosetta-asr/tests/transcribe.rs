//! Test golden de F2: decodifica un audio en español y lo transcribe con el
//! motor Parakeet. Se omite si el modelo o el fixture no están presentes.
//! Requiere `ORT_DYLIB_PATH` apuntando a onnxruntime.

use std::path::Path;

use rosetta_asr::{Engine, ParakeetEngine};

#[test]
fn transcribe_es_prueba() {
    let models = std::env::var("ROSETTA_MODELS_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../models").to_string());
    let model_dir = Path::new(&models).join("parakeet-tdt-0.6b-v3");
    let wav = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/es_prueba.wav"
    ));
    if !model_dir.exists() || !wav.exists() {
        eprintln!("modelo o fixture ausente; test omitido");
        return;
    }

    let pcm = rosetta_audio::load_audio_16k_mono(wav).expect("decodificar wav");
    let hw = rosetta_accel::detect_hw();
    let mut engine = ParakeetEngine::from_dir(&model_dir, &hw, rosetta_accel::Device::Cpu, 4)
        .expect("cargar modelo Parakeet");
    let transcript = engine
        .transcribe(&pcm.samples, pcm.sample_rate)
        .expect("transcribir");

    eprintln!(
        "=== TEXTO TRANSCRITO ===\n{}\n========================",
        transcript.text
    );

    let low = transcript.text.to_lowercase();
    assert!(!low.trim().is_empty(), "transcripción vacía");
    // Ground truth: "Hola, esto es una prueba de transcripción con Rosetta."
    assert!(
        low.contains("prueba") || low.contains("transcrip") || low.contains("hola"),
        "texto inesperado: {}",
        transcript.text
    );
}
