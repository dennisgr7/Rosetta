//! Smoke F6: la diarización separa dos hablantes en un audio sintético con dos
//! voces alternadas (A,B,A,B). Requiere los modelos silero + campplus y
//! `ORT_DYLIB_PATH`/PATH a un runtime onnxruntime.

use std::collections::HashSet;
use std::path::Path;

use rosetta_audio::load_audio_16k_mono;
use rosetta_diarize::{DiarizeConfig, Diarizer};

#[test]
#[ignore = "requiere models/{silero-vad,campplus-sv-zh-en} y ORT_DYLIB_PATH"]
fn diariza_dos_hablantes() {
    let models = std::env::var("ROSETTA_MODELS_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../models").to_string());
    let models = Path::new(&models);
    let vad = models.join("silero-vad").join("silero_vad.onnx");
    let emb = models.join("campplus-sv-zh-en").join("campplus.onnx");
    let seg = models.join("pyannote-segmentation-3.0").join("model.onnx");
    let wav = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/es_2voces.wav"
    ));
    if !emb.exists() || !seg.exists() || !wav.exists() {
        eprintln!("modelo/fixture ausente; test omitido");
        return;
    }

    let hw = rosetta_accel::detect_hw();
    let audio = load_audio_16k_mono(wav).expect("decode");

    let mut d = Diarizer::new(&vad, &emb, &seg, &hw, 4, DiarizeConfig::default())
        .expect("cargar diarizador");
    let (turns, overlaps) = d.diarize(&audio).expect("diarize");
    eprintln!("solapes detectados = {}", overlaps.len());

    eprintln!("turns ({}): {turns:?}", turns.len());
    let speakers: HashSet<usize> = turns.iter().map(|t| t.speaker).collect();
    eprintln!("hablantes detectados = {}", speakers.len());

    assert!(
        turns.len() >= 4,
        "esperaba >=4 tramos de voz, hubo {}",
        turns.len()
    );
    assert_eq!(
        speakers.len(),
        2,
        "esperaba 2 hablantes distintos, detectó {}",
        speakers.len()
    );
}
