//! Smoke F5: Silero VAD detecta voz en el clip de prueba. Requiere el modelo en
//! `models/silero-vad/` y `ORT_DYLIB_PATH`/PATH a un runtime onnxruntime.

use std::path::Path;

use rosetta_audio::{SileroVad, load_audio_16k_mono};

#[test]
#[ignore = "requiere models/silero-vad y ORT_DYLIB_PATH"]
fn silero_detects_speech() {
    let models = std::env::var("ROSETTA_MODELS_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../models").to_string());
    let model = Path::new(&models)
        .join("silero-vad")
        .join("silero_vad.onnx");
    let wav = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/es_prueba.wav"
    ));
    if !model.exists() {
        eprintln!("modelo silero ausente; test omitido");
        return;
    }

    let hw = rosetta_accel::detect_hw();
    let audio = load_audio_16k_mono(wav).expect("decode wav");
    let mut vad = SileroVad::from_file(&model, &hw, 4).expect("cargar Silero");
    let segs = vad.detect(&audio).expect("detect");

    eprintln!("duración={:.2}s segmentos={segs:?}", audio.duration_s());
    assert!(!segs.is_empty(), "Silero no detectó voz en un clip hablado");
    let total: f32 = segs.iter().map(|s| s.end_s - s.start_s).sum();
    eprintln!("voz total detectada = {total:.2}s");
    assert!(total > 1.0, "voz detectada demasiado corta: {total:.2}s");
    // Monotonía y no solape.
    for w in segs.windows(2) {
        assert!(w[0].end_s <= w[1].start_s + 0.001, "segmentos solapados");
    }
}
