//! Volcado del contrato I/O del modelo pyannote-segmentation-3.0 (Bloque E2).
//! Ignorado: requiere el modelo descargado y `ORT_DYLIB_PATH`.
//!   cargo test -p rosetta-diarize --test pyannote_io_dump -- --ignored --nocapture

use std::path::Path;

#[test]
#[ignore = "requiere pyannote-segmentation-3.0 descargado"]
fn dump_pyannote_io() {
    let models = std::env::var("ROSETTA_MODELS_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../models").to_string());
    let path = Path::new(&models)
        .join("pyannote-segmentation-3.0")
        .join("model.onnx");
    if !path.exists() {
        eprintln!("ausente: {}", path.display());
        return;
    }
    let session = ort::session::Session::builder()
        .expect("SessionBuilder")
        .commit_from_file(&path)
        .expect("commit_from_file");
    println!("-- inputs --");
    for i in session.inputs() {
        println!("  {i:?}");
    }
    println!("-- outputs --");
    for o in session.outputs() {
        println!("  {o:?}");
    }
}
