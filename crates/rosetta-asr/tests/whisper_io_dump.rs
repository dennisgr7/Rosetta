//! Volcado del contrato I/O real de los ONNX de Whisper (Bloque D, paso D0).
//!
//! Ignorado por defecto: requiere los modelos descargados y `ORT_DYLIB_PATH`.
//! Ejecutar con:
//!   cargo test -p rosetta-asr --test whisper_io_dump -- --ignored --nocapture

use std::path::Path;

fn dump(path: &str) {
    if !Path::new(path).exists() {
        eprintln!("ausente: {path}");
        return;
    }
    let session = ort::session::Session::builder()
        .expect("SessionBuilder")
        .commit_from_file(path)
        .expect("commit_from_file");
    println!("===== {path} =====");
    println!("-- inputs --");
    for i in session.inputs() {
        println!("  {i:?}");
    }
    println!("-- outputs --");
    for o in session.outputs() {
        println!("  {o:?}");
    }
}

#[test]
#[ignore = "requiere los modelos Whisper descargados"]
fn dump_whisper_io() {
    let models = std::env::var("ROSETTA_MODELS_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../models").to_string());
    let base = Path::new(&models).join("whisper-large-v3-turbo");
    dump(&base.join("encoder_model_int8.onnx").to_string_lossy());
    dump(
        &base
            .join("decoder_model_merged_int8.onnx")
            .to_string_lossy(),
    );
}
