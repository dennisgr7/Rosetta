//! Smoke F4: ¿cargan los modelos Whisper-QNN (EPContext, `.bin` compilado para
//! Snapdragon X-Elite) en la NPU Hexagon de este equipo (X-Plus, X1-26-100)?
//! Diagnóstico del riesgo central de F4. Se omiten si el modelo no está.
//! Requiere `ORT_DYLIB_PATH` -> runtime/win-arm64-qnn/onnxruntime.dll y la carpeta
//! win-arm64-qnn en el PATH. Ejecutar a mano: `--ignored`.

use std::path::Path;

use ort::ep::{self, ExecutionProvider};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

fn qnn_backend() -> String {
    std::env::var("ROSETTA_QNN_BACKEND").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../runtime/win-arm64-qnn/QnnHtp.dll"
        )
        .to_string()
    })
}

/// Raíz del catálogo de modelos: `ROSETTA_MODELS_DIR` o `models/` en la raíz del
/// workspace (relativa a este crate). Se usa para componer rutas de modelos sin
/// codificar ninguna ruta absoluta del usuario.
fn modelo(rel: &str) -> String {
    let models = std::env::var("ROSETTA_MODELS_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../models").to_string());
    Path::new(&models).join(rel).to_string_lossy().into_owned()
}

fn try_load(onnx: &str) {
    if !Path::new(onnx).exists() {
        eprintln!("[smoke] modelo ausente: {onnx}");
        return;
    }
    let backend = qnn_backend();

    let qnn = ep::QNN::default()
        .with_backend_path(backend)
        .with_performance_mode(ep::qnn::PerformanceMode::Burst)
        .with_htp_fp16_precision(true);
    eprintln!("[smoke] QNN is_available = {:?}", qnn.is_available());

    let result = Session::builder()
        .expect("builder")
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .expect("opt level")
        .with_execution_providers([qnn.build()])
        .expect("registrar QNN")
        .commit_from_file(onnx);

    match result {
        Ok(_session) => eprintln!("[smoke] RESULTADO: OK — {onnx} cargó en QNN"),
        Err(e) => eprintln!("[smoke] RESULTADO: FALLO QNN -> {onnx} -> {e}"),
    }
}

fn try_load_plain(onnx: &str) {
    if !Path::new(onnx).exists() {
        eprintln!("[smoke] modelo ausente: {onnx}");
        return;
    }
    let backend = qnn_backend();
    let qnn = ep::QNN::default().with_backend_path(backend);
    eprintln!("[smoke-plain] QNN is_available = {:?}", qnn.is_available());
    let result = Session::builder()
        .expect("builder")
        .with_execution_providers([qnn.build()])
        .expect("registrar QNN")
        .commit_from_file(onnx);
    match result {
        Ok(_session) => eprintln!("[smoke-plain] RESULTADO: OK — {onnx} cargó en QNN"),
        Err(e) => eprintln!("[smoke-plain] RESULTADO: FALLO QNN -> {onnx} -> {e}"),
    }
}

fn spawn_big_stack(onnx: String) {
    // La carga del modelo QNN necesita mucho stack; thread dedicado de 512 MB.
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(move || try_load(&onnx))
        .expect("spawn")
        .join()
        .expect("join");
}

#[test]
#[ignore = "F4: el encoder QNN (X-Elite) desborda el stack al cargar en el X-Plus. Ejecutar a mano con ORT_DYLIB_PATH/PATH al runtime QNN."]
fn whisper_qnn_encoder_loads() {
    spawn_big_stack(modelo(
        "whisper-large-v3-turbo-qnn/snapdragon-x-elite/encoder/model.onnx",
    ));
}

#[test]
#[ignore = "F4: diagnóstico — ¿el decoder QNN (X-Elite, 452 MB) carga en el X-Plus? Ejecutar a mano."]
fn whisper_qnn_decoder_loads() {
    spawn_big_stack(modelo(
        "whisper-large-v3-turbo-qnn/snapdragon-x-elite/decoder/model.onnx",
    ));
}

#[test]
#[ignore = "F4: diagnóstico decisivo — ¿carga/compila JIT un ONNX int8 PROPIO (Parakeet decoder_joint, no ligado a ningún SoC) en el QNN del X-Plus? Si OK, el runtime QNN + el chip funcionan y el fallo de Whisper es solo el .bin de X-Elite."]
fn parakeet_qnn_decoder_joint_loads() {
    spawn_big_stack(modelo("parakeet-tdt-0.6b-v3/decoder_joint-model.int8.onnx"));
}

#[test]
#[ignore = "F4: ¿QNN con opciones por defecto (sin Burst/fp16) sobre el modelo más pequeño (nemo128) tampoco carga? Aísla si la recursión la dispara una opción concreta."]
fn nemo128_qnn_plain_loads() {
    // try_load_plain en thread de 512 MB.
    let onnx = modelo("parakeet-tdt-0.6b-v3/nemo128.onnx");
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(move || try_load_plain(&onnx))
        .expect("spawn")
        .join()
        .expect("join");
}

// --- DirectML (vía elegida tras el bloqueo de QNN) ---

fn try_load_dml(onnx: &str) {
    if !Path::new(onnx).exists() {
        eprintln!("[dml] modelo ausente: {onnx}");
        return;
    }
    let dml = ep::DirectML::default();
    eprintln!("[dml] DirectML is_available = {:?}", dml.is_available());
    let result = Session::builder()
        .expect("builder")
        .with_execution_providers([dml.build()])
        .expect("registrar DML")
        .commit_from_file(onnx);
    match result {
        Ok(_s) => eprintln!("[dml] RESULTADO: OK — {onnx} cargó en DirectML"),
        Err(e) => eprintln!("[dml] RESULTADO: FALLO DML -> {onnx} -> {e}"),
    }
}

#[test]
#[ignore = "F4 DirectML: ¿carga nemo128 (0.13 MB) en DirectML (GPU Adreno del Snapdragon)? Requiere ORT_DYLIB_PATH/PATH -> runtime/win-arm64-dml."]
fn nemo128_dml_loads() {
    let onnx = modelo("parakeet-tdt-0.6b-v3/nemo128.onnx");
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || try_load_dml(&onnx))
        .expect("spawn")
        .join()
        .expect("join");
}

#[test]
#[ignore = "F4 DirectML: ¿carga el Parakeet decoder_joint int8 (17 MB) en DirectML?"]
fn parakeet_decoder_joint_dml_loads() {
    let onnx = modelo("parakeet-tdt-0.6b-v3/decoder_joint-model.int8.onnx");
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || try_load_dml(&onnx))
        .expect("spawn")
        .join()
        .expect("join");
}
