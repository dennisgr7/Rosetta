//! Spike NPU (de-risk Windows ML / EpDevice V2). ¿El runtime cargado expone QNN/NPU
//! como `EpDevice` auto-descubrible, y `with_auto_device` (ruta V2, distinta del
//! `AppendExecutionProvider_QNN` heredado que desborda el stack) puede crear sesión?
//! Requiere `ORT_DYLIB_PATH` -> onnxruntime.dll y su carpeta en el PATH.
//! Ejecutar a mano: `--ignored --nocapture`.

use ort::session::Session;
use ort::session::builder::AutoDevicePolicy;

#[test]
#[ignore = "spike NPU: enumera EpDevices y prueba with_auto_device(PreferNPU). Manual."]
fn autoep_enumera_y_prueba() {
    let env = ort::environment::current().expect("entorno ort");

    // Registrar el plugin-EP de QNN (API moderna RegisterExecutionProviderLibrary,
    // que ort rc.12 envuelve en `register_ep_library`). Si el DLL es un plugin-EP
    // válido, QNN debería aparecer como EpDevice de tipo NPU.
    let qnn_lib = std::env::var("ROSETTA_QNN_EP_LIB").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../runtime/win-arm64-qnn/onnxruntime_providers_qnn.dll"
        )
        .to_string()
    });
    let _lib = match env.register_ep_library("QNN", &qnn_lib) {
        Ok(l) => {
            eprintln!("register_ep_library OK: {qnn_lib}");
            Some(l)
        }
        Err(e) => {
            eprintln!("register_ep_library FALLO: {e}");
            None
        }
    };

    eprintln!("== EpDevices que expone el runtime cargado ==");
    let mut hay_npu = false;
    for d in env.devices() {
        let ep = d.ep().unwrap_or("?");
        let ty = d.ty();
        if matches!(ty, ort::memory::DeviceType::NPU) {
            hay_npu = true;
        }
        eprintln!(
            "  ep={ep:24} vendor_ep={:?} type={ty:?} hw_vendor={:?}",
            d.ep_vendor().unwrap_or("?"),
            d.vendor().unwrap_or("?"),
        );
    }
    eprintln!("== ¿hay algún EpDevice de tipo NPU? {hay_npu} ==");

    let model = std::env::var("ROSETTA_PROBE_MODEL").unwrap_or_else(|_| {
        let models = std::env::var("ROSETTA_MODELS_DIR")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../models").to_string());
        std::path::Path::new(&models)
            .join("parakeet-tdt-0.6b-v3")
            .join("nemo128.onnx")
            .to_string_lossy()
            .into_owned()
    });
    eprintln!("== with_auto_device(PreferNPU) sobre {model} ==");
    // Thread de 512 MB por si la ruta V2 tropieza con la misma recursión.
    let r = std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(move || {
            Session::builder()
                .expect("builder")
                .with_auto_device(AutoDevicePolicy::PreferNPU)
                .expect("with_auto_device")
                .commit_from_file(&model)
        })
        .expect("spawn")
        .join();
    match r {
        Ok(Ok(_)) => eprintln!("RESULTADO: OK — sesión creada por auto-device(PreferNPU)"),
        Ok(Err(e)) => eprintln!("RESULTADO: Err -> {e}"),
        Err(_) => eprintln!("RESULTADO: el thread abortó (stack overflow / panic)"),
    }
}
