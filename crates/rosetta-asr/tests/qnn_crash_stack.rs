//! Diagnóstico del crash QNN: ¿qué módulo recursa al crear la sesión? Captura el
//! call stack del STATUS_STACK_OVERFLOW con un vectored exception handler (+
//! `SetThreadStackGuarantee` para que el handler tenga pila) y resuelve el módulo
//! de cada frame. Si los frames repetidos son de `QnnHtpPrepare.dll` (backend QAIRT
//! de preparación de grafo) → swapeable a QAIRT 2.45. Si son de `onnxruntime*` →
//! lado onnxruntime, no swapeable con ort rc.12 (api-24).
//!
//! Manual: `ORT_DYLIB_PATH` + PATH -> runtime/win-arm64-qnn, `--ignored --nocapture`.
#![cfg(windows)]

use std::ffi::c_void;

use ort::session::Session;
use ort::session::builder::AutoDevicePolicy;
use windows_sys::Win32::Foundation::HMODULE;
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_POINTERS, RtlCaptureStackBackTrace,
};
use windows_sys::Win32::System::LibraryLoader::{
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    GetModuleFileNameW, GetModuleHandleExW,
};
use windows_sys::Win32::System::Threading::{ExitProcess, SetThreadStackGuarantee};

const STATUS_STACK_OVERFLOW: u32 = 0xC000_00FD;

/// Nombre base del módulo que contiene `addr` (p. ej. "QnnHtpPrepare.dll").
unsafe fn module_for(addr: *mut c_void) -> String {
    let mut hmod: HMODULE = core::ptr::null_mut();
    let ok = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            addr as *const u16,
            &mut hmod,
        )
    };
    if ok == 0 {
        return "??".to_string();
    }
    let mut buf = [0u16; 260];
    let len = unsafe { GetModuleFileNameW(hmod, buf.as_mut_ptr(), buf.len() as u32) } as usize;
    let full = String::from_utf16_lossy(&buf[..len]);
    full.rsplit(['\\', '/']).next().unwrap_or(&full).to_string()
}

/// Handler de excepción: solo actúa en el stack-overflow; el resto las deja pasar.
unsafe extern "system" fn veh(info: *mut EXCEPTION_POINTERS) -> i32 {
    let code = unsafe { (*(*info).ExceptionRecord).ExceptionCode } as u32;
    if code != STATUS_STACK_OVERFLOW {
        return 0; // EXCEPTION_CONTINUE_SEARCH
    }
    let mut frames = [core::ptr::null_mut::<c_void>(); 256];
    let n = unsafe { RtlCaptureStackBackTrace(0, 256, frames.as_mut_ptr(), core::ptr::null_mut()) }
        as usize;

    eprintln!(
        "\n[crash-stack] STATUS_STACK_OVERFLOW — {n} frames. Módulos (cima -> base, agrupados):"
    );
    let mut last = String::new();
    let mut run = 0usize;
    let mut grupos = 0usize;
    for &fr in frames.iter().take(n) {
        let m = unsafe { module_for(fr) };
        if m == last {
            run += 1;
        } else {
            if !last.is_empty() {
                eprintln!("  {last}  x{run}");
                grupos += 1;
                if grupos > 25 {
                    break;
                }
            }
            last = m;
            run = 1;
        }
    }
    if !last.is_empty() {
        eprintln!("  {last}  x{run}");
    }
    use std::io::Write;
    let _ = std::io::stderr().flush();
    unsafe { ExitProcess(0) };
}

#[test]
#[ignore = "diagnóstico: módulo que recursa en el crash QNN. Manual con runtime QNN en PATH."]
fn qnn_crash_stack() {
    unsafe { AddVectoredExceptionHandler(1, Some(veh)) };

    let env = ort::environment::current().expect("entorno ort");
    let qnn_lib = std::env::var("ROSETTA_QNN_EP_LIB").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../runtime/win-arm64-qnn/onnxruntime_providers_qnn.dll"
        )
        .to_string()
    });
    let _lib = env
        .register_ep_library("QNN", &qnn_lib)
        .expect("register_ep_library");

    let model = std::env::var("ROSETTA_PROBE_MODEL").unwrap_or_else(|_| {
        let models = std::env::var("ROSETTA_MODELS_DIR")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../models").to_string());
        std::path::Path::new(&models)
            .join("parakeet-tdt-0.6b-v3")
            .join("nemo128.onnx")
            .to_string_lossy()
            .into_owned()
    });
    eprintln!("[crash-stack] creando sesión QNN (PreferNPU) sobre {model} ...");

    let h = std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(move || {
            // Reservar pila para que el handler pueda ejecutarse al desbordar.
            let mut guarantee: u32 = 512 * 1024;
            unsafe { SetThreadStackGuarantee(&mut guarantee) };
            let r = Session::builder()
                .expect("builder")
                .with_auto_device(AutoDevicePolicy::PreferNPU)
                .expect("auto_device")
                .commit_from_file(&model);
            // Si llega aquí, NO hubo overflow.
            eprintln!("[crash-stack] sin overflow: {:?}", r.map(|_| "sesión OK"));
        })
        .expect("spawn");
    let _ = h.join();
}

/// Variante: QNN EP heredado con `htp_arch`/`soc_model` EXPLÍCITOS. Hipótesis: la
/// recursión de `QnnHtp.dll` está en la auto-detección del SoC (las hojas del stack
/// son `QcSoCServiceUtils.dll`); darle la arch/SoC a mano podría cortocircuitarla.
/// Parametrizable por env para iterar sin recompilar:
///   ROSETTA_QNN_BACKEND (default QnnHtp.dll de win-arm64-qnn-247),
///   ROSETTA_QNN_HTP_ARCH (default 73), ROSETTA_QNN_SOC_MODEL (opcional).
#[test]
#[ignore = "diagnóstico: ¿htp_arch/soc_model explícitos evitan la recursión? Manual."]
fn qnn_with_arch_hint() {
    unsafe { AddVectoredExceptionHandler(1, Some(veh)) };

    let backend = std::env::var("ROSETTA_QNN_BACKEND").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../runtime/win-arm64-qnn-247/QnnHtp.dll"
        )
        .to_string()
    });
    let arch: u32 = std::env::var("ROSETTA_QNN_HTP_ARCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(73);
    let soc = std::env::var("ROSETTA_QNN_SOC_MODEL").ok();
    let model = std::env::var("ROSETTA_PROBE_MODEL").unwrap_or_else(|_| {
        let models = std::env::var("ROSETTA_MODELS_DIR")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../models").to_string());
        std::path::Path::new(&models)
            .join("parakeet-tdt-0.6b-v3")
            .join("nemo128.onnx")
            .to_string_lossy()
            .into_owned()
    });
    eprintln!("[qnn-opts] backend={backend} htp_arch={arch} soc_model={soc:?} model={model}");

    let h = std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(move || {
            let mut guarantee: u32 = 512 * 1024;
            unsafe { SetThreadStackGuarantee(&mut guarantee) };
            let mut qnn = ort::ep::QNN::default()
                .with_backend_path(backend)
                .with_htp_arch(arch)
                .with_htp_graph_finalization_optimization_mode(3)
                .with_performance_mode(ort::ep::qnn::PerformanceMode::Burst);
            if let Some(s) = soc {
                qnn = qnn.with_soc_model(s);
            }
            let r = Session::builder()
                .expect("builder")
                .with_execution_providers([qnn.build()])
                .expect("registrar QNN")
                .commit_from_file(&model);
            eprintln!(
                "[qnn-opts] sin overflow -> {:?}",
                r.map(|_| "sesión OK".to_string())
                    .map_err(|e| e.to_string())
            );
        })
        .expect("spawn");
    let _ = h.join();
}
