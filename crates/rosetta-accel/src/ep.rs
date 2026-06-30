//! Cascada real de Execution Providers sobre `ort`: planificación por
//! `(SO, arch, vendor)`, construcción de sesiones con fallback NPU→GPU→CPU, y
//! comprobación de disponibilidad para el flag `--device` con error duro.

use std::fmt;
use std::path::{Path, PathBuf};

use ort::ep::{self, ExecutionProviderDispatch};
// `ExecutionProvider` aporta `is_available()`, invocado en los EPs de Windows
// (QNN/DirectML), Linux/Windows (OpenVINO) y macOS (CoreML).
#[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
use ort::ep::ExecutionProvider;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

use rosetta_core::{Result, RosettaError};

use crate::{Arch, Device, EpKind, HwProfile, Os, OvDevice, Vendor};

/// Cómo materializar/consultar un EP concreto. Una sola entrada por `EpKind` en
/// [`ep_spec`] es la **fuente única** de la que salen etiqueta, dispatch y
/// disponibilidad (antes había que sincronizar 4 sitios a mano).
///
/// `dispatch`/`available` son closures (no `const`): un `ExecutionProviderDispatch`
/// se crea con `.build()`, que NO es `const fn`, así que el dispatch se materializa
/// bajo demanda. Devuelven `None`/`false` cuando el EP no está compilado en este
/// target (gateado por `#[cfg]` dentro del closure).
pub(crate) struct EpSpec {
    /// Etiqueta legible (logs, `info`, `doctor`).
    pub label: &'static str,
    /// Construye el `ExecutionProviderDispatch` real, o `None` si no está compilado.
    dispatch: fn() -> Option<ExecutionProviderDispatch>,
    /// ¿Disponible en el runtime cargado? (`false` si no está compilado.)
    available: fn() -> bool,
}

/// Tabla única de EPs: label + dispatch + disponibilidad por `EpKind`. Añadir un EP
/// nuevo = una entrada aquí (más, si toca, su feature de `ort` por target en
/// `Cargo.toml`). Los `#[cfg]` viven dentro de los closures para que en plataformas
/// donde el EP no existe `dispatch`→`None` y `available`→`false` sin tocar la tabla.
pub(crate) fn ep_spec(kind: EpKind) -> EpSpec {
    match kind {
        EpKind::Qnn => EpSpec {
            label: "QNN (Qualcomm Hexagon NPU)",
            dispatch: || {
                #[cfg(windows)]
                {
                    Some(
                        ep::QNN::default()
                            .with_performance_mode(ep::qnn::PerformanceMode::Burst)
                            .with_htp_fp16_precision(true)
                            .build(),
                    )
                }
                #[cfg(not(windows))]
                {
                    None
                }
            },
            available: || {
                #[cfg(windows)]
                {
                    ep::QNN::default().is_available().unwrap_or(false)
                }
                #[cfg(not(windows))]
                {
                    false
                }
            },
        },
        EpKind::OpenVino => EpSpec {
            label: "OpenVINO (Intel NPU)",
            dispatch: || openvino_dispatch(OvDevice::Npu),
            available: openvino_available,
        },
        EpKind::OpenVinoGpu => EpSpec {
            label: "OpenVINO (Intel GPU)",
            dispatch: || openvino_dispatch(OvDevice::Gpu),
            available: openvino_available,
        },
        EpKind::DirectMlNpu => EpSpec {
            label: "DirectML (NPU)",
            dispatch: || directml_dispatch(DmlKind::Npu),
            available: directml_available,
        },
        EpKind::DirectMlGpu => EpSpec {
            label: "DirectML (GPU)",
            dispatch: || directml_dispatch(DmlKind::Gpu),
            available: directml_available,
        },
        EpKind::CoreMl => EpSpec {
            label: "CoreML (Apple ANE/GPU)",
            dispatch: || {
                #[cfg(target_os = "macos")]
                {
                    Some(ep::CoreML::default().build())
                }
                #[cfg(not(target_os = "macos"))]
                {
                    None
                }
            },
            available: || {
                #[cfg(target_os = "macos")]
                {
                    ep::CoreML::default().is_available().unwrap_or(false)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    false
                }
            },
        },
        // CUDA / TensorRT (NVIDIA, Windows+Linux x86_64). SCAFFOLD OPT-IN (arq-1): el
        // EP se registra si la feature de `ort` está compilada y el runtime CUDA está
        // presente, PERO el % de nodos colocados en GPU discreta NO está validado en
        // HW. NO afirmar "corre en GPU" sin medir con ROSETTA_ORT_PROFILE.
        EpKind::Cuda => EpSpec {
            label: "CUDA (NVIDIA GPU)",
            dispatch: cuda_dispatch,
            available: cuda_available,
        },
        EpKind::TensorRt => EpSpec {
            label: "TensorRT (NVIDIA GPU)",
            dispatch: tensorrt_dispatch,
            available: tensorrt_available,
        },
        // Aún no cableados (sin feature/EP): solo etiqueta. dispatch→None ⇒ la cascada
        // los omite; available→false ⇒ nunca cuentan para el error duro forzado.
        EpKind::Vitis => EpSpec {
            label: "Vitis AI (AMD XDNA NPU)",
            dispatch: || None,
            available: || false,
        },
        EpKind::Xnnpack => EpSpec {
            label: "XNNPACK (CPU optimizado)",
            dispatch: || None,
            available: || false,
        },
        EpKind::Cpu => EpSpec {
            label: "CPU",
            dispatch: || Some(ep::CPU::default().build()),
            available: || true,
        },
    }
}

// --- Materializadores por familia (mantienen los `#[cfg]` en un solo sitio) ---

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn openvino_dispatch(dev: OvDevice) -> Option<ExecutionProviderDispatch> {
    // `with_device_type` espera un string; `OvDevice: Display` da el literal exacto
    // ("NPU"/"GPU") sin que un typo compile.
    Some(
        ep::OpenVINO::default()
            .with_device_type(dev.to_string())
            .build(),
    )
}
#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn openvino_dispatch(_dev: OvDevice) -> Option<ExecutionProviderDispatch> {
    None
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn openvino_available() -> bool {
    ep::OpenVINO::default().is_available().unwrap_or(false)
}
#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn openvino_available() -> bool {
    false
}

/// Selector interno de filtro de dispositivo de DirectML (evita acoplar la tabla al
/// tipo `ep::directml::DeviceFilter`, que solo existe en Windows).
#[derive(Clone, Copy)]
enum DmlKind {
    Npu,
    Gpu,
}

#[cfg(windows)]
fn directml_dispatch(kind: DmlKind) -> Option<ExecutionProviderDispatch> {
    let filter = match kind {
        DmlKind::Npu => ep::directml::DeviceFilter::Npu,
        DmlKind::Gpu => ep::directml::DeviceFilter::Gpu,
    };
    Some(ep::DirectML::default().with_device_filter(filter).build())
}
#[cfg(not(windows))]
fn directml_dispatch(_kind: DmlKind) -> Option<ExecutionProviderDispatch> {
    None
}

#[cfg(windows)]
fn directml_available() -> bool {
    ep::DirectML::default().is_available().unwrap_or(false)
}
#[cfg(not(windows))]
fn directml_available() -> bool {
    false
}

// CUDA / TensorRT: gateados a x86_64 Windows/Linux (sin peso muerto en aarch64). Con
// `load-dynamic` la feature NO compila C; la disponibilidad real depende del runtime
// CUDA cargado. arq-1 SCAFFOLD: no validado en HW; ver nota en `ep_spec`.
#[cfg(all(
    any(target_os = "windows", target_os = "linux"),
    target_arch = "x86_64"
))]
fn cuda_dispatch() -> Option<ExecutionProviderDispatch> {
    Some(ep::CUDA::default().build())
}
#[cfg(not(all(
    any(target_os = "windows", target_os = "linux"),
    target_arch = "x86_64"
)))]
fn cuda_dispatch() -> Option<ExecutionProviderDispatch> {
    None
}

#[cfg(all(
    any(target_os = "windows", target_os = "linux"),
    target_arch = "x86_64"
))]
fn cuda_available() -> bool {
    ep::CUDA::default().is_available().unwrap_or(false)
}
#[cfg(not(all(
    any(target_os = "windows", target_os = "linux"),
    target_arch = "x86_64"
)))]
fn cuda_available() -> bool {
    false
}

#[cfg(all(
    any(target_os = "windows", target_os = "linux"),
    target_arch = "x86_64"
))]
fn tensorrt_dispatch() -> Option<ExecutionProviderDispatch> {
    Some(ep::TensorRT::default().build())
}
#[cfg(not(all(
    any(target_os = "windows", target_os = "linux"),
    target_arch = "x86_64"
)))]
fn tensorrt_dispatch() -> Option<ExecutionProviderDispatch> {
    None
}

#[cfg(all(
    any(target_os = "windows", target_os = "linux"),
    target_arch = "x86_64"
))]
fn tensorrt_available() -> bool {
    ep::TensorRT::default().is_available().unwrap_or(false)
}
#[cfg(not(all(
    any(target_os = "windows", target_os = "linux"),
    target_arch = "x86_64"
)))]
fn tensorrt_available() -> bool {
    false
}

/// Planificación (pura) de la cascada de EPs para `(SO, arch, vendor)` y la
/// preferencia `--device`. Devuelve los tipos en orden de prioridad. CPU solo se
/// incluye en `Auto`/`Cpu` (para permitir error duro en `Npu`/`Gpu` forzados).
///
/// Cableado por plataforma/vendor: Windows QNN(bloqueado)/DirectML, Intel OpenVINO
/// (Win/Linux), Apple CoreML (macOS). AMD-NPU (Vitis) queda como scaffold. Brazos de
/// GPU discreta (NVIDIA CUDA/TensorRT, AMD Radeon) se derivan de `hw.gpu_vendor`
/// (arq-1, opt-in: filtrados por `ep_available` en runtime). El EP solo se usa si
/// además está `disponible` en runtime (ver `build_session`).
pub fn cascade_kinds(hw: &HwProfile, want: Device) -> Vec<EpKind> {
    use EpKind as K;
    let (npus, mut gpus): (Vec<K>, Vec<K>) = match (hw.os, hw.arch, hw.cpu_vendor) {
        // Snapdragon: QNN bloqueado (stack overflow al crear sesión; ver
        // rosetta-asr/tests/whisper_qnn_smoke.rs). DirectML-NPU PROBADO y DESCARTADO
        // (2026-06-30): registra "DirectML (NPU)" pero coloca el **0% de los nodos**
        // en el acelerador (100% CPUExecutionProvider, verificado con
        // ROSETTA_ORT_PROFILE) → fallback silencioso a CPU, no toca la Hexagon
        // (cf. DirectML #640/#659). La NPU real solo por la vía Windows ML/QNN.
        // → NPU fuera; GPU Adreno por DirectML.
        (Os::Windows, Arch::Arm64, Vendor::Qualcomm) => (vec![], vec![K::DirectMlGpu]),
        // Intel: NPU + GPU por OpenVINO; en Windows, DirectML como respaldo de GPU.
        (Os::Windows, Arch::X64, Vendor::Intel) => {
            (vec![K::OpenVino], vec![K::OpenVinoGpu, K::DirectMlGpu])
        }
        (Os::Linux, Arch::X64, Vendor::Intel) => (vec![K::OpenVino], vec![K::OpenVinoGpu]),
        // AMD: GPU Radeon por DirectML (Windows). NPU XDNA (Vitis) = scaffold, fuera.
        (Os::Windows, Arch::X64, Vendor::Amd) => (vec![], vec![K::DirectMlGpu]),
        // Otros x64 Windows (NPU genérica) por DirectML.
        (Os::Windows, Arch::X64, _) => (vec![K::DirectMlNpu], vec![K::DirectMlGpu]),
        // Apple Silicon: CoreML (ANE + GPU integrados).
        (Os::Mac, Arch::Arm64, _) => (vec![K::CoreMl], vec![]),
        // Linux/macOS x64 y demás: CPU (onnxruntime usa XNNPACK/MLAS por dentro).
        _ => (vec![], vec![]),
    };

    // arq-1 (SCAFFOLD OPT-IN, no validado en HW): brazos de GPU discreta según
    // `hw.gpu_vendor`. Se ANTEPONEN a las GPU integradas ya cableadas (DirectML), de
    // modo que una NVIDIA/AMD discreta tenga prioridad si su EP está disponible. En
    // máquinas sin esa GPU (p.ej. Snapdragon), `ep_available()` los filtra en runtime
    // ⇒ la cascada efectiva NO cambia y los tests de plataforma siguen byte-equivalentes.
    // HONESTIDAD: el % de nodos colocados en la GPU discreta NO está medido; no afirmar
    // "corre en GPU" sin ROSETTA_ORT_PROFILE.
    match hw.gpu_vendor {
        // NVIDIA: TensorRT (mayor rendimiento) y CUDA como respaldo. TensorRT built-in
        // está deprecado en onnxruntime reciente; se mantiene primero pero si su EP no
        // está disponible, CUDA cubre. Solo aplica en x86_64 Win/Linux (gate de feature).
        Some(Vendor::Nvidia) => {
            let mut discrete = vec![K::TensorRt, K::Cuda];
            discrete.append(&mut gpus);
            gpus = discrete;
        }
        // AMD discreta:
        //  - Windows → DirectML (Radeon), ya cubierto por el brazo de plataforma; si el
        //    cpu_vendor no era AMD, lo aseguramos aquí anteponiéndolo.
        //  - Linux → ROCm sería el EP nativo, pero NO es un EP portable maduro en
        //    onnxruntime/ort (no hay variante `K::Rocm` ni feature estable). TODO(arq-1):
        //    añadir `K::Rocm` + feature `rocm` por target Linux cuando madure; de momento
        //    Linux+AMD cae a CPU (sin brazo de GPU) — comportamiento conservador, no se
        //    inventa un EP no validado.
        Some(Vendor::Amd) if hw.os == Os::Windows && !gpus.contains(&K::DirectMlGpu) => {
            gpus.insert(0, K::DirectMlGpu);
        }
        _ => {}
    }

    let mut eps = match want {
        Device::Auto => {
            let mut v = npus;
            v.extend(gpus);
            v
        }
        Device::Npu => npus,
        Device::Gpu => gpus,
        Device::Cpu => vec![],
    };
    if matches!(want, Device::Auto | Device::Cpu) {
        eps.push(K::Cpu);
    }
    eps
}

/// Construye el `ExecutionProviderDispatch` real para un `EpKind` (None si su EP
/// no está compilado en esta build). Sale de la tabla única [`ep_spec`].
fn dispatch_for(kind: EpKind) -> Option<ExecutionProviderDispatch> {
    (ep_spec(kind).dispatch)()
}

/// ¿Está disponible el EP en el runtime cargado? Requiere `ORT_DYLIB_PATH`. Sale de
/// la tabla única [`ep_spec`].
fn ep_available(kind: EpKind) -> bool {
    (ep_spec(kind).available)()
}

/// Error de selección de dispositivo forzado (`--device npu|gpu`).
#[derive(Debug, Clone)]
pub enum AccelError {
    ForcedNpuUnavailable { tried: Vec<EpKind> },
    ForcedGpuUnavailable { tried: Vec<EpKind> },
}

impl fmt::Display for AccelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccelError::ForcedNpuUnavailable { tried } => write!(
                f,
                "se forzó --device npu pero no hay ninguna NPU utilizable (probados: {}). \
                 Instala el runtime del acelerador o usa --device auto|cpu.",
                labels(tried)
            ),
            AccelError::ForcedGpuUnavailable { tried } => write!(
                f,
                "se forzó --device gpu pero no hay ninguna GPU utilizable (probados: {}). \
                 Usa --device auto|cpu.",
                labels(tried)
            ),
        }
    }
}

impl std::error::Error for AccelError {}

fn labels(kinds: &[EpKind]) -> String {
    if kinds.is_empty() {
        return "ninguno".to_string();
    }
    kinds
        .iter()
        .map(|k| k.label())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Para `--device npu|gpu`, comprueba que al menos un EP del tipo está disponible
/// en runtime. Error duro si no. (`Auto`/`Cpu` nunca fallan.)
pub fn check_forced_device(hw: &HwProfile, want: Device) -> std::result::Result<(), AccelError> {
    let kinds = match want {
        Device::Npu | Device::Gpu => cascade_kinds(hw, want),
        _ => return Ok(()),
    };
    if kinds.iter().copied().any(ep_available) {
        return Ok(());
    }
    Err(match want {
        Device::Gpu => AccelError::ForcedGpuUnavailable { tried: kinds },
        _ => AccelError::ForcedNpuUnavailable { tried: kinds },
    })
}

/// Para `doctor`: la cascada planificada con su disponibilidad real en runtime.
pub fn cascade_availability(hw: &HwProfile, want: Device) -> Vec<(EpKind, bool)> {
    cascade_kinds(hw, want)
        .into_iter()
        .map(|k| (k, ep_available(k)))
        .collect()
}

/// Construye una sesión registrando la cascada de EPs en orden, con fallback.
/// Devuelve la sesión y el EP de mayor prioridad de la cascada (estimación; la
/// verificación de placement real se hace aparte con profiling).
pub fn build_session(
    model: &Path,
    hw: &HwProfile,
    want: Device,
    threads: usize,
) -> Result<(Session, EpKind)> {
    // Vía NPU experimental (opt-in, Carril C / Windows ML): registra el QNN como
    // plugin-EP (`register_ep_library`, API moderna que ort rc.12 envuelve) y crea la
    // sesión por la ruta EpDevice V2 (`with_auto_device(PreferNPU)`), que ve la Hexagon.
    // Gated por `ROSETTA_ENABLE_QNN_EP` porque con QAIRT 2.42 la compilación de grafo en
    // `QnnHtp.dll` desborda el stack (recursión sin límite, verificado en
    // tests/qnn_crash_stack.rs); con QAIRT 2.45 (swap de los `Qnn*.dll`) debería ir.
    // NO entra en la cascada por defecto. Ver AGENTS.md §8.
    #[cfg(windows)]
    if qnn_ep_enabled() && matches!(want, Device::Auto | Device::Npu) && ensure_qnn_registered() {
        return build_qnn_session(model, threads);
    }

    // EPs disponibles Y materializables (con dispatch), en orden de prioridad.
    // Filtrar por disponibilidad real evita registrar EPs ausentes (ruido en stderr
    // y EP "primario" engañoso) y reporta el EP que de verdad se usará.
    let kinds: Vec<EpKind> = cascade_kinds(hw, want)
        .into_iter()
        .filter(|k| ep_available(*k))
        .filter(|k| dispatch_for(*k).is_some())
        .collect();
    // Si se forzó NPU/GPU y no hay ningún EP usable, error duro en vez de degradar
    // a CPU en silencio (garantía aunque el llamador no use check_forced_device).
    if kinds.is_empty() && matches!(want, Device::Npu | Device::Gpu) {
        return Err(RosettaError::Accel(format!(
            "se forzó --device {want:?} pero no hay ningún Execution Provider disponible para ese dispositivo"
        )));
    }
    // `primary` = EP PLANEADO (primero usable), no el placement real (eso lo mide el
    // profiling, `ROSETTA_ORT_PROFILE`).
    let primary = kinds.first().copied().unwrap_or(EpKind::Cpu);

    // opt#4: caché de grafo optimizado (opt-in por `ROSETTA_GRAPH_CACHE`). ORT
    // re-optimiza el grafo (Level3) en CADA arranque; serializarlo una vez y
    // recargarlo con la optimización desactivada ahorra ese coste de arranque en
    // frío. La caché es específica de (modelo, EP, versión de onnxruntime): se
    // invalida por el nombre (EP + tag de versión) y se VALIDA por el sha256 del
    // modelo ORIGEN guardado en un sidecar (`seguridad-2`, ver `cache_is_valid`).
    if let Some(cache) = graph_cache_path(model, primary) {
        let sidecar = cache_meta_path(&cache);
        if cache_is_valid(&cache, &sidecar, model) {
            match commit_session(
                &cache,
                &kinds,
                threads,
                GraphOptimizationLevel::Disable,
                None,
            ) {
                Ok(s) => return Ok((s, primary)),
                Err(e) => {
                    tracing::warn!(?cache, error = %e, "caché de grafo inválida; regenerando");
                    let _ = std::fs::remove_file(&cache);
                    let _ = std::fs::remove_file(&sidecar);
                }
            }
        }
        // (Re)generar: optimiza desde el modelo original y serializa a `cache`. El
        // sidecar (sha256 del origen) se escribe DESPUÉS de que ORT termine de
        // serializar el `.opt.onnx`, de modo que un `.opt.onnx` sin sidecar válido
        // (escritura a medias / manipulado) nunca se cargará (cierra el TOCTOU).
        // Invalidar primero el sidecar viejo evita aceptar una caché a medio generar.
        let _ = std::fs::remove_file(&sidecar);
        match commit_session(
            model,
            &kinds,
            threads,
            GraphOptimizationLevel::Level3,
            Some(&cache),
        ) {
            Ok(s) => {
                // Solo tras serializar OK: sella el sidecar con el sha del modelo.
                match sha256_file(model) {
                    Ok(sha) => {
                        if let Err(e) = std::fs::write(&sidecar, &sha) {
                            tracing::warn!(?sidecar, error = %e, "no se pudo sellar el sidecar de la caché; se regenerará la próxima vez");
                            let _ = std::fs::remove_file(&sidecar);
                        }
                    }
                    Err(e) => {
                        tracing::warn!(?model, error = %e, "no se pudo hashear el modelo para sellar la caché");
                        let _ = std::fs::remove_file(&sidecar);
                    }
                }
                return Ok((s, primary));
            }
            Err(e) => {
                tracing::warn!(?cache, error = %e, "no se pudo generar la caché; build normal");
                let _ = std::fs::remove_file(&cache);
                let _ = std::fs::remove_file(&sidecar);
            }
        }
    }

    let session = commit_session(model, &kinds, threads, GraphOptimizationLevel::Level3, None)?;
    Ok((session, primary))
}

/// Construye y compila una sesión con la cascada de EPs `kinds`, el nivel de
/// optimización `opt` y, opcionalmente, serializando el grafo optimizado a
/// `optimized_out` (opt#4). Materializa los dispatch en el momento (son baratos).
fn commit_session(
    model: &Path,
    kinds: &[EpKind],
    threads: usize,
    opt: GraphOptimizationLevel,
    optimized_out: Option<&Path>,
) -> Result<Session> {
    let dispatches: Vec<ExecutionProviderDispatch> =
        kinds.iter().filter_map(|&k| dispatch_for(k)).collect();
    let mut builder = Session::builder()
        .map_err(|e| RosettaError::Accel(e.to_string()))?
        .with_optimization_level(opt)
        .map_err(|e| RosettaError::Accel(e.to_string()))?
        .with_intra_threads(threads.max(1))
        .map_err(|e| RosettaError::Accel(e.to_string()))?;
    if let Some(out) = optimized_out {
        builder = builder
            .with_optimized_model_path(out)
            .map_err(|e| RosettaError::Accel(e.to_string()))?;
    }
    // Verificación HONESTA de placement (regla dura: "EP activo = % de nodos
    // colocados, no solo register()"): con `ROSETTA_ORT_PROFILE`, ORT escribe un
    // Chrome-trace JSON con el EP REAL por nodo. Solo bajo demanda; cero coste en prod.
    if let Ok(prefix) = std::env::var("ROSETTA_ORT_PROFILE") {
        let stem = model
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        builder = builder
            .with_profiling(format!("{prefix}_{stem}"))
            .map_err(|e| RosettaError::Accel(e.to_string()))?;
    }
    if !dispatches.is_empty() {
        builder = builder
            .with_execution_providers(dispatches)
            .map_err(|e| RosettaError::Accel(e.to_string()))?;
    }
    builder
        .commit_from_file(model)
        .map_err(|e| RosettaError::Accel(e.to_string()))
}

/// Tag de versión de onnxruntime con la que es válida la caché de grafo (parte de la
/// clave de invalidación; cambiar de versión invalida todas las cachés). Se DERIVA de
/// `ort::MINOR_VERSION` (= `ORT_API_VERSION`, 24 con la feature `api-24`); ort mismo
/// formatea las versiones como `1.{MINOR_VERSION}.x`, así que esto queda alineado y no
/// se desfasa con un literal hardcodeado (`simplificar-6`).
fn graph_cache_ort_tag() -> String {
    format!("ort1.{}", ort::MINOR_VERSION)
}

/// Ruta de la caché de grafo optimizado para `(model, primary)`, o `None` si la
/// caché no está activada (`ROSETTA_GRAPH_CACHE`). El nombre incluye el EP y el tag
/// de versión de ORT para no cruzar grafos entre devices/versiones.
fn graph_cache_path(model: &Path, primary: EpKind) -> Option<PathBuf> {
    std::env::var_os("ROSETTA_GRAPH_CACHE")?; // opt-in: sin la env, no se cachea
    let stem = model.file_stem()?.to_str()?;
    let dir = model.parent()?;
    Some(dir.join(format!(
        "{stem}.{}.{}.opt.onnx",
        primary.slug(),
        graph_cache_ort_tag()
    )))
}

/// Ruta del sidecar de integridad (`.sha256`) junto al `.opt.onnx`: contiene el
/// sha256 hex del modelo ORIGEN con el que se generó la caché.
fn cache_meta_path(cache: &Path) -> PathBuf {
    cache.with_extension("sha256")
}

/// La caché es válida (seguridad-2) si:
///  1. el `.opt.onnx` y su sidecar existen, y
///  2. el sha256 guardado en el sidecar coincide con el sha256 ACTUAL del modelo
///     origen.
///
/// El sha (no el mtime) es la fuente de verdad: detecta un modelo re-descargado/
/// distinto Y un `.opt.onnx` manipulado de forma independiente (un atacante tendría
/// que reproducir el sha del modelo legítimo para que se cargue su grafo). Como
/// `ROSETTA_GRAPH_CACHE` es opt-in (default off), el camino normal NO hashea nada.
fn cache_is_valid(cache: &Path, sidecar: &Path, model: &Path) -> bool {
    if !cache.exists() {
        return false;
    }
    let Ok(stored) = std::fs::read_to_string(sidecar) else {
        return false;
    };
    let stored = stored.trim();
    match sha256_file(model) {
        Ok(actual) => {
            let ok = !stored.is_empty() && stored.eq_ignore_ascii_case(&actual);
            if !ok {
                tracing::warn!(
                    ?cache,
                    "sha del sidecar no coincide con el modelo; se regenera la caché"
                );
            }
            ok
        }
        Err(e) => {
            tracing::warn!(?model, error = %e, "no se pudo hashear el modelo para validar la caché");
            false
        }
    }
}

/// sha256 hex (minúsculas) del fichero, por streaming (no carga el modelo entero en
/// RAM). Solo se invoca con la caché de grafo activa.
fn sha256_file(path: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

/// ¿Está activada la vía QNN experimental (plugin-EP / Windows ML)? Opt-in por env
/// porque con QAIRT 2.42 la creación de sesión desborda el stack (`QnnHtp.dll`).
#[cfg(windows)]
fn qnn_ep_enabled() -> bool {
    std::env::var_os("ROSETTA_ENABLE_QNN_EP").is_some()
}

/// Ruta del plugin-EP de QNN: `ROSETTA_QNN_EP_LIB` o `onnxruntime_providers_qnn.dll`
/// junto a la dylib de ORT (`ORT_DYLIB_PATH`).
#[cfg(windows)]
fn qnn_ep_library_path() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("ROSETTA_QNN_EP_LIB") {
        return Some(std::path::PathBuf::from(p));
    }
    let dylib = std::env::var_os("ORT_DYLIB_PATH")?;
    let cand = Path::new(&dylib)
        .parent()?
        .join("onnxruntime_providers_qnn.dll");
    cand.exists().then_some(cand)
}

/// Registra el plugin-EP de QNN en el entorno (una sola vez) y devuelve si quedó
/// disponible un `EpDevice` de tipo NPU. La biblioteca se mantiene registrada durante
/// toda la vida del proceso (`mem::forget`).
#[cfg(windows)]
fn ensure_qnn_registered() -> bool {
    use std::sync::OnceLock;
    static DONE: OnceLock<bool> = OnceLock::new();
    *DONE.get_or_init(|| {
        let Some(path) = qnn_ep_library_path() else {
            tracing::warn!(
                "ROSETTA_ENABLE_QNN_EP activo pero no se halló onnxruntime_providers_qnn.dll"
            );
            return false;
        };
        let env = match ort::environment::current() {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("entorno ort no disponible para registrar QNN: {e}");
                return false;
            }
        };
        match env.register_ep_library("QNN", &path) {
            Ok(lib) => {
                std::mem::forget(lib);
                let has_npu = env
                    .devices()
                    .any(|d| matches!(d.ty(), ort::memory::DeviceType::NPU));
                tracing::info!(?path, has_npu, "QNN plugin-EP registrado");
                has_npu
            }
            Err(e) => {
                tracing::warn!("register_ep_library(QNN) falló: {e}");
                false
            }
        }
    })
}

/// Crea la sesión por la ruta EpDevice V2 prefiriendo la NPU (Hexagon vía QNN).
#[cfg(windows)]
fn build_qnn_session(model: &Path, threads: usize) -> Result<(Session, EpKind)> {
    use ort::session::builder::AutoDevicePolicy;
    let mut builder = Session::builder()
        .map_err(|e| RosettaError::Accel(e.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| RosettaError::Accel(e.to_string()))?
        .with_intra_threads(threads.max(1))
        .map_err(|e| RosettaError::Accel(e.to_string()))?
        .with_auto_device(AutoDevicePolicy::PreferNPU)
        .map_err(|e| RosettaError::Accel(e.to_string()))?;
    if let Ok(prefix) = std::env::var("ROSETTA_ORT_PROFILE") {
        let stem = model
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        builder = builder
            .with_profiling(format!("{prefix}_{stem}"))
            .map_err(|e| RosettaError::Accel(e.to_string()))?;
    }
    let session = builder
        .commit_from_file(model)
        .map_err(|e| RosettaError::Accel(e.to_string()))?;
    Ok((session, EpKind::Qnn))
}
