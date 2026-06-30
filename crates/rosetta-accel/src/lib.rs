//! Detección de hardware y cascada de Execution Providers (NPU → GPU → CPU).
//!
//! `detect_hw()` detecta SO/arch/CPU/NPU. La cascada **real** de EPs que se
//! registran en `ort` vive en [`ep`]; aquí están los tipos compartidos y el
//! [`EpKind`] (etiquetas para logs y `info`).

pub mod ep;

use tracing::debug;

/// Sistema operativo detectado.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    Windows,
    Linux,
    Mac,
    Other,
}

/// Arquitectura detectada.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    X64,
    Arm64,
    Other,
}

/// Fabricante de CPU/GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vendor {
    Intel,
    Amd,
    Qualcomm,
    Apple,
    Nvidia,
    Unknown,
}

/// Preferencia de dispositivo del usuario (`--device`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Auto,
    Npu,
    Gpu,
    Cpu,
}

/// Clase de Execution Provider, para etiquetas/logs. La cascada real son
/// `ort::ep::ExecutionProviderDispatch` (ver [`ep::build_session`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpKind {
    Qnn,
    /// OpenVINO en NPU Intel.
    OpenVino,
    /// OpenVINO en GPU Intel.
    OpenVinoGpu,
    DirectMlNpu,
    DirectMlGpu,
    CoreMl,
    /// Vitis AI (NPU AMD XDNA). Scaffold: no cableado todavía (F8b).
    Vitis,
    Cuda,
    TensorRt,
    Xnnpack,
    Cpu,
}

impl EpKind {
    /// Etiqueta legible. Fuente única: derivada de la tabla [`ep::ep_spec`].
    pub fn label(self) -> &'static str {
        ep::ep_spec(self).label
    }

    /// Slug alfanumérico estable para nombrar artefactos (caché de grafo, etc.).
    /// Derivado de la etiqueta: minúsculas, solo `[a-z0-9]`.
    pub fn slug(self) -> String {
        self.label()
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect()
    }
}

/// Tipo de dispositivo de OpenVINO. Reemplaza el string-literal `"NPU"`/`"GPU"`
/// (un typo no debe compilar); implementa [`fmt::Display`] con el literal exacto
/// que espera el EP de OpenVINO (`with_device_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OvDevice {
    Npu,
    Gpu,
}

impl std::fmt::Display for OvDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            OvDevice::Npu => "NPU",
            OvDevice::Gpu => "GPU",
        })
    }
}

/// Perfil de hardware detectado.
#[derive(Debug, Clone)]
pub struct HwProfile {
    pub os: Os,
    pub arch: Arch,
    pub cpu_brand: String,
    pub cpu_vendor: Vendor,
    pub cpu_cores: usize,
    pub ram_gb: f64,
    pub gpu_vendor: Option<Vendor>,
    pub has_npu: bool,
    pub npu_name: Option<String>,
}

fn detect_os() -> Os {
    match std::env::consts::OS {
        "windows" => Os::Windows,
        "linux" => Os::Linux,
        "macos" => Os::Mac,
        _ => Os::Other,
    }
}

fn detect_arch() -> Arch {
    match std::env::consts::ARCH {
        "x86_64" => Arch::X64,
        "aarch64" => Arch::Arm64,
        _ => Arch::Other,
    }
}

fn vendor_from_brand(brand: &str) -> Vendor {
    let b = brand.to_lowercase();
    if b.contains("qualcomm") || b.contains("snapdragon") || b.contains("oryon") {
        Vendor::Qualcomm
    } else if b.contains("intel") {
        Vendor::Intel
    } else if b.contains("amd") || b.contains("ryzen") {
        Vendor::Amd
    } else if b.contains("apple") {
        Vendor::Apple
    } else {
        Vendor::Unknown
    }
}

/// Vendor de CPU: en x86 usa CPUID (fiable); en otras arch, deduce del brand.
fn cpu_vendor_detect(brand: &str) -> Vendor {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(vf) = raw_cpuid::CpuId::new().get_vendor_info() {
            let s = vf.as_str();
            if s.contains("Intel") {
                return Vendor::Intel;
            }
            if s.contains("AMD") {
                return Vendor::Amd;
            }
        }
    }
    vendor_from_brand(brand)
}

/// Heurística de presencia de NPU (la verificación real es vía placement en [`ep`]).
fn detect_npu(os: Os, arch: Arch, vendor: Vendor, brand: &str) -> (bool, Option<String>) {
    let b = brand.to_lowercase();
    match vendor {
        Vendor::Qualcomm => (true, Some("Qualcomm Hexagon NPU".to_string())),
        Vendor::Apple if arch == Arch::Arm64 || os == Os::Mac => {
            (true, Some("Apple Neural Engine".to_string()))
        }
        Vendor::Intel if b.contains("core ultra") || b.contains("ultra") => {
            (true, Some("Intel AI Boost NPU".to_string()))
        }
        Vendor::Amd if b.contains("ryzen ai") || b.contains("hx") => {
            (true, Some("AMD Ryzen AI (XDNA) NPU".to_string()))
        }
        _ => (false, None),
    }
}

/// Obtiene el nombre comercial de la CPU por plataforma (más fiable que sysinfo
/// en Windows ARM64, donde el brand suele venir vacío).
fn cpu_brand_platform() -> Option<String> {
    #[cfg(target_os = "windows")]
    {
        use windows_registry::LOCAL_MACHINE;
        let key = LOCAL_MACHINE
            .open(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            .ok()?;
        let name = key.get_string("ProcessorNameString").ok()?;
        let name = name.trim().to_string();
        return if name.is_empty() { None } else { Some(name) };
    }
    #[cfg(target_os = "linux")]
    {
        let txt = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        for line in txt.lines() {
            if let Some((k, v)) = line.split_once(':') {
                let (k, v) = (k.trim(), v.trim());
                if (k == "model name" || k == "Model" || k == "Hardware") && !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
        return None;
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()?;
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return Some(s);
            }
        }
        return None;
    }
    #[allow(unreachable_code)]
    {
        None
    }
}

/// Detecta el vendor de la GPU de forma nativa (sin ORT: `env.devices()` no está
/// en onnxruntime ≤1.24/ort rc.12). En Linux lee los PCI vendor IDs de
/// `/sys/class/drm`; en el resto asume GPU integrada del mismo vendor que la CPU
/// (Adreno en Qualcomm, iGPU Intel, Radeon en APU AMD, GPU Apple en Apple Silicon).
/// La GPU discreta NVIDIA en Windows se difiere a DXCore/`env.devices()` (futuro).
fn detect_gpu_vendor(cpu_vendor: Vendor) -> Option<Vendor> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(rd) = std::fs::read_dir("/sys/class/drm") {
            for e in rd.flatten() {
                let vpath = e.path().join("device/vendor");
                if let Ok(v) = std::fs::read_to_string(&vpath) {
                    match v.trim() {
                        "0x10de" => return Some(Vendor::Nvidia),
                        "0x1002" => return Some(Vendor::Amd),
                        "0x8086" => return Some(Vendor::Intel),
                        _ => {}
                    }
                }
            }
        }
    }
    match cpu_vendor {
        Vendor::Qualcomm | Vendor::Intel | Vendor::Amd | Vendor::Apple => Some(cpu_vendor),
        _ => None,
    }
}

/// Detecta el hardware de esta máquina.
pub fn detect_hw() -> HwProfile {
    use sysinfo::System;

    let os = detect_os();
    let arch = detect_arch();

    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu_brand = cpu_brand_platform()
        .or_else(|| {
            sys.cpus()
                .first()
                .map(|c| c.brand().trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| "CPU desconocida".to_string());
    let cpu_cores = sys.cpus().len();
    let ram_gb = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);

    let cpu_vendor = cpu_vendor_detect(&cpu_brand);
    let (has_npu, npu_name) = detect_npu(os, arch, cpu_vendor, &cpu_brand);
    let gpu_vendor = detect_gpu_vendor(cpu_vendor);

    let profile = HwProfile {
        os,
        arch,
        cpu_brand,
        cpu_vendor,
        cpu_cores,
        ram_gb,
        gpu_vendor,
        has_npu,
        npu_name,
    };
    debug!(?profile, "hardware detectado");
    profile
}

/// Nombre del host (para identificar la máquina en la telemetría).
pub fn host_name() -> String {
    sysinfo::System::host_name().unwrap_or_default()
}

/// RSS (memoria residente) del proceso actual en MB; 0.0 si no se puede medir.
/// Muestra aproximada del pico de memoria al final de una corrida (telemetría).
pub fn process_rss_mb() -> f64 {
    use sysinfo::{ProcessesToUpdate, System, get_current_pid};
    let Ok(pid) = get_current_pid() else {
        return 0.0;
    };
    let mut sys = System::new();
    sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    sys.process(pid)
        .map(|p| p.memory() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ep_smoke_cascade_qualcomm() {
        let hw = HwProfile {
            os: Os::Windows,
            arch: Arch::Arm64,
            cpu_brand: "Snapdragon X".into(),
            cpu_vendor: Vendor::Qualcomm,
            cpu_cores: 8,
            ram_gb: 32.0,
            gpu_vendor: None,
            has_npu: true,
            npu_name: Some("Qualcomm Hexagon NPU".into()),
        };
        // Auto en Qualcomm ARM64 (QNN bloqueado por bug de runtime): GPU
        // (DirectML/Adreno) + CPU. La aceleración pasa por DirectML.
        let kinds = ep::cascade_kinds(&hw, Device::Auto);
        assert_eq!(kinds.first(), Some(&EpKind::DirectMlGpu));
        assert!(kinds.contains(&EpKind::Cpu));
        // Forzar CPU: solo CPU.
        assert_eq!(ep::cascade_kinds(&hw, Device::Cpu), vec![EpKind::Cpu]);
        // Forzar GPU: solo DirectML-GPU, sin CPU.
        assert_eq!(
            ep::cascade_kinds(&hw, Device::Gpu),
            vec![EpKind::DirectMlGpu]
        );
        // Forzar NPU: vacío mientras QNN esté bloqueado (DirectML-NPU sin verificar).
        assert!(ep::cascade_kinds(&hw, Device::Npu).is_empty());
    }

    #[test]
    fn ep_smoke_detect_runs() {
        let hw = detect_hw();
        assert!(!ep::cascade_kinds(&hw, Device::Auto).is_empty());
    }
}
