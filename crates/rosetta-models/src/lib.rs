//! Catálogo, descarga y caché de modelos ONNX.
//!
//! El catálogo vive en `models.toml` (embebido). Cada modelo tiene uno o más
//! archivos que se obtienen por URL directa o extrayendo un miembro de un
//! `.tar.bz2`. La descarga usa `ureq` (TLS por SChannel/native-tls, sin `ring`),
//! la descompresión `bzip2-rs` + `tar` (Rust puro) y la verificación `sha2`.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

const MODELS_TOML: &str = include_str!("../models.toml");

/// Límite de tamaño por archivo descargado o extraído (defensa anti-DoS frente a
/// descargas comprometidas o bombas de descompresión). Holgado para los modelos
/// reales (decenas de MB) pero acotado.
const MAX_FILE_BYTES: u64 = 2_000_000_000; // 2 GB

/// Error del gestor de modelos.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("modelo desconocido: {0}")]
    UnknownModel(String),
    #[error("descarga de {url}: {source}")]
    Download {
        url: String,
        source: Box<ureq::Error>,
    },
    #[error("E/S en {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("sha256 de {file}: esperado {expected}, obtenido {got}")]
    Sha256Mismatch {
        file: String,
        expected: String,
        got: String,
    },
    #[error("catálogo inválido: {0}")]
    Catalog(String),
    #[error("descompresión: {0}")]
    Archive(String),
    #[error("el modelo '{0}' no define sha256 en el catálogo (verificación obligatoria)")]
    MissingSha256(String),
    #[error("{what} excede el límite de {limit} bytes (posible descarga maliciosa)")]
    TooLarge { what: String, limit: u64 },
}

pub type Result<T> = std::result::Result<T, ModelError>;

#[derive(Debug, Deserialize)]
struct Catalog {
    #[serde(default)]
    model: Vec<Model>,
}

/// Un modelo del catálogo (uno o más archivos).
#[derive(Debug, Clone, Deserialize)]
pub struct Model {
    pub id: String,
    pub kind: String,
    pub license: String,
    #[serde(default)]
    pub file: Vec<ModelFile>,
}

/// Un archivo de un modelo, con su origen de descarga.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelFile {
    /// Nombre local del archivo en la caché del modelo.
    pub name: String,
    /// Descarga directa.
    #[serde(default)]
    pub url: Option<String>,
    /// Descarga de un `.tar.bz2` del que se extrae `archive_member`.
    #[serde(default)]
    pub archive_url: Option<String>,
    #[serde(default)]
    pub archive_member: Option<String>,
    /// SHA-256 esperado (hex). Si está, se verifica tras descargar.
    #[serde(default)]
    pub sha256: Option<String>,
}

/// Catálogo completo de modelos.
pub fn catalog() -> Vec<Model> {
    let c: Catalog = toml::from_str(MODELS_TOML).expect("models.toml embebido válido");
    c.model
}

/// Busca un modelo por id.
pub fn find(id: &str) -> Option<Model> {
    catalog().into_iter().find(|m| m.id == id)
}

/// Raíz de la caché de modelos: `ROSETTA_MODELS_DIR`, o `models/` (cwd, desarrollo),
/// o `models/` junto al ejecutable (`target/<profile>/../../models`), o la caché
/// del usuario (`directories`) en una instalación.
pub fn cache_root() -> PathBuf {
    if let Some(d) = std::env::var_os("ROSETTA_MODELS_DIR") {
        return PathBuf::from(d);
    }
    let local = PathBuf::from("models");
    if local.exists() {
        return local;
    }
    // Fallback relativo al ejecutable: binario en `target/<profile>/`, modelos en
    // `<workspace>/models`. Permite correr el binario compilado sin definir nada.
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let p = dir.join("../../models");
        if p.exists() {
            return p;
        }
    }
    directories::ProjectDirs::from("dev", "Rosetta", "rosetta")
        .map(|p| p.cache_dir().join("models"))
        .unwrap_or(local)
}

/// Directorio local de un modelo.
pub fn model_dir(id: &str) -> PathBuf {
    cache_root().join(id)
}

/// ¿Están presentes en disco todos los archivos del modelo?
pub fn is_present(id: &str) -> bool {
    match find(id) {
        Some(m) => {
            let dir = model_dir(id);
            !m.file.is_empty() && m.file.iter().all(|f| dir.join(&f.name).is_file())
        }
        None => false,
    }
}

/// Descarga (si falta) y verifica un modelo; devuelve su directorio local.
pub fn ensure_model(id: &str) -> Result<PathBuf> {
    let model = find(id).ok_or_else(|| ModelError::UnknownModel(id.to_string()))?;
    let dir = model_dir(id);
    fs::create_dir_all(&dir).map_err(|e| io_err(&dir, e))?;
    for f in &model.file {
        // La verificación de integridad es OBLIGATORIA: sin sha256 no se descarga
        // (evita ejecutar binarios ONNX sin verificar, aunque el upstream cambie).
        let expected = f
            .sha256
            .as_deref()
            .ok_or_else(|| ModelError::MissingSha256(id.to_string()))?;
        let dest = dir.join(&f.name);
        let marker = verified_marker_path(&dir, &f.name);
        // Fast-path: si un marcador confirma que ESTE archivo (mismo mtime+size) ya
        // se verificó contra este sha, saltar el re-hash (caro: ~1 GB por modelo,
        // 15-48 s por arranque). El marcador es solo caché; la integridad la da el
        // sha verificado al descargar (abajo).
        if dest.is_file() && marker_matches(&marker, &dest, expected) {
            continue;
        }
        // Sin marcador (p. ej. modelo colocado a mano): verificación completa una
        // vez y se escribe el marcador para las siguientes corridas.
        if dest.is_file() && sha256_file(&dest)? == expected {
            write_marker(&marker, &dest, expected);
            continue;
        }
        if dest.is_file() {
            tracing::warn!(file = %f.name, "sha256 no coincide; re-descargando");
        }
        tracing::info!(model = %id, file = %f.name, "descargando modelo");
        download_file(f, &dest)?;
        let got = sha256_file(&dest)?;
        if got != expected {
            let _ = fs::remove_file(&dest); // no dejar un binario sin verificar en caché
            let _ = fs::remove_file(&marker);
            return Err(ModelError::Sha256Mismatch {
                file: f.name.clone(),
                expected: expected.to_string(),
                got,
            });
        }
        write_marker(&marker, &dest, expected);
    }
    Ok(dir)
}

/// Descarga (si falta) un modelo de **un solo archivo** y devuelve la ruta
/// completa a ese `.onnx`. Evita hardcodear el nombre del archivo en los
/// llamadores (silero-vad, gtcrn-simple, campplus, pyannote-segmentation): la
/// fuente de verdad del nombre es el catálogo (`models.toml`).
///
/// Error si el modelo no existe en el catálogo o si declara 0 o más de 1
/// archivo (para esos casos multi-archivo, p. ej. Parakeet/Whisper, el motor
/// resuelve los nombres con su propio `find_first`).
pub fn ensure_model_file(id: &str) -> Result<PathBuf> {
    let model = find(id).ok_or_else(|| ModelError::UnknownModel(id.to_string()))?;
    if model.file.len() != 1 {
        return Err(ModelError::Catalog(format!(
            "el modelo '{id}' tiene {} archivos; `ensure_model_file` exige exactamente 1",
            model.file.len()
        )));
    }
    let dir = ensure_model(id)?;
    let name = &model.file[0].name;
    Ok(dir.join(name))
}

/// Ruta del marcador de verificación de un archivo (`.verified-<name>`).
fn verified_marker_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!(".verified-{name}"))
}

/// `(mtime_secs, size)` del archivo, o `None` si no se puede leer.
fn file_sig(path: &Path) -> Option<(u64, u64)> {
    let m = fs::metadata(path).ok()?;
    let mtime = m
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some((mtime, m.len()))
}

/// ¿El marcador (`sha:mtime:size`) coincide con el archivo y el sha esperado?
fn marker_matches(marker: &Path, dest: &Path, expected_sha: &str) -> bool {
    let Some((mtime, size)) = file_sig(dest) else {
        return false;
    };
    let Ok(content) = fs::read_to_string(marker) else {
        return false;
    };
    let mut parts = content.trim().split(':');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(sha), Some(mt), Some(sz)) => {
            sha == expected_sha
                && mt.parse::<u64>().ok() == Some(mtime)
                && sz.parse::<u64>().ok() == Some(size)
        }
        _ => false,
    }
}

/// Escribe el marcador `sha:mtime:size` para `dest` (best-effort).
fn write_marker(marker: &Path, dest: &Path, sha: &str) {
    if let Some((mtime, size)) = file_sig(dest) {
        let _ = fs::write(marker, format!("{sha}:{mtime}:{size}"));
    }
}

/// Verifica los SHA-256 presentes en el catálogo para un modelo ya descargado.
/// Devuelve `true` si todo (lo verificable) coincide.
pub fn verify(id: &str) -> Result<bool> {
    let model = find(id).ok_or_else(|| ModelError::UnknownModel(id.to_string()))?;
    let dir = model_dir(id);
    for f in &model.file {
        let dest = dir.join(&f.name);
        if !dest.is_file() {
            return Ok(false);
        }
        if let Some(expected) = &f.sha256
            && &sha256_file(&dest)? != expected
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Borra del disco los archivos cacheados de un modelo.
pub fn remove(id: &str) -> Result<()> {
    let dir = model_dir(id);
    if dir.exists() {
        fs::remove_dir_all(&dir).map_err(|e| io_err(&dir, e))?;
    }
    Ok(())
}

/// Tamaño total en bytes de un directorio (recursivo). Best-effort: ignora errores
/// puntuales de lectura y cuenta lo que puede.
fn dir_size(path: &Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = fs::read_dir(path) {
        for e in rd.flatten() {
            match e.file_type() {
                Ok(ft) if ft.is_dir() => total += dir_size(&e.path()),
                Ok(_) => total += e.metadata().map(|m| m.len()).unwrap_or(0),
                Err(_) => {}
            }
        }
    }
    total
}

/// Borra TODA la caché de modelos (`cache_root()`): los modelos, los marcadores
/// `.verified-*`, las cachés de grafo `.opt.onnx`/`.sha256` y cualquier otro
/// artefacto. Pensada para ejecutarse antes de desinstalar y no dejar la caché
/// (varios GB) huérfana. Devuelve `(ruta borrada, bytes liberados)`.
pub fn clean() -> Result<(PathBuf, u64)> {
    let root = cache_root();
    if !root.exists() {
        return Ok((root, 0));
    }
    let freed = dir_size(&root);
    fs::remove_dir_all(&root).map_err(|e| io_err(&root, e))?;
    Ok((root, freed))
}

// ─── ONNX Runtime: descarga de la librería precompilada ──────────────────────
// Es una dylib OFICIAL verificada por sha256 (no compila C), análoga a la descarga
// de modelos. El CLI la usa como último recurso cuando no la encuentra junto al
// binario. La verificación sha256 la hace MÁS segura que cargar una dylib local sin
// verificar (cf. seguridad-1).

/// Versión de ONNX Runtime que distribuye el proyecto (techo in-process de `ort`
/// con `api-24`).
pub const ORT_VERSION: &str = "1.24.4";

/// Nombre de la librería compartida de ONNX Runtime por plataforma.
#[cfg(windows)]
const ORT_DYLIB_NAME: &str = "onnxruntime.dll";
#[cfg(target_os = "linux")]
const ORT_DYLIB_NAME: &str = "libonnxruntime.so";
#[cfg(target_os = "macos")]
const ORT_DYLIB_NAME: &str = "libonnxruntime.dylib";

struct OrtAsset {
    url: String,
    sha256: &'static str,
}

/// Asset oficial de ONNX Runtime 1.24.4 para la plataforma actual, o `None` si no
/// hay build oficial publicado (p. ej. macOS x86_64: aporta `ORT_DYLIB_PATH`).
fn ort_asset() -> Option<OrtAsset> {
    let base = "https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/";
    let (file, sha256): (&str, &str) = if cfg!(all(target_os = "windows", target_arch = "aarch64"))
    {
        (
            "onnxruntime-win-arm64-1.24.4.zip",
            "47dc80aa39da792271af10be5993919536a4dab0965ec1e6043ef37f1df7a693",
        )
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        (
            "onnxruntime-win-x64-1.24.4.zip",
            "d2319fddfb6ea4db99ccc4b60c85c517bcd855721f5daa6a06d40d7cb2ee2357",
        )
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        (
            "onnxruntime-linux-x64-1.24.4.tgz",
            "3a211fbea252c1e66290658f1b735b772056149f28321e71c308942cdb54b747",
        )
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        (
            "onnxruntime-linux-aarch64-1.24.4.tgz",
            "866109a9248d057671a039b9d725be4bd86888e3754140e6701ec621be9d4d7e",
        )
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        (
            "onnxruntime-osx-arm64-1.24.4.tgz",
            "93787795f47e1eee369182e43ed51b9e5da0878ab0346aecf4258979b8bba989",
        )
    } else {
        return None;
    };
    Some(OrtAsset {
        url: format!("{base}{file}"),
        sha256,
    })
}

/// Asegura la librería de ONNX Runtime en la caché del usuario: si falta, descarga
/// el build OFICIAL (verificando sha256) y extrae la dylib; devuelve su ruta. No
/// compila C: es una librería precompilada, como los modelos.
pub fn ensure_ort_runtime() -> Result<PathBuf> {
    let dir = cache_root().join(".ort-runtime").join(ORT_VERSION);
    let dylib = dir.join(ORT_DYLIB_NAME);
    if dylib.exists() {
        return Ok(dylib);
    }
    let Some(asset) = ort_asset() else {
        return Err(ModelError::Catalog(format!(
            "no hay build oficial de ONNX Runtime {ORT_VERSION} para esta plataforma; \
             define ORT_DYLIB_PATH con la ruta a la librería"
        )));
    };
    fs::create_dir_all(&dir).map_err(|e| io_err(&dir, e))?;
    eprintln!(
        "ONNX Runtime no encontrado; descargando el build oficial {ORT_VERSION} a la caché (solo la primera vez)…"
    );
    // OJO: el nombre NO debe acabar en `.part` (http_get_to usa internamente
    // `<dest>.part` y su guard lo borraría tras el rename).
    let tmp = dir.join("ort-runtime-download");
    let _guard = PartGuard(&tmp);
    http_get_to(&asset.url, &tmp)?;
    let got = sha256_file(&tmp)?;
    if got != asset.sha256 {
        return Err(ModelError::Catalog(format!(
            "el sha256 de ONNX Runtime no coincide (esperado {}, obtenido {got})",
            asset.sha256
        )));
    }
    #[cfg(windows)]
    extract_ort_zip(&tmp, &dir)?;
    #[cfg(not(windows))]
    extract_ort_tgz(&tmp, &dir)?;
    if !dylib.exists() {
        return Err(ModelError::Catalog(format!(
            "la librería {ORT_DYLIB_NAME} no apareció tras extraer ONNX Runtime"
        )));
    }
    Ok(dylib)
}

/// Extrae de un `.zip` de ONNX Runtime (Windows) las DLLs bajo `*/lib/` de forma
/// plana en `dir`.
#[cfg(windows)]
fn extract_ort_zip(archive: &Path, dir: &Path) -> Result<()> {
    use std::io::Read;
    let f = fs::File::open(archive).map_err(|e| io_err(archive, e))?;
    let mut zip = zip::ZipArchive::new(f)
        .map_err(|e| ModelError::Archive(format!("zip ONNX Runtime: {e}")))?;
    for i in 0..zip.len() {
        let mut file = zip
            .by_index(i)
            .map_err(|e| ModelError::Archive(format!("zip ONNX Runtime: {e}")))?;
        let Some(name) = file.enclosed_name() else {
            continue;
        };
        let in_lib = name.components().any(|c| c.as_os_str() == "lib");
        let fname = name
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if in_lib && fname.ends_with(".dll") {
            let dest = dir.join(&fname);
            let mut limited = (&mut file).take(MAX_FILE_BYTES + 1);
            let mut out = fs::File::create(&dest).map_err(|e| io_err(&dest, e))?;
            let copied = std::io::copy(&mut limited, &mut out).map_err(|e| io_err(&dest, e))?;
            if copied > MAX_FILE_BYTES {
                return Err(ModelError::TooLarge {
                    what: fname,
                    limit: MAX_FILE_BYTES,
                });
            }
        }
    }
    Ok(())
}

/// Extrae de un `.tar.gz` de ONNX Runtime (Linux/macOS) las librerías bajo `*/lib/`
/// de forma plana en `dir`, y recrea el nombre canónico (los `.so`/`.dylib`
/// versionados vienen como symlink en el tar).
#[cfg(not(windows))]
fn extract_ort_tgz(archive: &Path, dir: &Path) -> Result<()> {
    use std::io::Read;
    let f = fs::File::open(archive).map_err(|e| io_err(archive, e))?;
    let gz = flate2::read::GzDecoder::new(f);
    let mut tar = tar::Archive::new(gz);
    for entry in tar.entries().map_err(|e| io_err(archive, e))? {
        let mut entry = entry.map_err(|e| io_err(archive, e))?;
        if !entry.header().entry_type().is_file() {
            continue; // los symlinks de versión se recrean por copia abajo
        }
        let path = entry.path().map_err(|e| io_err(archive, e))?.into_owned();
        let in_lib = path.components().any(|c| c.as_os_str() == "lib");
        let fname = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let is_lib =
            fname.contains("onnxruntime") && (fname.contains(".so") || fname.contains(".dylib"));
        if in_lib && is_lib {
            let dest = dir.join(&fname);
            let mut limited = entry.by_ref().take(MAX_FILE_BYTES + 1);
            let mut out = fs::File::create(&dest).map_err(|e| io_err(&dest, e))?;
            let copied = std::io::copy(&mut limited, &mut out).map_err(|e| io_err(&dest, e))?;
            if copied > MAX_FILE_BYTES {
                return Err(ModelError::TooLarge {
                    what: fname,
                    limit: MAX_FILE_BYTES,
                });
            }
        }
    }
    // Nombre canónico (libonnxruntime.so / .dylib): copia el archivo real versionado.
    let canon = dir.join(ORT_DYLIB_NAME);
    if !canon.exists()
        && let Ok(rd) = fs::read_dir(dir)
    {
        for e in rd.flatten() {
            let n = e.file_name();
            let n = n.to_string_lossy();
            let is_main = n.starts_with("libonnxruntime")
                && !n.contains("providers")
                && (n.contains(".so") || n.contains(".dylib"));
            if is_main {
                fs::copy(e.path(), &canon).map_err(|er| io_err(&canon, er))?;
                break;
            }
        }
    }
    Ok(())
}

fn download_file(f: &ModelFile, dest: &Path) -> Result<()> {
    if let Some(url) = &f.url {
        http_get_to(url, dest)
    } else if let (Some(au), Some(member)) = (&f.archive_url, &f.archive_member) {
        let tmp = dest.with_extension("tar.bz2.part");
        http_get_to(au, &tmp)?;
        let r = extract_tar_bz2_member(&tmp, member, dest);
        let _ = fs::remove_file(&tmp);
        r
    } else {
        Err(ModelError::Catalog(format!(
            "archivo {} sin `url` ni `archive_url`/`archive_member`",
            f.name
        )))
    }
}

/// Agente HTTP con el proveedor TLS de la plataforma: `native-tls` (SChannel) en
/// Windows, `rustls` en el resto. El proveedor debe casar con la feature de
/// `ureq` activada por target (ver Cargo.toml), o falla en runtime.
fn http_agent() -> &'static ureq::Agent {
    use std::sync::OnceLock;
    use ureq::tls::{TlsConfig, TlsProvider};
    #[cfg(windows)]
    let provider = TlsProvider::NativeTls;
    #[cfg(not(windows))]
    let provider = TlsProvider::Rustls;
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            // Solo https: todas las URLs del catálogo lo son; un redirect (los CDN
            // de HuggingFace/GitHub redirigen) o un MITM del primer hop que intente
            // degradar a http en claro se vuelve error duro en vez de descargar sin
            // cifrar. La integridad ya la ancla el sha256 obligatorio; esto cierra
            // el canal.
            .https_only(true)
            .tls_config(TlsConfig::builder().provider(provider).build())
            .build()
            .into()
    })
}

/// Limpia el archivo `.part` si la descarga no llega a renombrarse.
struct PartGuard<'a>(&'a Path);
impl Drop for PartGuard<'_> {
    fn drop(&mut self) {
        let _ = fs::remove_file(self.0);
    }
}

fn http_get_to(url: &str, dest: &Path) -> Result<()> {
    let res = http_agent()
        .get(url)
        .call()
        .map_err(|e| ModelError::Download {
            url: url.to_string(),
            source: Box::new(e),
        })?;
    let tmp = dest.with_extension("part");
    let _guard = PartGuard(&tmp); // borra el .part si algo falla antes del rename
    let mut file = fs::File::create(&tmp).map_err(|e| io_err(&tmp, e))?;
    let mut reader = res.into_body().into_reader().take(MAX_FILE_BYTES + 1);
    let copied = std::io::copy(&mut reader, &mut file).map_err(|e| io_err(&tmp, e))?;
    if copied > MAX_FILE_BYTES {
        return Err(ModelError::TooLarge {
            what: format!("descarga de {url}"),
            limit: MAX_FILE_BYTES,
        });
    }
    file.sync_all().ok();
    drop(file);
    fs::rename(&tmp, dest).map_err(|e| io_err(dest, e))?;
    Ok(())
}

fn extract_tar_bz2_member(archive: &Path, member: &str, dest: &Path) -> Result<()> {
    // El miembro viene del catálogo (de confianza), pero se valida por defensa
    // en profundidad contra path traversal / zip-slip.
    if member.contains("..") || member.starts_with('/') || member.starts_with('\\') {
        return Err(ModelError::Archive(format!("miembro inseguro: {member}")));
    }
    let f = fs::File::open(archive).map_err(|e| io_err(archive, e))?;
    let decoder = bzip2_rs::DecoderReader::new(f);
    let mut tar = tar::Archive::new(decoder);
    let entries = tar
        .entries()
        .map_err(|e| ModelError::Archive(e.to_string()))?;
    for entry in entries {
        let mut entry = entry.map_err(|e| ModelError::Archive(e.to_string()))?;
        let path = entry
            .path()
            .map_err(|e| ModelError::Archive(e.to_string()))?;
        if path.to_string_lossy().replace('\\', "/") == member {
            let mut out = fs::File::create(dest).map_err(|e| io_err(dest, e))?;
            let mut limited = entry.by_ref().take(MAX_FILE_BYTES + 1);
            let copied = std::io::copy(&mut limited, &mut out).map_err(|e| io_err(dest, e))?;
            if copied > MAX_FILE_BYTES {
                let _ = fs::remove_file(dest);
                return Err(ModelError::TooLarge {
                    what: format!("miembro {member}"),
                    limit: MAX_FILE_BYTES,
                });
            }
            return Ok(());
        }
    }
    Err(ModelError::Archive(format!(
        "miembro `{member}` no encontrado en {}",
        archive.display()
    )))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).map_err(|e| io_err(path, e))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf).map_err(|e| io_err(path, e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

fn io_err(path: &Path, source: std::io::Error) -> ModelError {
    ModelError::Io {
        path: path.display().to_string(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_parses_and_has_known_models() {
        let c = catalog();
        assert!(c.iter().any(|m| m.id == "silero-vad"));
        assert!(c.iter().any(|m| m.id == "dpdfnet2"));
        // Cada archivo tiene un origen de descarga.
        for m in &c {
            for f in &m.file {
                assert!(
                    f.url.is_some() || (f.archive_url.is_some() && f.archive_member.is_some()),
                    "modelo {} archivo {} sin origen",
                    m.id,
                    f.name
                );
            }
        }
    }
}
