//! Modo lote: transcribe varios archivos en paralelo a nivel de **archivo**.
//!
//! El paralelismo es por archivo, no por bloque: cada worker tiene su propio
//! motor (y VAD/denoiser/diarizador) y procesa archivos completos de forma
//! secuencial. Así se evita la sobre-suscripción del intra-op de ONNX (que ya
//! usa varios hilos por sesión). Para un único acelerador (GPU/NPU) se fuerza
//! `jobs = 1`, porque varios archivos a la vez solo competirían por él.
//!
//! Se lanzan **exactamente `jobs` workers** (no uno por archivo): cada worker
//! construye su motor una sola vez y drena una cola compartida de archivos.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use rayon::prelude::*;

use rosetta_accel::{Device, EpKind, HwProfile};
use rosetta_audio::{Denoiser, GtcrnDenoiser, SileroVad};
use rosetta_core::OutputFormat;
use rosetta_diarize::Diarizer;

use crate::Cli;

/// Extensiones de medios reconocidas al expandir directorios.
const MEDIA_EXT: &[&str] = &[
    "wav", "mp3", "flac", "m4a", "aac", "ogg", "oga", "opus", "wma", "mka", "mp4", "mkv", "mov",
    "webm", "avi", "m4v", "mpg", "mpeg", "ts",
];

/// Configuración inmutable compartida por todos los workers del lote.
struct BatchCtx {
    hw: HwProfile,
    device: Device,
    /// Hilos por motor (total / jobs) para no saturar el intra-op de ONNX.
    threads: usize,
    model: String,
    model_dir: PathBuf,
    language: String,
    init_prompt: String,
    /// Ruta a `gtcrn_simple.onnx` si se pidió denoise/realce.
    denoise_model: Option<PathBuf>,
    /// Ruta a `silero_vad.onnx` (para audio largo).
    vad_model: PathBuf,
    diar: Option<DiarCfg>,
    format: OutputFormat,
    timestamps: bool,
    out_dir: Option<PathBuf>,
    force: bool,
    /// Red de seguridad (seguridad-5b): contabilidad del PCM acumulado en vuelo
    /// entre todos los workers. `ram_por_worker` solo modela la RAM ESTÁTICA del
    /// motor (sesión ONNX), no el audio decodificado; varios archivos largos a la
    /// vez pueden inflar el residente muy por encima de esa estimación.
    pcm_budget: PcmBudget,
}

/// Cota de la RAM ocupada por PCM decodificado simultáneamente en el lote.
///
/// No cambia el camino normal: archivos cortos caben de sobra y nunca esperan.
/// Solo acota la carga PATOLÓGICA (varios workers decodificando archivos muy
/// largos a la vez): si una reserva nueva excediera el presupuesto **y** ya hay
/// PCM de otros workers en vuelo, ese worker espera a que baje el residente en
/// vez de amontonar otro buffer gigante. El que ya tiene su buffer nunca se
/// bloquea (no hay deadlock: siempre puede progresar al menos un worker).
struct PcmBudget {
    /// Bytes de PCM actualmente en vuelo (sumados por todos los workers).
    in_flight: AtomicU64,
    /// Techo de bytes de PCM en vuelo a la vez (derivado de la RAM física).
    max_bytes: u64,
    /// Serializa las esperas para que el aviso/log no se repita en tromba.
    gate: Mutex<()>,
}

impl PcmBudget {
    /// Presupuesto de PCM: el 20% de la RAM física (los motores ya reservan el
    /// grueso vía `ram_por_worker`). Mínimo 256 MB para no estrangular en equipos
    /// con poca RAM, donde `jobs` ya viene recortado.
    fn from_hw(hw: &HwProfile) -> Self {
        let by_ram = (hw.ram_gb * 0.20 * 1e9) as u64;
        let max_bytes = by_ram.max(256_000_000);
        PcmBudget {
            in_flight: AtomicU64::new(0),
            max_bytes,
            gate: Mutex::new(()),
        }
    }

    /// Reserva `bytes` de PCM, esperando si hace falta para no exceder el techo
    /// mientras haya otros buffers en vuelo. Devuelve un guard RAII que descuenta
    /// los bytes al soltarse (fin del procesamiento del archivo).
    fn reserve(&self, bytes: u64) -> PcmReservation<'_> {
        // Si la reserva sola ya supera el techo (archivo gigantesco), no hay nada
        // que esperar: hay que decodificarlo igualmente. Solo esperamos si lo que
        // empuja por encima del techo es la SUMA con lo que ya hay en vuelo —es
        // decir, hay otro worker al que dejar terminar primero.
        if bytes < self.max_bytes && self.in_flight.load(Ordering::Acquire) + bytes > self.max_bytes
        {
            // `gate` serializa la espera: solo un worker hace busy-wait educado a
            // la vez y emite el aviso una sola vez por congestión.
            let _g = self.gate.lock().expect("mutex del presupuesto de PCM");
            let mut avisado = false;
            while self.in_flight.load(Ordering::Acquire) > 0
                && self.in_flight.load(Ordering::Acquire) + bytes > self.max_bytes
            {
                if !avisado {
                    tracing::warn!(
                        en_vuelo_mb = self.in_flight.load(Ordering::Acquire) / 1_000_000,
                        reserva_mb = bytes / 1_000_000,
                        techo_mb = self.max_bytes / 1_000_000,
                        "PCM acumulado cerca del techo: el worker espera para no agotar RAM"
                    );
                    avisado = true;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
        self.in_flight.fetch_add(bytes, Ordering::AcqRel);
        PcmReservation {
            budget: self,
            bytes,
        }
    }
}

/// Guard RAII: descuenta del PCM en vuelo cuando el worker termina con el audio.
struct PcmReservation<'a> {
    budget: &'a PcmBudget,
    bytes: u64,
}

impl Drop for PcmReservation<'_> {
    fn drop(&mut self) {
        self.budget
            .in_flight
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

struct DiarCfg {
    vad_model: PathBuf,
    emb_model: PathBuf,
    seg_model: PathBuf,
    max_speakers: Option<usize>,
}

/// Estado mutable de un worker: sus sesiones ONNX propias (no compartibles).
struct Worker {
    engine: Box<dyn rosetta_asr::Engine>,
    vad: SileroVad,
    denoiser: Option<GtcrnDenoiser>,
    diarizer: Option<Diarizer>,
}

/// Expande las entradas posicionales (archivos o directorios) y `--batch` a una
/// lista de archivos de medios, ordenada y sin duplicados.
pub fn gather_inputs(
    positional: &[PathBuf],
    batch_dir: Option<&Path>,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for p in positional {
        if p.is_dir() {
            collect_dir(p, &mut files)?;
        } else {
            files.push(p.clone()); // archivo nombrado explícitamente: sin filtrar por extensión
        }
    }
    if let Some(d) = batch_dir {
        collect_dir(d, &mut files)?;
    }
    files.sort();
    files.dedup();
    Ok(files)
}

fn collect_dir(dir: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    let rd = std::fs::read_dir(dir)
        .map_err(|e| anyhow::anyhow!("leer directorio {}: {e}", dir.display()))?;
    for entry in rd {
        let path = entry
            .map_err(|e| anyhow::anyhow!("entrada de {}: {e}", dir.display()))?
            .path();
        if path.is_file() && has_media_ext(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn has_media_ext(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| MEDIA_EXT.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Ejecuta el lote: asegura los modelos una vez, abre un pool de `jobs` workers
/// y procesa los archivos en paralelo. Reporta un resumen y devuelve error si
/// algún archivo falló (sin abortar el resto).
pub fn run(cli: &Cli, files: Vec<PathBuf>) -> anyhow::Result<()> {
    crate::ensure_ort_dylib();
    if cli.background {
        crate::apply_background_priority();
    }

    let hw = rosetta_accel::detect_hw();
    let device: Device = cli.device.into();
    rosetta_accel::ep::check_forced_device(&hw, device).map_err(|e| anyhow::anyhow!("{e}"))?;

    let jobs = resolve_jobs(&hw, device, cli.jobs, files.len(), &cli.model);
    // El cap de `--background` (mitad de núcleos) va sobre el total de hilos, no
    // sobre `jobs`: así limitamos la carga global de CPU sin cambiar cuántos
    // archivos corren en paralelo.
    let total_threads = crate::resolve_threads(cli.threads, cli.background);
    let per_engine_threads = (total_threads / jobs).max(1);

    // Asegurar TODOS los modelos una sola vez, en el hilo principal, para evitar
    // que varios workers intenten descargar lo mismo a la vez.
    let model_dir = rosetta_models::ensure_model(&cli.model)
        .map_err(|e| anyhow::anyhow!("preparar modelo ASR '{}': {e}", cli.model))?;
    // Modelos auxiliares de UN solo archivo: el nombre del `.onnx` lo resuelve el
    // catálogo vía `ensure_model_file` (sin hardcodear el nombre aquí).
    let vad_model = rosetta_models::ensure_model_file("silero-vad")
        .map_err(|e| anyhow::anyhow!("preparar VAD: {e}"))?;
    let denoise_model = if cli.denoise || cli.enhance_voice {
        Some(
            rosetta_models::ensure_model_file("gtcrn-simple")
                .map_err(|e| anyhow::anyhow!("preparar modelo de denoise: {e}"))?,
        )
    } else {
        None
    };
    let diar = if cli.diarize {
        let emb_model = rosetta_models::ensure_model_file("campplus-sv-zh-en")
            .map_err(|e| anyhow::anyhow!("preparar modelo de hablantes: {e}"))?;
        let seg_model = rosetta_models::ensure_model_file("pyannote-segmentation-3.0")
            .map_err(|e| anyhow::anyhow!("preparar segmentador pyannote: {e}"))?;
        Some(DiarCfg {
            vad_model: vad_model.clone(),
            emb_model,
            seg_model,
            max_speakers: cli.max_speakers.map(|n| n as usize),
        })
    } else {
        None
    };

    // Calcular el presupuesto antes de mover `hw` al contexto.
    let pcm_budget = PcmBudget::from_hw(&hw);
    let ctx = BatchCtx {
        hw,
        device,
        threads: per_engine_threads,
        model: cli.model.clone(),
        model_dir,
        language: cli.language.clone(),
        init_prompt: cli.init_prompt.clone(),
        denoise_model,
        vad_model,
        diar,
        format: cli.format.into(),
        timestamps: cli.timestamps,
        out_dir: cli.out_dir.clone(),
        force: cli.force,
        pcm_budget,
    };

    tracing::info!(
        archivos = files.len(),
        jobs,
        hilos_totales = total_threads,
        hilos_por_engine = per_engine_threads,
        background = cli.background,
        "modo lote"
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .map_err(|e| anyhow::anyhow!("crear pool de hilos: {e}"))?;

    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(PathBuf, std::result::Result<PathBuf, String>)>> =
        Mutex::new(Vec::with_capacity(files.len()));

    pool.install(|| {
        // Exactamente `jobs` workers; cada uno construye su motor una vez y drena
        // la cola compartida (índice atómico).
        (0..jobs).into_par_iter().for_each(|_| {
            // Aislar pánicos por worker/archivo: un pánico INESPERADO (p. ej. un
            // índice fuera de rango ante una salida de modelo malformada, o un
            // pánico que unwinde desde un EP nativo) NO debe abortar el lote ni
            // perder los resultados ya acumulados. rayon re-lanza el pánico de un
            // worker fuera de `pool.install`, así que lo atrapamos aquí y lo
            // convertimos en el error del archivo (los errores normales ya son
            // `Result`, no pánicos).
            let mut worker = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                make_worker(&ctx)
            })) {
                Ok(w) => w,
                Err(p) => Err(format!("pánico construyendo worker: {}", panic_message(p))),
            };
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= files.len() {
                    break;
                }
                let input = &files[i];
                let r = match &mut worker {
                    Ok(w) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        process_one(&ctx, w, input)
                    }))
                    .unwrap_or_else(|p| {
                        Err(format!("pánico procesando archivo: {}", panic_message(p)))
                    }),
                    Err(e) => Err(e.clone()),
                };
                results
                    .lock()
                    .expect("mutex de resultados")
                    .push((input.clone(), r));
            }
        });
    });

    let mut results = results.into_inner().expect("mutex de resultados");
    results.sort_by(|a, b| a.0.cmp(&b.0));

    let mut ok = 0usize;
    let mut failed = 0usize;
    for (input, r) in &results {
        match r {
            Ok(out) => {
                ok += 1;
                println!("OK    {} -> {}", input.display(), out.display());
            }
            Err(e) => {
                failed += 1;
                eprintln!("FALLO {}: {e}", input.display());
            }
        }
    }
    println!("Lote: {ok} ok, {failed} fallos de {} archivos", files.len());
    if failed > 0 {
        anyhow::bail!("{failed} de {} archivos fallaron", files.len());
    }
    Ok(())
}

/// RAM (GB) estimada por worker según el modelo ASR. Conservador: cada worker
/// carga su propio motor (sesión ONNX + buffers), así que el nº de archivos en
/// paralelo está acotado por la memoria disponible además de por los núcleos.
fn ram_por_worker(model: &str) -> f64 {
    // Whisper large-v3-turbo es bastante más pesado en RAM que Parakeet.
    if model.contains("whisper") { 1.5 } else { 1.0 }
}

/// Decide cuántos archivos procesar en paralelo. Con un acelerador único
/// (GPU/NPU) siempre 1; en CPU, hasta el nº de núcleos acotado por archivos y
/// por la RAM disponible (cada worker carga su propio motor).
fn resolve_jobs(
    hw: &HwProfile,
    device: Device,
    requested: usize,
    n_files: usize,
    model: &str,
) -> usize {
    let first = rosetta_accel::ep::cascade_kinds(hw, device)
        .first()
        .copied();
    let is_accel = !matches!(first, Some(EpKind::Cpu) | Some(EpKind::Xnnpack) | None);
    if is_accel {
        if requested > 1 {
            tracing::warn!(
                jobs = requested,
                ep = first.map(|k| k.label()).unwrap_or("?"),
                "acelerador único: se procesa 1 archivo a la vez (se ignora --jobs)"
            );
        }
        return 1;
    }
    let hw_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let base = if requested == 0 {
        hw_threads
    } else {
        requested
    };
    // Cap por RAM (solo en CPU; el acelerador ya devolvió 1 arriba): reserva el
    // 70% de la RAM física y reparte entre el coste estimado por worker.
    let max_by_ram = (hw.ram_gb * 0.7 / ram_por_worker(model)).floor().max(1.0) as usize;
    let jobs = base.clamp(1, n_files.max(1)).min(max_by_ram);
    if jobs < base.clamp(1, n_files.max(1)) {
        tracing::warn!(
            ram_gb = hw.ram_gb,
            max_por_ram = max_by_ram,
            "jobs recortado por RAM disponible"
        );
    }
    jobs
}

/// Construye el estado de un worker (motor + VAD + denoiser/diarizador opcionales).
/// Los modelos ya están en disco (asegurados en `run`), así que no se descarga aquí.
fn make_worker(ctx: &BatchCtx) -> std::result::Result<Worker, String> {
    let engine =
        rosetta_asr::build_engine(&ctx.model, &ctx.model_dir, &ctx.hw, ctx.device, ctx.threads)
            .map_err(|e| format!("cargar motor ASR: {e}"))?;
    let vad = SileroVad::from_file(&ctx.vad_model, &ctx.hw, ctx.threads)
        .map_err(|e| format!("cargar VAD: {e}"))?;
    let denoiser = ctx
        .denoise_model
        .as_ref()
        .map(|m| {
            GtcrnDenoiser::from_file(m, &ctx.hw, ctx.device, ctx.threads)
                .map_err(|e| format!("cargar denoiser: {e}"))
        })
        .transpose()?;
    let diarizer = match &ctx.diar {
        Some(d) => {
            let cfg = rosetta_diarize::DiarizeConfig {
                max_speakers: d.max_speakers,
                ..Default::default()
            };
            Some(
                Diarizer::new(
                    &d.vad_model,
                    &d.emb_model,
                    &d.seg_model,
                    &ctx.hw,
                    ctx.threads,
                    cfg,
                )
                .map_err(|e| format!("cargar diarizador: {e}"))?,
            )
        }
        None => None,
    };
    Ok(Worker {
        engine,
        vad,
        denoiser,
        diarizer,
    })
}

/// Procesa un archivo de principio a fin y escribe su salida. Devuelve la ruta
/// escrita o un mensaje de error (no propaga panics ni aborta el lote).
fn process_one(
    ctx: &BatchCtx,
    w: &mut Worker,
    input: &Path,
) -> std::result::Result<PathBuf, String> {
    let mut audio =
        rosetta_audio::load_audio_16k_mono(input).map_err(|e| format!("decodificar: {e}"))?;

    // seguridad-5b: contabiliza el PCM decodificado que este worker mantiene vivo
    // durante todo el procesamiento (decode→denoise→transcribe→diarize). Se reserva
    // ~3× el buffer base por los temporales del denoiser/pipeline (que también
    // alojan PCM). El guard descuenta al salir de la función. En el caso normal
    // (archivos cortos) no se espera; solo acota la congestión patológica.
    let pcm_bytes = (audio.samples.len() as u64) * 4 * 3;
    let _pcm = ctx.pcm_budget.reserve(pcm_bytes);

    if let Some(d) = &mut w.denoiser {
        audio = d.process(&audio).map_err(|e| format!("denoise: {e}"))?;
    }

    let vad = if audio.duration_s() > rosetta_pipeline::SINGLE_PASS_MAX_S {
        Some(&mut w.vad)
    } else {
        None
    };
    let (mut transcript, speech) = rosetta_pipeline::transcribe(
        w.engine.as_mut(),
        &audio,
        vad,
        input.display().to_string(),
        ctx.language.clone(),
        ctx.init_prompt.clone(),
    )
    .map_err(|e| format!("transcribir: {e}"))?;

    if let Some(diar) = &mut w.diarizer {
        // opt#3: reutiliza los segmentos del pipeline (audio largo); VAD propio en corto.
        let (turns, overlaps) = match &speech {
            Some(sp) => diar.diarize_with_segments(&audio, sp),
            None => diar.diarize(&audio),
        }
        .map_err(|e| format!("diarizar: {e}"))?;
        rosetta_diarize::segment_by_speaker(&mut transcript, &turns);
        rosetta_diarize::mark_overlaps(&mut transcript, &turns, &overlaps);
    }

    let rendered = rosetta_core::render(&transcript, ctx.format, ctx.timestamps);
    let out_path = out_path_for(ctx, input);
    if out_path.exists() && !ctx.force {
        return Err(format!("{} ya existe (usa --force)", out_path.display()));
    }
    std::fs::write(&out_path, rendered)
        .map_err(|e| format!("escribir {}: {e}", out_path.display()))?;
    Ok(out_path)
}

/// Ruta de salida en modo lote: `-d`/carpeta-del-input + nombre con la extensión
/// del formato (`-o`/`--output`, que apunta a un único archivo, no aplica aquí).
/// Reusa el helper compartido del CLI (única fuente de verdad con `output_path`).
fn out_path_for(ctx: &BatchCtx, input: &Path) -> PathBuf {
    crate::default_out_path(input, ctx.out_dir.as_deref(), ctx.format)
}

/// Extrae un mensaje legible del payload de un pánico atrapado con `catch_unwind`.
fn panic_message(p: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "pánico desconocido".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rosetta_accel::{Arch, Os, Vendor};

    /// `HwProfile` mínimo para tests, con la RAM que interese y sin NPU/GPU para
    /// que `--device cpu` caiga seguro en la rama CPU de `resolve_jobs`.
    fn hw_con_ram(ram_gb: f64) -> HwProfile {
        HwProfile {
            os: Os::Linux,
            arch: Arch::X64,
            cpu_brand: "test".to_string(),
            cpu_vendor: Vendor::Unknown,
            cpu_cores: 16,
            ram_gb,
            gpu_vendor: None,
            has_npu: false,
            npu_name: None,
        }
    }

    #[test]
    fn cap_por_ram_recorta_jobs_en_cpu() {
        // 4 GB * 0.7 = 2.8 GB; Parakeet ~1.0 GB/worker → máx 2 jobs aunque se
        // pidan 8 y haya 8 archivos.
        let hw = hw_con_ram(4.0);
        let jobs = resolve_jobs(&hw, Device::Cpu, 8, 8, "parakeet-tdt-0.6b-v3");
        assert_eq!(jobs, 2, "el cap por RAM debe limitar a 2 jobs");
    }

    #[test]
    fn cap_por_ram_modelo_pesado() {
        // 6 GB * 0.7 = 4.2 GB; Whisper ~1.5 GB/worker → máx 2 jobs.
        let hw = hw_con_ram(6.0);
        let jobs = resolve_jobs(&hw, Device::Cpu, 8, 8, "whisper-large-v3-turbo");
        assert_eq!(jobs, 2, "Whisper consume más RAM, así que se recorta antes");
    }

    #[test]
    fn cap_por_ram_minimo_un_job() {
        // RAM muy baja: nunca debe bajar de 1 job.
        let hw = hw_con_ram(0.5);
        let jobs = resolve_jobs(&hw, Device::Cpu, 8, 8, "parakeet-tdt-0.6b-v3");
        assert_eq!(jobs, 1, "el cap nunca debe dejar 0 jobs");
    }

    #[test]
    fn ram_abundante_no_recorta() {
        // 64 GB de RAM: el límite manda es n_files / núcleos, no la RAM.
        let hw = hw_con_ram(64.0);
        let jobs = resolve_jobs(&hw, Device::Cpu, 4, 4, "parakeet-tdt-0.6b-v3");
        assert_eq!(jobs, 4, "con RAM de sobra el cap no interviene");
    }

    #[test]
    fn pcm_budget_techo_minimo() {
        // RAM muy baja: el techo no baja del piso de 256 MB.
        let b = PcmBudget::from_hw(&hw_con_ram(0.5));
        assert_eq!(b.max_bytes, 256_000_000);
    }

    #[test]
    fn pcm_budget_reserva_y_libera() {
        // Reservar y soltar deja el contador a cero (sin esperas en el camino
        // normal: una sola reserva por debajo del techo nunca se bloquea).
        let b = PcmBudget::from_hw(&hw_con_ram(16.0)); // techo = 3.2 GB
        {
            let _r = b.reserve(100_000_000);
            assert_eq!(b.in_flight.load(Ordering::Acquire), 100_000_000);
        }
        assert_eq!(b.in_flight.load(Ordering::Acquire), 0);
    }

    #[test]
    fn pcm_budget_archivo_gigante_no_se_bloquea() {
        // Una reserva mayor que el techo, sin nada en vuelo, NO espera (no hay otro
        // worker al que ceder el paso): se admite y se contabiliza igualmente.
        let b = PcmBudget::from_hw(&hw_con_ram(2.0)); // techo = 256 MB (piso)
        let _r = b.reserve(1_000_000_000);
        assert_eq!(b.in_flight.load(Ordering::Acquire), 1_000_000_000);
    }
}
