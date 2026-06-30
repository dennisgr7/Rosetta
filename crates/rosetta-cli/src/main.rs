//! `rosetta`: CLI de transcripción de audio/vídeo on-device.
//!
//! Parsea los flags (clap), resuelve la dylib de ONNX Runtime, detecta el hardware
//! y orquesta el pipeline (decode → VAD/troceo → ASR → diarización → render) o los
//! subcomandos `info`/`doctor`/`models`.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

mod batch;
mod profile;

#[derive(Parser, Debug)]
#[command(
    name = "rosetta",
    version,
    about = "Transcripción de audio/vídeo on-device (NPU → GPU → CPU)",
    args_conflicts_with_subcommands = true
)]
struct Cli {
    /// Archivos de entrada (audio o vídeo, cualquier formato). Acepta varios y
    /// expande directorios a los medios que contengan (modo lote).
    #[arg(value_name = "INPUT")]
    input: Vec<PathBuf>,

    /// Formato de salida.
    #[arg(short = 'f', long, value_enum, default_value_t = FormatArg::Md)]
    format: FormatArg,

    /// Ruta del archivo de salida (por defecto: junto al input).
    #[arg(short = 'o', long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Directorio de salida (si no se da -o).
    #[arg(short = 'd', long, value_name = "DIR")]
    out_dir: Option<PathBuf>,

    /// Escribir a stdout en vez de a archivo.
    #[arg(long)]
    stdout: bool,

    /// Sobrescribir el archivo de salida si existe.
    #[arg(long)]
    force: bool,

    /// Procesa todos los medios de un directorio (modo lote).
    #[arg(long, value_name = "DIR")]
    batch: Option<PathBuf>,

    /// Archivos en paralelo en modo lote (0 = automático; se fuerza 1 en GPU/NPU).
    #[arg(short = 'j', long, default_value_t = 0)]
    jobs: usize,

    /// Reducción de ruido.
    #[arg(long)]
    denoise: bool,

    /// Realce / aislamiento de voz.
    #[arg(long, visible_alias = "vocal")]
    enhance_voice: bool,

    /// Diarización (separar hablantes).
    #[arg(long)]
    diarize: bool,

    /// Nº máximo de hablantes esperados (pista para la diarización; mínimo 1).
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
    max_speakers: Option<u32>,

    /// Modelo ASR. Por defecto Parakeet TDT 0.6B v3 (rápido, corre en GPU/DirectML,
    /// 25 idiomas). Usa `whisper-large-v3-turbo` para máxima calidad y 99 idiomas
    /// (en Snapdragon corre en CPU: DirectML cuelga su decoder int8).
    #[arg(short = 'm', long, default_value = "parakeet-tdt-0.6b-v3")]
    model: String,

    /// Idioma (código ISO-639 o "auto").
    #[arg(short = 'l', long, default_value = "auto")]
    language: String,

    /// Texto que sesga la transcripción (términos propios, nombres, formato). Solo
    /// afecta a Whisper (E4); Parakeet lo ignora.
    #[arg(long, default_value = "")]
    init_prompt: String,

    /// Dispositivo / acelerador.
    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    /// Nº de hilos de CPU (0 = automático).
    #[arg(short = 't', long, default_value_t = 0)]
    threads: usize,

    /// Preset "buen ciudadano": baja la prioridad del proceso y, si --threads no
    /// se da, limita los hilos a la mitad de los núcleos para no saturar el
    /// sistema. Un --threads explícito gana sobre el cap automático.
    #[arg(long, visible_alias = "nice")]
    background: bool,

    /// Incluir marcas de tiempo en la salida.
    #[arg(long)]
    timestamps: bool,

    /// Emite métricas de la corrida (1 línea JSONL) para comparar hardware:
    /// latencias por etapa, RTF, EP, RSS. A stderr salvo que se dé --trace-out.
    #[arg(long)]
    trace: bool,

    /// Archivo al que anexar (append) las métricas de --trace.
    #[arg(long, value_name = "FILE", requires = "trace")]
    trace_out: Option<PathBuf>,

    /// Verbosidad: -v info, -vv debug, -vvv trace.
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Silenciar todo salvo errores.
    #[arg(short, long, conflicts_with = "verbose")]
    quiet: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Muestra el hardware detectado y la cascada de aceleración.
    Info,
    /// Diagnostica la disponibilidad real de Execution Providers (NPU/GPU/CPU).
    Doctor {
        #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
        device: DeviceArg,

        /// Parsea los Chrome-trace `prof_*.json` de un directorio (los que emite
        /// `ROSETTA_ORT_PROFILE`) y reporta el placement REAL: % de nodos y de
        /// tiempo por Execution Provider, por sesión. Verificación honesta del
        /// "EP activo" (no la etiqueta, sino los nodos colocados de verdad).
        #[arg(long, value_name = "DIR")]
        profile: Option<PathBuf>,
    },
    /// Gestiona los modelos en caché.
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },
}

#[derive(Subcommand, Debug)]
enum ModelsAction {
    /// Lista el catálogo y el estado de cada modelo.
    List,
    /// Descarga un modelo a la caché.
    Pull { id: String },
    /// Elimina un modelo de la caché.
    Rm { id: String },
    /// Muestra la ruta de la caché de modelos.
    Path,
    /// Verifica los modelos descargados (SHA-256).
    Verify,
    /// Borra TODA la caché de modelos (ejecútalo antes de desinstalar para no
    /// dejar la caché huérfana). Pide confirmación salvo que se pase `--yes`.
    Clean {
        /// No pedir confirmación (para scripts).
        #[arg(long)]
        yes: bool,
    },
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum FormatArg {
    Md,
    Json,
    Txt,
    Srt,
    Vtt,
}

impl From<FormatArg> for rosetta_core::OutputFormat {
    fn from(f: FormatArg) -> Self {
        match f {
            FormatArg::Md => rosetta_core::OutputFormat::Md,
            FormatArg::Json => rosetta_core::OutputFormat::Json,
            FormatArg::Txt => rosetta_core::OutputFormat::Txt,
            FormatArg::Srt => rosetta_core::OutputFormat::Srt,
            FormatArg::Vtt => rosetta_core::OutputFormat::Vtt,
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum DeviceArg {
    Auto,
    Npu,
    Gpu,
    Cpu,
}

impl From<DeviceArg> for rosetta_accel::Device {
    fn from(d: DeviceArg) -> Self {
        match d {
            DeviceArg::Auto => rosetta_accel::Device::Auto,
            DeviceArg::Npu => rosetta_accel::Device::Npu,
            DeviceArg::Gpu => rosetta_accel::Device::Gpu,
            DeviceArg::Cpu => rosetta_accel::Device::Cpu,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose, cli.quiet);

    match &cli.command {
        Some(Command::Info) => cmd_info(),
        Some(Command::Doctor { device, profile }) => match profile {
            Some(dir) => profile::cmd_profile(dir),
            None => cmd_doctor(*device),
        },
        Some(Command::Models { action }) => cmd_models(action),
        None => {
            let files = batch::gather_inputs(&cli.input, cli.batch.as_deref())?;
            let multi = files.len() > 1 || cli.batch.is_some();
            if multi && (cli.stdout || cli.output.is_some()) {
                anyhow::bail!(
                    "--stdout y -o/--output solo valen para un único archivo; usa -d/--out-dir en modo lote"
                );
            }
            match files.len() {
                0 => {
                    if cli.batch.is_some() {
                        anyhow::bail!("no se encontraron medios en el directorio de --batch");
                    }
                    use clap::CommandFactory;
                    Cli::command().print_help()?;
                    println!();
                    Ok(())
                }
                1 if cli.batch.is_none() => {
                    cmd_transcribe(&cli, files.into_iter().next().expect("un archivo"))
                }
                _ => batch::run(&cli, files),
            }
        }
    }
}

/// Resuelve el nº de hilos de CPU a usar según `--threads` y `--background`.
///
/// - `threads > 0`: el valor explícito gana siempre (incluso con `--background`).
/// - `threads == 0 && background`: la mitad de los núcleos (mínimo 1) para dejar
///   el sistema fluido.
/// - `threads == 0 && !background`: todos los núcleos lógicos (el comportamiento
///   por defecto histórico).
pub(crate) fn resolve_threads(threads: usize, background: bool) -> usize {
    if threads > 0 {
        return threads;
    }
    let n = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    if background { (n / 2).max(1) } else { n }
}

/// Baja la prioridad del proceso para el preset `--background`. Se llama UNA vez
/// al inicio del comando. Si falla, solo avisa (no aborta): el preset es
/// best-effort, perder la prioridad reducida no debe impedir la transcripción.
pub(crate) fn apply_background_priority() {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Threading::{
            GetCurrentProcess, PROCESS_MODE_BACKGROUND_BEGIN, SetPriorityClass,
        };
        // SAFETY: GetCurrentProcess devuelve un pseudo-handle válido al proceso
        // actual; SetPriorityClass solo lee ese handle y ajusta la clase de
        // prioridad. No se comparte estado entre hilos.
        let ok = unsafe { SetPriorityClass(GetCurrentProcess(), PROCESS_MODE_BACKGROUND_BEGIN) };
        if ok == 0 {
            tracing::warn!("no se pudo bajar la prioridad del proceso (--background)");
        }
    }
    #[cfg(unix)]
    {
        // SAFETY: setpriority con PRIO_PROCESS y who=0 (el proceso actual) es una
        // llamada FFI segura; no toca memoria de Rust.
        let rc = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, 10) };
        if rc != 0 {
            tracing::warn!("no se pudo bajar la prioridad del proceso (--background)");
        }
    }
    #[cfg(not(any(windows, unix)))]
    {
        // Plataforma sin API conocida de prioridad: no-op.
    }
}

fn init_tracing(verbose: u8, quiet: bool) {
    use tracing_subscriber::{EnvFilter, fmt};

    let level = if quiet {
        "error"
    } else {
        match verbose {
            0 => "warn",
            1 => "info",
            2 => "debug",
            _ => "trace",
        }
    };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(format!(
            "rosetta={level},rosetta_cli={level},rosetta_accel={level},rosetta_asr={level},rosetta_audio={level},rosetta_diarize={level},rosetta_models={level},rosetta_pipeline={level}"
        ))
    });
    let _ = fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time()
        .try_init();
}

fn cmd_info() -> anyhow::Result<()> {
    let hw = rosetta_accel::detect_hw();

    println!("Rosetta {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Sistema:");
    println!("  SO / arch   : {:?} / {:?}", hw.os, hw.arch);
    println!(
        "  CPU         : {} ({} hilos lógicos)",
        hw.cpu_brand, hw.cpu_cores
    );
    println!("  Vendor CPU  : {:?}", hw.cpu_vendor);
    match hw.gpu_vendor {
        Some(v) => println!("  GPU         : {v:?}"),
        None => println!("  GPU         : no detectada"),
    }
    println!("  RAM         : {:.1} GB", hw.ram_gb);
    match &hw.npu_name {
        Some(n) => println!("  NPU         : {n}  (detección heurística)"),
        None => println!("  NPU         : no detectada"),
    }
    println!();
    println!("Cascada de aceleración (--device auto):");
    for (i, kind) in rosetta_accel::ep::cascade_kinds(&hw, rosetta_accel::Device::Auto)
        .iter()
        .enumerate()
    {
        println!("  {}. {}", i + 1, kind.label());
    }
    println!();
    println!(
        "Sugerencia: `rosetta doctor` mide qué Execution Provider ejecuta de verdad un modelo."
    );
    Ok(())
}

fn cmd_doctor(device: DeviceArg) -> anyhow::Result<()> {
    ensure_ort_dylib();
    let hw = rosetta_accel::detect_hw();
    let want: rosetta_accel::Device = device.into();

    println!("Rosetta doctor — dispositivo solicitado: {device:?}");
    println!("CPU: {} | NPU detectada: {}", hw.cpu_brand, hw.has_npu);
    println!();
    println!("Cascada y disponibilidad de Execution Providers:");
    for (kind, avail) in rosetta_accel::ep::cascade_availability(&hw, want) {
        let mark = if avail {
            "OK  disponible"
        } else {
            "--  no disponible"
        };
        println!("  {:<30} {mark}", kind.label());
    }
    println!();
    println!("Nota: 'disponible' = el runtime carga el EP. La verificación honesta de");
    println!("placement (% de nodos por EP) YA existe hoy: define ROSETTA_ORT_PROFILE=<prefijo>");
    println!(
        "y ORT emite un Chrome-trace por sesión; parséalo con 'rosetta doctor --profile <dir>'"
    );
    println!("(reporta el % de nodos y de tiempo por EP, por sesión).");
    Ok(())
}

fn cmd_models(action: &ModelsAction) -> anyhow::Result<()> {
    use rosetta_models as models;
    match action {
        ModelsAction::List => {
            println!(
                "Catálogo de modelos (caché: {})",
                models::cache_root().display()
            );
            for m in models::catalog() {
                let mark = if models::is_present(&m.id) {
                    "✓"
                } else {
                    "·"
                };
                println!("  {mark} {:<28} {:<10} {}", m.id, m.kind, m.license);
            }
        }
        ModelsAction::Pull { id } => {
            let dir = models::ensure_model(id)?;
            println!("Modelo '{id}' listo en {}", dir.display());
        }
        ModelsAction::Rm { id } => {
            models::remove(id)?;
            println!("Modelo '{id}' eliminado de la caché.");
        }
        ModelsAction::Path => println!("{}", models::cache_root().display()),
        ModelsAction::Verify => {
            let mut all_ok = true;
            for m in models::catalog() {
                if !models::is_present(&m.id) {
                    continue;
                }
                let ok = models::verify(&m.id)?;
                all_ok &= ok;
                println!("  {}  {}", if ok { "OK   " } else { "FALLO" }, m.id);
            }
            if !all_ok {
                anyhow::bail!("algún modelo descargado no supera la verificación SHA-256");
            }
        }
        ModelsAction::Clean { yes } => {
            let root = models::cache_root();
            if !root.exists() {
                println!(
                    "La caché de modelos no existe; nada que borrar ({}).",
                    root.display()
                );
                return Ok(());
            }
            if !*yes {
                use std::io::Write;
                print!(
                    "Se borrará TODA la caché de modelos en {} ¿Continuar? [y/N] ",
                    root.display()
                );
                std::io::stdout().flush().ok();
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                let ans = input.trim().to_ascii_lowercase();
                if ans != "y" && ans != "yes" {
                    println!("Cancelado.");
                    return Ok(());
                }
            }
            let (path, freed) = models::clean()?;
            println!(
                "Caché borrada: {} ({:.1} MB liberados).",
                path.display(),
                freed as f64 / 1_048_576.0
            );
        }
    }
    Ok(())
}

fn cmd_transcribe(cli: &Cli, input: PathBuf) -> anyhow::Result<()> {
    use std::time::Instant;
    ensure_ort_dylib();
    if cli.background {
        apply_background_priority();
    }
    let run_start = Instant::now();

    let threads = resolve_threads(cli.threads, cli.background);
    let hw = rosetta_accel::detect_hw();
    let device: rosetta_accel::Device = cli.device.into();
    rosetta_accel::ep::check_forced_device(&hw, device).map_err(|e| anyhow::anyhow!("{e}"))?;
    let casc = rosetta_accel::ep::cascade_kinds(&hw, device);
    tracing::info!(
        input = %input.display(),
        model = %cli.model,
        cascada = ?casc.iter().map(|k| k.label()).collect::<Vec<_>>(),
        background = cli.background,
        threads,
        "transcripción"
    );

    let t = Instant::now();
    let mut audio = rosetta_audio::load_audio_16k_mono(&input)
        .map_err(|e| anyhow::anyhow!("no se pudo decodificar {}: {e}", input.display()))?;
    let t_load_ms = t.elapsed().as_secs_f64() * 1000.0;
    let audio_s = audio.duration_s() as f64;
    tracing::info!(
        duracion_s = audio.duration_s(),
        sr = audio.sample_rate,
        "audio cargado"
    );

    // Resolución unificada: el modelo ASR se descarga/cachea y verifica igual que
    // los modelos auxiliares (una sola variable, ROSETTA_MODELS_DIR).
    let model_dir = rosetta_models::ensure_model(&cli.model)
        .map_err(|e| anyhow::anyhow!("preparar modelo ASR '{}': {e}", cli.model))?;

    // F7: realce/denoise de voz (mismo modelo SE para --denoise y --enhance-voice).
    let mut t_denoise_ms = 0.0;
    if cli.denoise || cli.enhance_voice {
        use rosetta_audio::Denoiser;
        let t = Instant::now();
        // arquitectura-4: un solo .onnx → resolución dedupe (id, no nombre de archivo).
        let model = rosetta_models::ensure_model_file("gtcrn-simple")
            .map_err(|e| anyhow::anyhow!("preparar modelo de denoise: {e}"))?;
        let mut denoiser = rosetta_audio::GtcrnDenoiser::from_file(&model, &hw, device, threads)
            .map_err(|e| anyhow::anyhow!("cargar denoiser GTCRN: {e}"))?;
        audio = denoiser
            .process(&audio)
            .map_err(|e| anyhow::anyhow!("aplicar denoise: {e}"))?;
        t_denoise_ms = t.elapsed().as_secs_f64() * 1000.0;
        tracing::info!("realce/denoise aplicado (GTCRN)");
    }

    let t = Instant::now();
    let mut engine = rosetta_asr::build_engine(&cli.model, &model_dir, &hw, device, threads)
        .map_err(|e| anyhow::anyhow!("cargar motor ASR: {e}"))?;
    let t_model_load_ms = t.elapsed().as_secs_f64() * 1000.0;

    // VAD para audio largo (se carga solo si supera el umbral de pasada única).
    let mut vad = if audio.duration_s() > rosetta_pipeline::SINGLE_PASS_MAX_S {
        // arquitectura-4: un solo .onnx → resolución dedupe (id, no nombre de archivo).
        let model = rosetta_models::ensure_model_file("silero-vad")
            .map_err(|e| anyhow::anyhow!("preparar modelo VAD: {e}"))?;
        Some(
            rosetta_audio::SileroVad::from_file(&model, &hw, threads)
                .map_err(|e| anyhow::anyhow!("cargar VAD Silero: {e}"))?,
        )
    } else {
        None
    };

    let t = Instant::now();
    let (mut transcript, speech) = rosetta_pipeline::transcribe(
        engine.as_mut(),
        &audio,
        vad.as_mut(),
        input.display().to_string(),
        cli.language.clone(),
        cli.init_prompt.clone(),
    )
    .map_err(|e| anyhow::anyhow!("transcribir: {e}"))?;
    let t_transcribe_ms = t.elapsed().as_secs_f64() * 1000.0;

    // F6: diarización de hablantes (re-segmenta la transcripción por turno).
    let mut t_diarize_ms = 0.0;
    if cli.diarize {
        let t = Instant::now();
        // arquitectura-4: tres .onnx de un solo archivo → resolución dedupe por id.
        let vad_model = rosetta_models::ensure_model_file("silero-vad")
            .map_err(|e| anyhow::anyhow!("preparar VAD: {e}"))?;
        let emb_model = rosetta_models::ensure_model_file("campplus-sv-zh-en")
            .map_err(|e| anyhow::anyhow!("preparar modelo de hablantes: {e}"))?;
        let seg_model = rosetta_models::ensure_model_file("pyannote-segmentation-3.0")
            .map_err(|e| anyhow::anyhow!("preparar segmentador pyannote: {e}"))?;
        let cfg = rosetta_diarize::DiarizeConfig {
            max_speakers: cli.max_speakers.map(|n| n as usize),
            ..Default::default()
        };
        let mut diar =
            rosetta_diarize::Diarizer::new(&vad_model, &emb_model, &seg_model, &hw, threads, cfg)
                .map_err(|e| anyhow::anyhow!("cargar diarizador: {e}"))?;
        // opt#3: reutiliza los segmentos de voz del pipeline (audio largo) para no
        // re-ejecutar el VAD; en audio corto la diarización corre su propio VAD.
        let (turns, overlaps) = match &speech {
            Some(sp) => diar.diarize_with_segments(&audio, sp),
            None => diar.diarize(&audio),
        }
        .map_err(|e| anyhow::anyhow!("diarizar: {e}"))?;
        rosetta_diarize::segment_by_speaker(&mut transcript, &turns);
        rosetta_diarize::mark_overlaps(&mut transcript, &turns, &overlaps);
        t_diarize_ms = t.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(
            turnos = turns.len(),
            solapes = overlaps.len(),
            "diarización aplicada"
        );
    }

    let t = Instant::now();
    let fmt: rosetta_core::OutputFormat = cli.format.into();
    let rendered = rosetta_core::render(&transcript, fmt, cli.timestamps);
    let t_render_ms = t.elapsed().as_secs_f64() * 1000.0;
    let n_segments = transcript.segments.len();

    if cli.stdout {
        print!("{rendered}");
        if !rendered.ends_with('\n') {
            println!();
        }
    } else {
        let out_path = output_path(cli, &input, fmt);
        if out_path.exists() && !cli.force {
            anyhow::bail!(
                "{} ya existe (usa --force para sobrescribir)",
                out_path.display()
            );
        }
        std::fs::write(&out_path, rendered)
            .map_err(|e| anyhow::anyhow!("escribir {}: {e}", out_path.display()))?;
        println!("Transcripción escrita en {}", out_path.display());
    }

    if cli.trace {
        let total_s = run_start.elapsed().as_secs_f64();
        let m = rosetta_core::RunMetrics {
            host: rosetta_accel::host_name(),
            os: format!("{:?}", hw.os),
            arch: format!("{:?}", hw.arch),
            device_arg: format!("{device:?}"),
            // EP REAL que usó el motor (el `device` de su sesión), no el planeado
            // por la cascada: así la telemetría no miente cuando el motor degrada
            // (p. ej. Whisper auto→CPU por el cuelgue de DirectML, o una caída
            // silenciosa GPU/NPU→CPU).
            ep_primary: transcript.model.device.clone(),
            model: cli.model.clone(),
            audio_s,
            rtf: if audio_s > 0.0 {
                total_s / audio_s
            } else {
                0.0
            },
            t_total_ms: total_s * 1000.0,
            t_load_ms,
            t_model_load_ms,
            t_denoise_ms,
            t_transcribe_ms,
            t_diarize_ms,
            t_render_ms,
            rss_mb: rosetta_accel::process_rss_mb(),
            n_segments,
        };
        emit_metrics(&m, cli.trace_out.as_deref());
    }
    Ok(())
}

/// Emite las métricas como una línea JSONL: anexa a `path` si se da, o a stderr.
fn emit_metrics(m: &rosetta_core::RunMetrics, path: Option<&std::path::Path>) {
    let line = m.to_jsonl();
    match path {
        Some(p) => {
            use std::io::Write;
            match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
            {
                Ok(mut f) => {
                    let _ = writeln!(f, "{line}");
                }
                Err(e) => eprintln!("métricas: no se pudo escribir {}: {e}", p.display()),
            }
        }
        None => eprintln!("{line}"),
    }
}

/// Ruta de salida por defecto (sin la rama `-o`): carpeta `-d` / la del input /
/// el cwd, más el nombre del input con la extensión del formato. Fuente única de
/// verdad compartida con el modo lote ([`batch::out_path_for`]).
pub(crate) fn default_out_path(
    input: &std::path::Path,
    out_dir: Option<&std::path::Path>,
    fmt: rosetta_core::OutputFormat,
) -> PathBuf {
    let ext = rosetta_core::extension(fmt);
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("salida");
    let dir = out_dir
        .map(|p| p.to_path_buf())
        .or_else(|| input.parent().map(|p| p.to_path_buf()))
        .unwrap_or_default();
    dir.join(format!("{stem}.{ext}"))
}

/// Determina la ruta del archivo de salida: `-o`, o [`default_out_path`].
fn output_path(cli: &Cli, input: &std::path::Path, fmt: rosetta_core::OutputFormat) -> PathBuf {
    if let Some(o) = &cli.output {
        return o.clone();
    }
    default_out_path(input, cli.out_dir.as_deref(), fmt)
}

/// Nombre de la librería de ONNX Runtime por plataforma.
#[cfg(windows)]
const ORT_DYLIB: &str = "onnxruntime.dll";
#[cfg(target_os = "linux")]
const ORT_DYLIB: &str = "libonnxruntime.so";
#[cfg(target_os = "macos")]
const ORT_DYLIB: &str = "libonnxruntime.dylib";

/// Si `ORT_DYLIB_PATH` no está definida, intenta localizar la librería de ONNX
/// Runtime junto al ejecutable o en `runtime/` del repo (desarrollo) y la fija.
///
/// # Modelo de confianza (seguridad-1)
///
/// La dylib de ONNX Runtime (y las DLLs/`.so` adyacentes que el loader resuelva
/// vía PATH/`LD_LIBRARY_PATH`/`DYLD_*`) **se cargan como código nativo SIN
/// verificación de firma ni sha-256** — al contrario que los modelos del
/// catálogo. Quien controle cualquiera de estas palancas:
///
/// - `ORT_DYLIB_PATH` (ruta explícita de la dylib),
/// - `ROSETTA_QNN_EP_LIB` / la carpeta donde se busca el provider QNN, o
/// - `ROSETTA_ENABLE_QNN_EP` (activa el registro plugin-EP del QNN),
///
/// puede lograr **ejecución de código nativo arbitrario** en el proceso. El
/// nivel de confianza es el mismo que el de `LD_PRELOAD` o de plantar una DLL en
/// el `PATH`: si un atacante puede fijar estas variables o escribir en las
/// carpetas que apuntan, ya tiene equivalente a ejecución local. Por eso NO se
/// resuelve la dylib desde el cwd (solo junto al exe / `runtime/` del repo), y
/// por eso cada vez que una de estas palancas entra en juego se emite un
/// `warn!` para que quede traza en el log.
fn ensure_ort_dylib() {
    // seguridad-1: avisar si el entorno fuerza el registro del QNN EP (carga el
    // provider QNN nativo, sin verificar; ver el modelo de confianza arriba).
    if std::env::var_os("ROSETTA_ENABLE_QNN_EP").is_some() {
        tracing::warn!(
            "ROSETTA_ENABLE_QNN_EP activo: se cargará el provider QNN nativo SIN verificación \
             de firma/sha (confianza equivalente a LD_PRELOAD/PATH). Solo en entorno de confianza."
        );
        if let Some(lib) = std::env::var_os("ROSETTA_QNN_EP_LIB") {
            tracing::warn!(
                qnn_ep_lib = %PathBuf::from(&lib).display(),
                "ROSETTA_QNN_EP_LIB apunta a una librería nativa que se cargará sin verificar."
            );
        }
    }

    if let Some(existing) = std::env::var_os("ORT_DYLIB_PATH") {
        // seguridad-1: el override carga una dylib nativa sin verificar (sha/firma).
        // Se respeta a propósito (es la vía soportada para una ruta a medida), pero
        // se deja traza: ver el modelo de confianza en el doc de esta función.
        tracing::warn!(
            ort_dylib_path = %PathBuf::from(&existing).display(),
            "ORT_DYLIB_PATH define la dylib de ONNX Runtime; se carga como código nativo SIN \
             verificación (confianza equivalente a LD_PRELOAD/PATH)."
        );
        // Respetar el override del usuario, pero asegurar que su carpeta esté en
        // el PATH (para DLLs adyacentes como DirectML.dll / QnnHtp.dll).
        if let Some(parent) = PathBuf::from(&existing).parent() {
            prepend_to_loader_path(parent);
        }
        return;
    }
    // Subcarpetas de `runtime/` por SO/acelerador, en orden de preferencia. En
    // Windows la build DirectML sirve también para CPU (habilita `--device
    // gpu/auto` sin romper CPU). En Linux/macOS los tarballs oficiales son
    // CPU-only; los subdirs con sufijo de EP (coreml/openvino) son para builds
    // específicos de F8b cuando estén disponibles.
    let subdirs: &[&str] = if cfg!(windows) {
        &["win-arm64-dml", "win-arm64-qnn", "win-x64-dml", ""]
    } else if cfg!(target_os = "linux") {
        &["linux-aarch64", "linux-x64", ""]
    } else if cfg!(target_os = "macos") {
        &["osx-arm64-coreml", "osx-arm64", ""]
    } else {
        &[""]
    };
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        roots.push(dir.to_path_buf()); // junto al exe (distribución)
        roots.push(dir.join("../../runtime")); // dev: target/<profile>
        roots.push(dir.join("../../../runtime"));
    }
    // Seguridad: NO añadimos `runtime/` relativo al cwd como candidato. La dylib
    // de ONNX Runtime no se verifica por sha (a diferencia de los modelos del
    // catálogo), así que resolverla desde el directorio de trabajo permitiría a un
    // atacante con escritura en él plantar un `onnxruntime.*` (y DLLs adyacentes)
    // y lograr ejecución de código nativo. Solo se busca junto al ejecutable o en
    // el `runtime/` del repo (relativo al exe); el override explícito
    // ORT_DYLIB_PATH sigue siendo la vía para una ruta a medida.
    for sub in subdirs {
        for root in &roots {
            let cand = root.join(sub).join(ORT_DYLIB);
            if cand.exists() {
                // SAFETY: se llama al inicio del manejo del comando, antes de
                // cualquier uso de ort y sin otros hilos tocando el entorno.
                unsafe {
                    std::env::set_var("ORT_DYLIB_PATH", &cand);
                }
                if let Some(parent) = cand.parent() {
                    prepend_to_loader_path(parent);
                }
                tracing::debug!(dylib = %cand.display(), "ORT_DYLIB_PATH resuelta");
                return;
            }
        }
    }
    tracing::warn!("no se encontró {ORT_DYLIB}; define ORT_DYLIB_PATH si la inferencia falla");
}

/// Antepone `dir` a la variable que usa el loader dinámico de cada SO para
/// encontrar las librerías adyacentes a la dylib de ONNX Runtime (Windows:
/// `DirectML.dll`/`QnnHtp*` vía `PATH`; Linux: `LD_LIBRARY_PATH`; macOS:
/// `DYLD_LIBRARY_PATH` + `DYLD_FALLBACK_LIBRARY_PATH`).
///
/// Nota macOS: SIP elimina las `DYLD_*` al lanzar subprocesos de binarios
/// protegidos; para una distribución (F9) lo robusto es un rpath
/// `@loader_path`/`$ORIGIN` en el binario, no la variable de entorno.
fn prepend_to_loader_path(dir: &std::path::Path) {
    // Variables del loader por SO. En Windows el loader usa `PATH`.
    #[cfg(windows)]
    let vars: &[&str] = &["PATH"];
    #[cfg(target_os = "linux")]
    let vars: &[&str] = &["LD_LIBRARY_PATH"];
    #[cfg(target_os = "macos")]
    let vars: &[&str] = &["DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"];
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    let vars: &[&str] = &["LD_LIBRARY_PATH"];

    for var in vars {
        let cur = std::env::var_os(var).unwrap_or_default();
        let mut paths = vec![dir.to_path_buf()];
        paths.extend(std::env::split_paths(&cur));
        // `join_paths` usa el separador correcto por plataforma (`;` Windows, `:` unix).
        if let Ok(joined) = std::env::join_paths(paths) {
            // SAFETY: al inicio del comando, sin otros hilos tocando el entorno.
            unsafe {
                std::env::set_var(var, joined);
            }
        }
    }
}
