# AGENTS.md — Rosetta

Guía operativa para agentes de IA que trabajen en este repositorio. Léela entera antes de
tocar código. Las **reglas duras** de abajo invalidan cualquier comportamiento por defecto.

---

## 1. Qué es Rosetta

CLI de Rust para **transcripción de audio/vídeo on-device**, multiplataforma
(Windows ARM64/x86, Linux, macOS). Prioriza ejecutar en acelerador (NPU → GPU → CPU) con
fallback automático. Workspace cargo, repo git (rama `main`, **sin remoto** todavía).

**Objetivo:** herramienta completa con ≥1 modelo en NPU + fallback, descarga de modelos
on-demand, salidas md/json/txt/srt/vtt. Repo pensado para distribuir **binarios
precompilados** (no se cuantiza en el cliente).

**Idioma de trabajo:** responder al usuario **en español** (incluye acentos y signos
correctos). Comentarios y mensajes de commit en español.

---

## 2. Reglas duras (invariantes — no negociables)

1. **NUNCA compilar C/C++ en el build de Windows.** La máquina de desarrollo es un
   **Copilot+ PC Snapdragon X (Qualcomm Oryon), Windows 11 ARM64** (`Snapdragon(R) X - X126100`),
   **sin entorno MSVC (`INCLUDE`/`LIB`) ni clang** configurado para compilar C. Cualquier
   dependencia o feature que arrastre compilación de C (p. ej. `ring`, `onig`, OpenBLAS,
   `download-binaries` de ort, auto-descarga de ffmpeg-sidecar) **rompe el build local**.
   - En **Linux/macOS** (CI o el Pi por SSH) compilar C **sí** es aceptable.
   - Antes de añadir cualquier dependencia, verifica que no compile C en Windows con
     `cargo xtask check-no-c` y `cargo tree --target aarch64-pc-windows-msvc`.
2. **Probar antes de declarar algo completo.** Nada se da por bueno sin validación: tests
   verdes **y** ejecución real en hardware cuando aplique (ver §7).
3. **Commit por bloque/hito.** Mensaje en español, claro y conciso. **PROHIBIDO incluir marcas
   de agua de IA / herramienta / harness** (p. ej. "Claude", "Anthropic", "Generated with…",
   trailers `Co-Authored-By` de un asistente, menciones a "ultracode" o al modelo) en mensajes de
   commit, tags, descripciones de PR ni en NINGÚN otro artefacto de Git. Atribución limpia: el autor
   es el del repositorio. (usar `git commit -F <archivo>` para evitar problemas de heredoc; borrar el
   espurio archivo `nul` si aparece antes de commitear).
4. **Usar versiones de librerías al día** salvo razón explícita documentada.
5. **Verificación de acelerador honesta:** "EP activo" = porcentaje real de nodos colocados
   (no solo que `register()` no falle). Con `--device npu|gpu` explícito, caer a CPU debe ser
   **error duro**, nunca fallback silencioso.
6. **Proyecto 100% Rust — sin Python ni otros lenguajes.** Todo el código, incluido el tooling
   de build/diagnóstico, vive en Rust. NO añadir scripts en Python, bash u otro lenguaje para
   lógica nueva. El guard sin-C es `cargo xtask check-no-c` (crate `xtask`, envuelve
   `cargo tree --target …`) y el parser de placement es `rosetta doctor --profile <dir>` (en
   `rosetta-cli`, con `serde_json`); ambos sustituyen a los antiguos scripts. No reintroducir
   lógica en otros lenguajes.
7. **Proyecto agnóstico de dispositivo.** Rosetta NO está atado a ningún equipo
   concreto: debe funcionar y optimizarse **al 100%, de forma
   nativa y limpia, en cualquier Windows/Windows-ARM, Linux/Linux-ARM y macOS**, con el acelerador
   disponible en cada host (NVIDIA/AMD discretas, iGPU, NPU o CPU). **Toda decisión de arquitectura
   que optimice para un dispositivo a costa de otro es un bug, no una feature.** Los "bancos de
   prueba" del §7 son solo medios de validación cruzada, NO el público objetivo ni el límite del
   diseño. La regla 1 (sin-C en Windows) NO contradice esto: `load-dynamic` permite EP de GPU/NPU
   sin compilar C, cargando la dylib aparte.

---

## 3. Decisiones de producto firmes

- **Motor por defecto: Parakeet TDT 0.6B v3** (`parakeet-tdt-0.6b-v3`). Rápido (RTF ~0.1–0.2),
  buen español, 25 idiomas. Corre en CPU y DirectML/GPU.
- **Máxima calidad opt-in: Whisper large-v3-turbo** vía `--model whisper-large-v3-turbo`
  (ONNX int8, 99 idiomas, más lento). El usuario **revirtió** un intento de hacer Whisper el
  default tras medir el coste de velocidad: **Parakeet es el default; Whisper es opt-in.**
- Runtime único de inferencia: crate **`ort` 2.0.0-rc.12** con `load-dynamic` + feature `api-24`.
- Diarización propia (no `speakrs`): pyannote-segmentation + embeddings CAM++ + clustering.
- VAD Silero v5 ONNX directo; denoise GTCRN ONNX; decode `symphonia` (+ ffmpeg sidecar sin
  auto-descarga); CLI `clap`.

---

## 4. Arquitectura (crates)

| Crate | Responsabilidad |
|-------|-----------------|
| `rosetta-cli` | binario `rosetta`: CLI, flags, render de salidas, modo lote (`batch.rs`) |
| `rosetta-core` | tipos (`Transcript`/`Segment`/`Word`/`Speaker`), errores, `render`, `RunMetrics` |
| `rosetta-audio` | decode→PCM 16k mono, resample, `vad.rs` (Silero), `denoise.rs` (GTCRN), `stft.rs` |
| `rosetta-asr` | motores ASR: `parakeet.rs`, `whisper/{mod,mel,tokenizer}.rs`; factoría `build_engine`, trait `Engine` |
| `rosetta-diarize` | diarización (pyannote + CAM++ + clustering `kodama`) + merge con ASR |
| `rosetta-accel` | detección de hardware + cascada de Execution Providers (`ep.rs`) |
| `rosetta-models` | catálogo `models.toml` embebido, descarga (ureq+native-tls/rustls), caché, sha256 |
| `rosetta-pipeline` | orquestador de audio largo (VAD → bloques → transcribe → reubicar timestamps) |

Edition **2024**, MSRV **1.88** (let-chains estables). El orquestador vive en
`rosetta-pipeline`, no en el engine (el engine no trocea).

---

## 5. Stack de inferencia y restricciones de `ort`

- **`ort` 2.0.0-rc.12** es la última release (confirmado jun-2026; no hay rc.13). **Multiversiona**
  onnxruntime 1.17–1.24 (feature `api-*`); `api-24` fija el techo **in-process en ≤1.24.x** (porque
  `ORT_API_VERSION` sube a 25 en onnxruntime 1.25). Usar **onnxruntime 1.24.4**. rc.12 **sí** expone
  la API de selección automática de EP: `SessionBuilder::with_auto_device(AutoDevicePolicy)` /
  `with_devices` / `env.devices()` (autoEP V2, requiere ORT≥1.22) — a evaluar en el spike NPU.
- `load-dynamic` ⇒ la dylib de onnxruntime se obtiene aparte (no se compila): wheel pip ARM64
  en `runtime/win-arm64-dml/onnxruntime.dll` (build DirectML, sirve CPU+GPU, lleva `DirectML.dll`
  al lado); en Linux/macOS el tarball oficial. **`load-dynamic` permite features de EP
  (directml/qnn/coreml/openvino) sin compilar C** — se gatean por target en `Cargo.toml`/`ep.rs`.
- **Gotcha:** sin `ORT_DYLIB_PATH` ort **cuelga** en vez de dar error. El CLI la resuelve sola
  (junto al exe / `runtime/<subdir>/`). La detección de HW es nativa (CPUID/OS/`/sys/class/drm`);
  `env.devices()` **sí** existe en rc.12 (device.rs:86) pero hoy no se usa para la cascada.
- API rc.12: `Session::builder()?.with_optimization_level()?.with_intra_threads()?.commit_from_file()`;
  inputs `ort::inputs!(...)`; salida `outputs["n"].try_extract_tensor::<T>()? -> (shape, &[data])`.

---

## 6. Flujo de verificación (por bloque, antes de commitear)

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo xtask check-no-c                      # guard sin-C por-target (Rust puro)
cargo tree --target aarch64-pc-windows-msvc # confirmar que no entró C nuevo en Windows
```
Todo verde **+ validación en hardware real** (§7) antes del commit. `cargo-deny` NO compila en
la dev ARM64 (necesita C) → solo corre en CI.

---

## 7. Bancos de prueba (validación punto-a-punto)

> **Recordatorio (regla dura §2.7):** estos bancos son **medios de validación cruzada, NO el
> público objetivo**. Sirven para cazar caídas silenciosas a CPU y divergencias int8 entre hosts;
> no para sesgar decisiones hacia un equipo. Una estación con GPU discreta NVIDIA/AMD —el caso más
> común fuera del dev— es tan ciudadano de primera como el Snapdragon.

Recoger telemetría cruzada entre hosts con `--trace` (JSONL `RunMetrics`, esquema fijo) +
instrumentación por etapa. Diff de los JSONL entre hosts caza caídas silenciosas a CPU y
divergencias int8 GPU/CPU.

- **Windows ARM64 (clase Snapdragon X)** — CPU + DirectML(GPU). Exportar `ROSETTA_MODELS_DIR`
  al directorio de modelos y apuntar a la dll DirectML (`runtime/win-arm64-dml/`).
- **Raspberry Pi 5 / Linux aarch64 (por SSH)** — CPU-only. Ejecutar con
  `ORT_DYLIB_PATH=…/runtime/linux-aarch64/libonnxruntime.so` (1.24.4) y `ROSETTA_MODELS_DIR=…`.
  Útil para cazar divergencias int8 entre hosts. (symphonia cubre WAV sin ffmpeg).
- **Docker x86 Linux CPU** — el mismo código Linux ya lo cubre el CI.

Fixtures: `tests/fixtures/{es_prueba,es_largo,es_2voces}.wav` (gitignored; los 2 últimos
sintéticos). Generadores y bancos de prueba son medios de validación, no parte del repo público.

### Patrones HW medidos (clave)
- **DirectML es 2–10× MÁS LENTO que CPU para int8 pequeño en Snapdragon** (Parakeet es_prueba
  CPU 296 ms vs DML 2876 ms; es_largo CPU 3084 vs DML 8768) y usa ~2× RAM (~730 MB vs
  1090–1560). Es un régimen específico (int8 pequeño + iGPU de memoria compartida sin unidades
  int8). **Decisión: NO se hardcodea preferir CPU** — rompería equipos con GPU potente (NVIDIA
  discreta gana). La cascada `NPU→GPU→CPU` se mantiene; en Snapdragon usar `--device cpu`
  (velocidad) o `--threads N` (no saturar el sistema).
- **CPU rápido pero satura el sistema; DirectML(GPU) más lento pero deja la CPU libre** (sistema
  fluido). Es un tradeoff, no "CPU gana siempre".
- Pi 5: Parakeet es_largo RTF 0.20 (5× tiempo real); **Whisper impráctico en Pi** (22 s para un
  clip de 4 s) → en CPU-only usar Parakeet.

---

## 8. Bloqueos conocidos (NPU / acelerador)

- **QNN (NPU Hexagon) BLOQUEADO:** crear cualquier sesión QNN → **stack overflow** en
  `commit_from_file` (recursión sin límite; ni 512 MB de stack bastan), con
  `QNN is_available=Ok(true)`. Probado con modelos de 0.13 MB a 1.46 GB y con el `.bin`
  precompilado → es un **bug de la pila `ort` rc.12 + onnxruntime-QNN 1.24.4 en Windows ARM64**
  (cf. onnxruntime #24082/#24166), no del modelo. No hay rc.13 que lo arregle todavía.
  **De-risk 2026-06-30 (spike Windows ML — fontanería RESUELTA, crash AISLADO al QNN EP):**
  (1) `ort` rc.12 **SÍ envuelve** la API plugin-EP V2: `Environment::register_ep_library(name, path)`
  (feature `api-22`, activa vía `api-24`) → **cero FFI crudo necesario**. (2) Registrando el propio
  `onnxruntime_providers_qnn.dll` (1.24 marzo-2026), `env.devices()` **expone la NPU Hexagon como
  `EpDevice` (QNNExecutionProvider, type=NPU)** — la ruta moderna VE la NPU. (3) PERO crear la sesión
  (`with_auto_device(PreferNPU)` / `with_devices`, ruta V2 distinta del `AppendExecutionProvider_QNN`
  heredado) **sigue dando STATUS_STACK_OVERFLOW** en `commit_from_file` → el bug está en la
  **compilación de grafo del QNN EP** (QAIRT **2.42**, onnxruntime 1.24), no en la ruta de registro.
  Palanca restante: registrar un **QNN EP más nuevo (QAIRT 2.45**, el del catálogo de Windows ML,
  GA ~jul-2026) o esperar onnxruntime 1.25+ con un ort api-25. (4) **Recursión LOCALIZADA**
  (`tests/qnn_crash_stack.rs`, vectored exception handler): los 234 frames repetidos del overflow
  están **enteramente en `QnnHtp.dll`**, NO en `onnxruntime*`. **PROBADO EXHAUSTIVAMENTE el swap a
  QAIRT 2.47 (+ `htp_arch=73`/`soc_model=60` explícitos vía `tests/qnn_crash_stack.rs::qnn_with_arch_hint`):
  recursa IDÉNTICO** (~236 frames en `QnnHtp.dll`, leaf en `QcSoCServiceUtils.dll` del SISTEMA —
  v1.0.0.6145, driver `qcsocservicekmdf8380`). ⇒ NO es bug de versión de QAIRT ni de opciones del EP,
  sino de la interacción de `QnnHtp` con el **servicio del SoC a nivel de plataforma** (ambos closed-source,
  no depurables). Único camino que lo resuelve: el **QNN EP del catálogo de Windows ML** (GA ~jul-2026,
  integración sancionada por Microsoft) o onnxruntime 1.25+ con un ort api-25. (5) **Integración opt-in
  YA LISTA** en
  `build_session`: `ROSETTA_ENABLE_QNN_EP=1` (+ `ROSETTA_QNN_EP_LIB` o el provider junto a
  `ORT_DYLIB_PATH`) registra el QNN como plugin-EP y crea la sesión con `with_auto_device(PreferNPU)`;
  fuera de la cascada por defecto (con QAIRT 2.42 crashea). Probes: `tests/{npu_autoep_probe,
  qnn_crash_stack,whisper_qnn_smoke}.rs`.
- **DirectML + decoder int8 de Whisper CUELGA** la GPU (`887A0007` "GPU device hung"; el
  `If`/`use_cache_branch` sobre int8 es inestable en DML). Devuelve un `Err` capturable. En
  Snapdragon, **la única vía estable de Whisper es CPU**. `WhisperEngine` en `auto` con DirectML
  planeado **degrada a CPU en runtime** al fallar la inferencia y escribe un marcador local
  `.whisper-dml-decoder-broken` para no repetir el cuelgue; con `--device gpu` explícito se
  respeta (y probablemente cuelga). En otras GPUs Windows, Whisper sí usaría DirectML.
- **DirectML-NPU NO alcanza la Hexagon en Snapdragon (PROBADO y DESCARTADO, 2026-06-30):**
  `ep::DirectML` con `DeviceFilter::Npu` registra "DirectML (NPU)" e `is_available()=true`, pero
  coloca el **0 % de los nodos** en el acelerador — **100 % `CPUExecutionProvider`** (encoder
  2380 nodos / decoder_joint 528 / nemo128 35, Parakeet int8). Es fallback silencioso a CPU
  (cf. DirectML #640/#659). Fuera de la cascada Qualcomm. **La NPU real solo por Windows ML/QNN.**
- **Verificación HONESTA de placement:** define `ROSETTA_ORT_PROFILE=<prefijo>` y `build_session`
  activa `EnableProfiling` → ORT escribe un Chrome-trace JSON por sesión con el EP REAL por nodo.
  Parsear con `rosetta doctor --profile <dir>` (cuenta nodos/tiempo por provider). **Regla dura: nunca
  declarar "corre en NPU/GPU" por la etiqueta `ep_primary` ni por el tiempo — solo por el % de
  nodos colocados.**

---

## 9. Contratos I/O de modelos (verificados inspeccionando los .onnx)

- **Whisper large-v3-turbo** (onnx-community, MIT, int8):
  - encoder: `input_features` f32 `[1,128,3000]` → `last_hidden_state` f32 `[1,1500,1280]`.
  - decoder (`decoder_model_merged`): `input_ids` i64, `encoder_hidden_states` f32,
    `use_cache_branch` **bool[1]**, `past_key_values.{0..3}.{decoder,encoder}.{key,value}`
    f32 `[1,20,L,64]` → `logits` `[1,seq,51866]` + `present.*`.
  - KV-cache 2 ramas: paso 1 `use_cache_branch=false` con KV dummy seq=0 (decoder **y**
    encoder); pasos ≥2 `use_cache_branch=true`, reutilizan `present.N.encoder.*` (constante 1500)
    y crecen `present.N.decoder.*`. Detección idioma: forward `[SOT=50258]` + argmax sobre
    50259–50358. Prompt: `[50258, lang, 50360(transcribe), 50364(no_ts)]`. EOS 50257.
  - **Mel frontend** (`whisper/mel.rs`, Rust puro): n_fft=400, hop=160, Hann periódica,
    reflect-pad 200, banco mel **Slaney** (htk=false, fmin0/fmax8000), log10, clamp a
    **máx global −8**, `(x+4)/4`, recorte a `[128,3000]`. Réplica exacta de HF (un error
    numérico aquí dispara el WER).
  - **Tokenizer** (`whisper/tokenizer.rs`, Rust puro, SIN crate `tokenizers`): byte-level GPT-2
    desde `vocab.json`; acumular bytes de todos los tokens y decodificar UTF-8 **al final**.
- **pyannote-segmentation-3.0** (`model.onnx`): input `x` f32 `[N,1,T]` (**waveform crudo** 16k,
  NO fbank), output `y` f32 `[N,frames,7]` (**powerset**; `frames` derivado de T, no hardcodear).
  **Solapes = clases {4,5,6}** (0=silencio, 1/2/3=spk solo, 4/5/6=pares).
- **CAM++**: `x[N,T,80]` → `emb[N,192]`. **Silero VAD**: procesa contexto(64)+ventana(512)=**576**
  (sin el contexto la prob sale ~0). **GTCRN**: n_fft=512/hop=256, ventana hann-sqrt, 3 caches a cero.

---

## 10. Estado actual y trabajo pendiente

**Hecho (F0–F8 + Fase 0 + bloques de auditoría + F + D + E1/E2-contrato):** decode audio/vídeo,
Parakeet CPU+DirectML, render, cascada EP + `doctor`, VAD + audio largo + word-timestamps,
diarización, denoise GTCRN, CI Linux/macOS, telemetría, motor **Whisper** completo (validado en
Snapdragon CPU), modo lote, quick wins de arranque (skip re-hash SHA), auto-nº-hablantes +
coalesce, contrato I/O de pyannote verificado.

**Hecho también (2026-06-30):** **opt#2** (Whisper enc+cross-KV por referencia, −11/−15 % t_transcribe)
+ **opt#5** (idioma plegado en el prefill); **E2-impl** (`PyannoteSegmenter` cableado en `Diarizer` →
`mark_overlaps` marca `overlap`/`speakers` en `Segment`, conservador); **opt#3** (VAD compartido
troceo↔diarización vía `diarize_with_segments`); **opt#4** (caché de grafo **opt-in** por
`ROSETTA_GRAPH_CACHE`, cold-start −25-33 %, duplica disco del modelo); **opt#7** (cap RAM×jobs en lote);
**`--background`** (cap hilos a núcleos/2 + prioridad baja; **spinning-off descartado** por dato);
**E4** (`Engine::transcribe_ctx` + `DecodeCtx`; **encoder BPE de Whisper** en `tokenizer.rs` con
`merges.txt` en el catálogo + `fancy-regex` sin C; flag `--init-prompt`; prefill inyecta
`[<|startofprev|>(50362), ...prompt..., SOT]`).

**Pendiente / aplazado:**
- **F9 (empaquetado): APLAZADO** hasta que el repo tenga **remoto de GitHub** (el release se valida con
  tags en Actions; sin remoto no es validable end-to-end). Por hacer: `[workspace.metadata.dist]`
  (6 targets, installers shell/powershell/npm/homebrew) + dylibs ORT vía `include` (con SHA del tarball)
  + `release.yml` en tag `v*` + shim npm para sidecar dylibs.
- **NPU: APLAZADA.** La fontanería plugin-EP está lista (`ROSETTA_ENABLE_QNN_EP`, ver §8) pero bloqueada
  por el bug `QnnHtp.dll`↔driver del SoC (no arreglado por QAIRT 2.47). Retomar con el QNN EP del
  **catálogo de Windows ML** (~jul-2026) u **onnxruntime 1.25+** con un `ort` api-25.
- **opt#6** (ventana fija 30 s): NO-FIX en cliente. **opt#1** (preferir CPU): descartada (portabilidad).
  **E3** (solape+dedup-LCS): DIFERIDO — `chunk_blocks` corta en silencio (bloques disjuntos), solo el
  monólogo continuo >28 s se beneficiaría → bajo valor.

**Docs del proyecto:** `README.md` / `README.es.md` (público) y este `AGENTS.md`. Los documentos
internos de proceso (auditorías, planes de fase) no se versionan en el repo público.

---

## 11. Modelos y caché

Catálogo embebido en `crates/rosetta-models/models.toml` (cada archivo con `sha256`). Descarga
on-demand a `ROSETTA_MODELS_DIR` o `models/`. Comandos: `rosetta models pull|list|verify|rm|path`.
Marcador `.verified-<name>` salta el re-hash si mtime+size coinciden (arranque ~10× más rápido).
Modelos en catálogo: parakeet-tdt-0.6b-v3 (int8), whisper-large-v3-turbo (int8),
silero-vad, pyannote-segmentation-3.0, CAM++, gtcrn. **No** confundir whisper-large-v3-turbo con
`whisper-large-v3-turbo-qnn` (export qai_hub `.bin` para X-Elite, bloqueado).
