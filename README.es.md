# Rosetta

**Transcripción de audio y vídeo on-device en Rust.** Rápida, privada y multiplataforma. Sin nube,
sin telemetría — tu audio nunca sale de tu máquina.

[![CI](https://github.com/dennisgr7/Rosetta/actions/workflows/ci.yml/badge.svg)](https://github.com/dennisgr7/Rosetta/actions/workflows/ci.yml)
&nbsp;·&nbsp; Licencia: MIT OR Apache-2.0 &nbsp;·&nbsp; Windows · Linux · macOS (x86-64 y ARM64)

🇬🇧 [Read me in English](README.md)

---

Rosetta elige el mejor acelerador disponible en cada equipo (**NPU → GPU → CPU**, con un fallback
automático y *honesto*) y ejecuta modelos ONNX localmente vía [ONNX Runtime](https://onnxruntime.ai/).
Los modelos se descargan bajo demanda y se cachean; el binario no lleva modelos dentro.

## Características

- **Motor por defecto — NVIDIA Parakeet TDT 0.6B v3**: rápido (RTF ~0.1–0.2), fuerte en español, 25 idiomas.
- **Máxima calidad (opt-in) — Whisper large-v3-turbo**: 99 idiomas, ONNX int8 (`--model whisper-large-v3-turbo`).
- **Diarización de hablantes** (`--diarize`), **reducción de ruido** (`--denoise`), marcas de tiempo por palabra.
- **Idioma forzado** (`--language es`) o autodetección, con metadato de idioma honesto.
- **Salidas**: Markdown, JSON, texto, **SRT**, **VTT** — a archivo o a stdout.
- **Modo lote** sobre un directorio, con paralelismo acotado.
- Cualquier formato de entrada (audio o vídeo): `symphonia` para audio común, `ffmpeg` para el resto.

## Instalación

Los instaladores precompilados se publican con cada release.

```sh
# Linux / macOS
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/dennisgr7/Rosetta/releases/latest/download/rosetta-installer.sh | sh

# Windows (PowerShell)
powershell -c "irm https://github.com/dennisgr7/Rosetta/releases/latest/download/rosetta-installer.ps1 | iex"

# npm        (cualquier SO con Node ≥ 14)
npm install -g rosetta-cli

# Homebrew   (macOS / Linux)
brew install dennisgr7/tap/rosetta

# Docker     (ver "Docker" abajo)
docker run --rm -v rosetta-models:/models -v "$PWD:/audio" ghcr.io/dennisgr7/rosetta /audio/clip.wav
```

O compila desde el código fuente (ver [Compilar](#compilar-desde-el-código-fuente)).

> En el **primer uso** se descargan, bajo demanda, el ONNX Runtime de tu plataforma y el modelo que
> uses (ambos verificados por sha256, en la caché de `rosetta models path`). Las siguientes corridas son offline.

## Uso rápido

```sh
rosetta entrevista.mp4                       # → entrevista.md junto al archivo
rosetta audio.wav -f srt -o subs.srt         # subtítulos
rosetta reunion.m4a --diarize --denoise      # hablantes + reducción de ruido
rosetta podcast.mp3 --model whisper-large-v3-turbo --language es
rosetta --batch ./grabaciones -d ./salida -f json
rosetta info                                 # hardware detectado + cascada de aceleración
rosetta doctor                               # qué Execution Providers están realmente disponibles
```

## Modelos y licencias

Los modelos se bajan bajo demanda a una caché por usuario (`rosetta models path`) y se verifican por SHA-256.

| Modelo | Uso | Licencia |
|--------|-----|----------|
| `parakeet-tdt-0.6b-v3` | ASR por defecto | CC-BY-4.0 (NVIDIA) |
| `whisper-large-v3-turbo` | ASR de alta calidad | MIT (OpenAI / onnx-community) |
| `silero-vad` | detección de voz | MIT |
| `pyannote-segmentation-3.0` | diarización (solapes) | MIT |
| `campplus-sv-zh-en` | embeddings de hablante | Apache-2.0 |
| `gtcrn-simple` | denoise | (ver model card) |

`rosetta models list | pull <id> | verify | rm <id> | path | clean`

## Aceleración y plataformas (notas honestas)

Rosetta nunca afirma "corre en GPU/NPU" por una etiqueta — mide el placement REAL de nodos
(`ROSETTA_ORT_PROFILE` + `rosetta doctor --profile`).

- **GPU discreta (NVIDIA/AMD)**: DirectML (Windows) está cableado; CUDA/TensorRT son scaffold
  (opt-in; valida el % de nodos en hardware real antes de fiarte).
- **Qualcomm Snapdragon X**: DirectML sobre la iGPU Adreno es **más lento que la CPU** para el
  modelo int8 por defecto — usa `--device cpu`. La **NPU Hexagon está bloqueada** por un fallo
  upstream de `QnnHtp.dll`; se retomará con el QNN EP de Windows ML / onnxruntime 1.25+.
- **Intel**: OpenVINO (NPU/GPU). **Apple Silicon**: CoreML.
- Con `--device gpu|npu` explícito y sin ese acelerador disponible, Rosetta **da error** en vez de
  caer a CPU en silencio.

## Actualizar

Si instalaste con el updater habilitado:

```sh
rosetta-update            # autoactualiza a la última release
```

O por tu canal: `npm update -g rosetta-cli`, `brew upgrade rosetta`, o re-ejecuta el instalador.

## Desinstalar

El binario lo quita tu instalador, pero la **caché de modelos** (varios GB) vive en el directorio de
caché del SO. Bórrala antes:

```sh
rosetta models clean      # borra toda la caché de modelos (pide confirmación)
```

## Docker

```sh
docker build -t rosetta .                              # CPU (portable)
docker build --build-arg ACCEL=cuda -t rosetta:cuda .  # GPU NVIDIA (best-effort)

docker run --rm -v rosetta-models:/models -v "$PWD:/audio" rosetta /audio/clip.wav
docker run --rm --gpus all -v rosetta-models:/models -v "$PWD:/audio" rosetta:cuda /audio/clip.wav
```

La detección de GPU es en runtime: la cascada usa el acelerador si está y cae a CPU si no. Monta un
volumen en `/models` para que los modelos persistan entre ejecuciones.

## Compilar desde el código fuente

Requiere Rust ≥ 1.88. Rosetta carga ONNX Runtime de forma dinámica (`load-dynamic`); **en el primer
uso descarga el runtime oficial de tu plataforma** (verificado por sha256) a la caché si no está ya
disponible — no tienes que aportarlo. Puedes forzar con `ORT_DYLIB_PATH` o dejar la librería en
`runtime/<plataforma>/`. En Windows esto mantiene el build **sin C**. (macOS x86_64/Intel no tiene
build oficial; ahí define `ORT_DYLIB_PATH`.)

```sh
git clone https://github.com/dennisgr7/Rosetta && cd Rosetta
# Coloca la librería de onnxruntime en runtime/<plataforma>/ (URLs en ci.yml)
cargo build --release
./target/release/rosetta info
```

`cargo xtask check-no-c` garantiza que los builds de Windows nunca compilen C/C++.

## Licencia

Doble licencia [MIT](LICENSE-MIT) o [Apache-2.0](LICENSE-APACHE), a tu elección.
Los pesos de los modelos se distribuyen bajo sus propias licencias (ver la tabla de arriba).
