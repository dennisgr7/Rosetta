# Rosetta

**On-device audio & video transcription in Rust.** Fast, private, multiplatform. No cloud,
no telemetry — your audio never leaves your machine.

[![CI](https://github.com/dennisgr7/Rosetta/actions/workflows/ci.yml/badge.svg)](https://github.com/dennisgr7/Rosetta/actions/workflows/ci.yml)
&nbsp;·&nbsp; License: MIT OR Apache-2.0 &nbsp;·&nbsp; Windows · Linux · macOS (x86-64 & ARM64)

🇪🇸 [Léeme en español](README.es.md)

---

Rosetta picks the best available accelerator on each machine (**NPU → GPU → CPU**, with an
automatic, *honest* fallback) and runs ONNX models locally via [ONNX Runtime](https://onnxruntime.ai/).
Models are downloaded on demand and cached; the binary ships with no models inside.

## Features

- **Default engine — NVIDIA Parakeet TDT 0.6B v3**: fast (RTF ~0.1–0.2), strong Spanish, 25 languages.
- **Max quality (opt-in) — Whisper large-v3-turbo**: 99 languages, ONNX int8 (`--model whisper-large-v3-turbo`).
- **Speaker diarization** (`--diarize`), **denoise** (`--denoise`), word-level timestamps.
- **Forced language** (`--language es`) or auto-detection, with honest language metadata.
- **Outputs**: Markdown, JSON, plain text, **SRT**, **VTT** — to a file or stdout.
- **Batch mode** over a directory, with bounded parallelism.
- Any input format (audio or video): `symphonia` for common audio, `ffmpeg` for the rest.

## Install

Prebuilt installers are published with every release.

```sh
# Linux / macOS
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/dennisgr7/Rosetta/releases/latest/download/rosetta-installer.sh | sh

# Windows (PowerShell)
powershell -c "irm https://github.com/dennisgr7/Rosetta/releases/latest/download/rosetta-installer.ps1 | iex"

# npm        (any OS with Node ≥ 14)
npm install -g rosetta-cli

# Homebrew   (macOS / Linux)
brew install dennisgr7/tap/rosetta

# Docker     (see "Docker" below)
docker run --rm -v rosetta-models:/models -v "$PWD:/audio" ghcr.io/dennisgr7/rosetta /audio/clip.wav
```

Or build from source (see [Building](#building-from-source)).

> **First run** downloads, on demand, the ONNX Runtime for your platform and the model you use
> (both sha256-verified, cached under `rosetta models path`). Subsequent runs are offline.

## Quick start

```sh
rosetta interview.mp4                       # → interview.md next to the file
rosetta audio.wav -f srt -o subs.srt        # subtitles
rosetta meeting.m4a --diarize --denoise     # speakers + noise reduction
rosetta podcast.mp3 --model whisper-large-v3-turbo --language en
rosetta --batch ./recordings -d ./out -f json
rosetta info                                # detected hardware + acceleration cascade
rosetta doctor                              # which Execution Providers are actually available
```

## Models & licenses

Models are fetched on demand to a per-user cache (`rosetta models path`) and verified by SHA-256.

| Model | Use | License |
|-------|-----|---------|
| `parakeet-tdt-0.6b-v3` | default ASR | CC-BY-4.0 (NVIDIA) |
| `whisper-large-v3-turbo` | high-quality ASR | MIT (OpenAI / onnx-community) |
| `silero-vad` | voice activity detection | MIT |
| `pyannote-segmentation-3.0` | diarization (overlap) | MIT |
| `campplus-sv-zh-en` | speaker embeddings | Apache-2.0 |
| `gtcrn-simple` | denoise | (see model card) |

`rosetta models list | pull <id> | verify | rm <id> | path | clean`

## Acceleration & platforms (honest notes)

Rosetta never claims "runs on GPU/NPU" by a label — it measures real node placement
(`ROSETTA_ORT_PROFILE` + `rosetta doctor --profile`).

- **Discrete GPU (NVIDIA/AMD)**: DirectML (Windows) is wired; CUDA/TensorRT are scaffolded
  (opt-in, validate node placement on real hardware before trusting).
- **Qualcomm Snapdragon X**: DirectML on the Adreno iGPU is **slower than CPU** for the small
  int8 default model — prefer `--device cpu`. The Hexagon **NPU is currently blocked** by an
  upstream `QnnHtp.dll` crash; it will be revisited with the Windows ML QNN EP / onnxruntime 1.25+.
- **Intel**: OpenVINO (NPU/GPU). **Apple Silicon**: CoreML.
- With `--device gpu|npu` explicit and no such accelerator available, Rosetta **errors out**
  instead of silently falling back to CPU.

## Updating

If you installed via an installer with the updater enabled:

```sh
rosetta-update            # self-update to the latest release
```

Or use your channel: `npm update -g rosetta-cli`, `brew upgrade rosetta`, or re-run the install one-liner.

## Uninstall

The binary is removed by your installer, but the **model cache** (potentially several GB) lives in
your OS cache dir. Clean it first:

```sh
rosetta models clean      # wipes the whole model cache (asks for confirmation)
```

## Docker

```sh
docker build -t rosetta .                              # CPU (portable)
docker build --build-arg ACCEL=cuda -t rosetta:cuda .  # NVIDIA GPU (best-effort)

docker run --rm -v rosetta-models:/models -v "$PWD:/audio" rosetta /audio/clip.wav
docker run --rm --gpus all -v rosetta-models:/models -v "$PWD:/audio" rosetta:cuda /audio/clip.wav
```

GPU detection is at runtime: the cascade uses the accelerator if present and falls back to CPU
otherwise. Mount a named volume at `/models` so models persist across runs.

## Building from source

Requires Rust ≥ 1.88. Rosetta loads ONNX Runtime dynamically (`load-dynamic`); **on first run it
downloads the official runtime for your platform** (sha256-verified) into the model cache if it
isn't already available — so you don't have to. You can override with `ORT_DYLIB_PATH` or drop the
library in `runtime/<platform>/`. On Windows this keeps the build **C-free**. (macOS x86_64/Intel
has no official runtime build; set `ORT_DYLIB_PATH` there.)

```sh
git clone https://github.com/dennisgr7/Rosetta && cd Rosetta
# Drop the matching onnxruntime shared library into runtime/<platform>/ (see ci.yml for URLs)
cargo build --release
./target/release/rosetta info
```

`cargo xtask check-no-c` enforces that Windows builds never compile C/C++.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Model weights are distributed under their own licenses (see the table above).
