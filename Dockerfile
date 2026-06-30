# Rosetta — imagen Docker. CPU por defecto; GPU (CUDA) opt-in.
#
#   docker build -t rosetta .                              # CPU (portable)
#   docker build --build-arg ACCEL=cuda -t rosetta:cuda .  # GPU NVIDIA
#
#   docker run --rm -v rosetta-models:/models -v "$PWD:/audio" rosetta /audio/clip.wav
#   docker run --rm --gpus all -v rosetta-models:/models -v "$PWD:/audio" rosetta:cuda /audio/clip.wav
#
# La detección de acelerador es EN RUNTIME: la cascada de Rosetta usa el EP
# disponible y cae a CPU si no hay GPU (o si no se pasa `--gpus all`). El binario
# usa `load-dynamic`, así que la dylib de ONNX Runtime viaja como sidecar y se
# resuelve por ORT_DYLIB_PATH.
#
# NOTA: el camino CUDA depende del scaffold de EP CUDA (no validado en hardware);
# se ofrece como best-effort. El camino CPU es el soportado.

ARG ACCEL=cpu
ARG ORT_VER=1.24.4

# ---- build -----------------------------------------------------------------
FROM rust:1.88-bookworm AS build
ARG ACCEL
ARG ORT_VER
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Runtime ONNX (CPU o GPU según ACCEL), con verificación sha256 del asset exacto.
RUN set -eux; \
    if [ "$ACCEL" = "cuda" ]; then \
        name="onnxruntime-linux-x64-gpu-${ORT_VER}"; \
        sha="c5f804ff5d239b436fa59e9f2fb288a39f7eb9552f6a636c8b71e792e91a8808"; \
    else \
        name="onnxruntime-linux-x64-${ORT_VER}"; \
        sha="3a211fbea252c1e66290658f1b735b772056149f28321e71c308942cdb54b747"; \
    fi; \
    curl -fL -o /tmp/ort.tgz "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${name}.tgz"; \
    echo "${sha}  /tmp/ort.tgz" | sha256sum -c -; \
    mkdir -p /ort && tar xf /tmp/ort.tgz -C /ort --strip-components=1; \
    cp -a /ort/lib/libonnxruntime.so* /usr/local/lib/; \
    cp -a /ort/lib/libonnxruntime_providers_*.so /usr/local/lib/ 2>/dev/null || true
COPY . .
RUN cargo build --release -p rosetta-cli

# ---- bases de runtime (se elige una con ACCEL) -----------------------------
FROM debian:bookworm-slim AS base-cpu
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base-cuda
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- imagen final ----------------------------------------------------------
FROM base-${ACCEL} AS final
COPY --from=build /app/target/release/rosetta /usr/local/bin/rosetta
COPY --from=build /usr/local/lib/libonnxruntime*.so* /usr/local/lib/
RUN ldconfig
ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
    ROSETTA_MODELS_DIR=/models \
    LD_LIBRARY_PATH=/usr/local/lib
# Caché de modelos persistente entre ejecuciones (evita re-descargar varios GB).
VOLUME ["/models"]
ENTRYPOINT ["rosetta"]
CMD ["--help"]
