# Qwen3.6-35B-A3B serving via llama.cpp + Unsloth GGUF
#
# Builds llama.cpp from source against CUDA 12.4 for sm_86 + sm_89 (fat binary).
# sm_89: RTX 4000 SFF Ada (GEX44), NVIDIA L4, RTX 4090, RTX 4080
# sm_86: NVIDIA A10G, RTX 3090, A40
# Model is downloaded at runtime to a /models volume — keep it persistent
# across restarts to avoid re-pulling 16 GB.
#
# Build:
#   docker build -f path-a-llamacpp.Dockerfile -t ghcr.io/<you>/qwen36-llamacpp:latest .
#   docker push ghcr.io/<you>/qwen36-llamacpp:latest
#
# Run locally:
#   docker run --gpus all -p 8000:8000 \
#     -v $PWD/models:/models \
#     -e HF_TOKEN=$HF_TOKEN \
#     ghcr.io/<you>/qwen36-llamacpp:latest

# ---------- Build stage ----------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git cmake curl wget ca-certificates \
        libcurl4-openssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/ggml-org/llama.cpp.git
WORKDIR /opt/llama.cpp

# Pin to a specific commit for reproducibility.
# Update this when you re-build; "master" is fine for testing but unstable across rebuilds.
# Required: at least PR #19408 (qwen3_5_moe arch) and build >= b8121 (--chat-template-kwargs).
ARG LLAMA_REF=master
RUN git checkout ${LLAMA_REF}

RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so && \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so.1 && \
    cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="86;89" \
        -DLLAMA_CURL=ON \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j2 --config Release --target llama-server

# ---------- Runtime stage ----------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl4 python3-pip ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U "huggingface_hub[cli]"

# Copy the binary plus libggml shared libs llama-server depends on
COPY --from=builder /opt/llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /opt/llama.cpp/build/bin/*.so /usr/local/lib/
RUN ldconfig

RUN mkdir -p /models
WORKDIR /models

# Configurable model + quant via env vars (defaults to UD-IQ4_XS)
ENV HF_REPO=unsloth/Qwen3.6-35B-A3B-GGUF
ENV HF_QUANT=UD-IQ4_XS
ENV CONTEXT_LEN=16384
ENV PORT=8000

EXPOSE 8000

# Entrypoint: serve /health immediately, download model, then hand off to llama-server
COPY --chmod=755 <<'EOF' /usr/local/bin/entrypoint.sh
#!/bin/bash
set -e

echo "[entrypoint] Starting — PORT=${PORT} HF_REPO=${HF_REPO} HF_QUANT=${HF_QUANT}"

MODEL_DIR=/models/qwen36
GGUF_FILE=$(ls ${MODEL_DIR}/*${HF_QUANT}*.gguf 2>/dev/null | head -n1 || true)

if [ -z "${GGUF_FILE}" ]; then
    echo "[entrypoint] Model not found — starting health stub on port ${PORT} while downloading..."
    # Serve 200 on /health during the download so HF's platform health check passes
    python3 -c "
import http.server, os, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type','application/json')
        self.end_headers()
        self.wfile.write(b'{\"status\":\"loading\"}')
    def log_message(self, *a): pass
port = int(os.environ.get('PORT','8000'))
print(f'[health-stub] listening on {port}', flush=True)
http.server.HTTPServer(('',port),H).serve_forever()
" &
    STUB_PID=$!

    echo "[entrypoint] Downloading ${HF_REPO} quant=${HF_QUANT}..."
    hf download "${HF_REPO}" \
        --include "*${HF_QUANT}*" \
        --local-dir "${MODEL_DIR}"
    GGUF_FILE=$(ls ${MODEL_DIR}/*${HF_QUANT}*.gguf | head -n1)

    echo "[entrypoint] Download done: ${GGUF_FILE} — stopping health stub"
    kill ${STUB_PID} 2>/dev/null || true
    sleep 1  # brief pause so the port is free before llama-server binds
fi

echo "[entrypoint] Starting llama-server: ${GGUF_FILE}"
exec llama-server \
    -m "${GGUF_FILE}" \
    -ngl 999 \
    -c "${CONTEXT_LEN}" \
    -fa on \
    -ctk q8_0 -ctv q8_0 \
    --jinja \
    --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 --presence-penalty 1.5 \
    --chat-template-kwargs '{"enable_thinking":false,"preserve_thinking":true}' \
    --threads 6 \
    --host 0.0.0.0 --port "${PORT}" \
    --metrics
EOF

# Dockerfile-level healthcheck — HF platform may override this with its own
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=10 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
