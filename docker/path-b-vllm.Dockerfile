# Qwen3.6-35B-A3B serving via vLLM + HLWQ CT INT4 (Marlin kernels)
#
# caiovicentino1/Qwen3.6-35B-A3B-HLWQ-CT-INT4 is 19.4 GB total vs ~24 GB
# for cyankiwi AWQ because it quantizes more layers (only 1.3 GB BF16 vs 3.7 GB).
# At gpu-memory-utilization=0.95 on A10G 24 GB: 22.8 GB usable - 19.4 GB model
# = ~3.4 GB for fp8 KV cache, enough for 36 K context.
#
# Build:
#   docker build -f path-b-vllm.Dockerfile -t ghcr.io/<you>/qwen36-vllm:latest .
#   docker push ghcr.io/<you>/qwen36-vllm:latest

FROM vllm/vllm-openai:v0.19.1

ENV DEBIAN_FRONTEND=noninteractive

# Install caiovicentino's expert-offload fork on top of the base vLLM image.
# This adds --moe-expert-cache-size and per-expert compressed-tensors loading,
# both required for caiovicentino1/Qwen3.6-35B-A3B-HLWQ-CT-INT4.
# Patch in caiovicentino's expert-offload fork (Python-only changes).
# We clone and overwrite vllm's Python package in-place so the already-compiled
# CUDA extensions from the base image are preserved — no recompilation needed.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && git clone --depth=1 https://github.com/caiovicentino/vllm-expert-offload.git /tmp/vllm-fork \
    && VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))") \
    && cp -r /tmp/vllm-fork/vllm/. "$VLLM_DIR/" \
    && rm -rf /tmp/vllm-fork \
    && apt-get purge -y git && rm -rf /var/lib/apt/lists/*

# Multiproc method explicitly required for Qwen3.5/3.6 family per Qwen docs
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# Configurable via env vars
ENV MODEL_REPO=caiovicentino1/Qwen3.6-35B-A3B-HLWQ-CT-INT4
ENV SERVED_NAME=qwen3.6-35b
ENV MAX_MODEL_LEN=36000
ENV GPU_MEMORY_UTILIZATION=0.95
ENV KV_CACHE_DTYPE=fp8
ENV PORT=8000

EXPOSE 8000

# Health stub responds to /health immediately while vLLM downloads + loads the model.
# Without it, HF's platform health check times out and kills the endpoint.
COPY --chmod=755 <<'EOF' /usr/local/bin/entrypoint.sh
#!/bin/bash
set -e

echo "[entrypoint] Starting health stub on port ${PORT}..."
python3 -c "
import http.server, os, threading, time

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type','application/json')
        self.end_headers()
        self.wfile.write(b'{\"status\":\"loading\"}')
    def log_message(self, *a): pass

port = int(os.environ.get('PORT','8000'))
server = http.server.HTTPServer(('',port), H)
print(f'[health-stub] listening on {port}', flush=True)

def serve():
    try:
        server.serve_forever()
    except Exception:
        pass

t = threading.Thread(target=serve, daemon=True)
t.start()
time.sleep(2)
server.shutdown()
" &
sleep 3

echo "[entrypoint] Launching vLLM..."
exec vllm serve ${MODEL_REPO} \
    --served-model-name ${SERVED_NAME} \
    --quantization compressed-tensors \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --kv-cache-dtype ${KV_CACHE_DTYPE} \
    --enforce-eager \
    --moe-expert-cache-size 64 \
    --reasoning-parser qwen3 \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --default-chat-template-kwargs '{"enable_thinking":false,"preserve_thinking":true}' \
    --host 0.0.0.0 --port ${PORT}
EOF

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
