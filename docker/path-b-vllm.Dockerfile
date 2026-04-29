# Qwen3.6-35B-A3B serving via vLLM + GPTQ Int4
#
# btbtyler09/Qwen3.6-35B-A3B-GPTQ-4bit uses standard GPTQ quantization that
# vLLM 0.19.1 supports natively (Marlin GPTQ kernels). ~20 GB GPU footprint,
# leaving ~3 GB for fp8 KV cache at 36K context on A10G 24 GB.
#
# Build:
#   docker build -f path-b-vllm.Dockerfile -t ghcr.io/<you>/qwen36-vllm:latest .
#   docker push ghcr.io/<you>/qwen36-vllm:latest

FROM vllm/vllm-openai:v0.19.1

ENV DEBIAN_FRONTEND=noninteractive

# Multiproc method explicitly required for Qwen3.5/3.6 family per Qwen docs
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# Configurable via env vars
ENV MODEL_REPO=btbtyler09/Qwen3.6-35B-A3B-GPTQ-4bit
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
    --quantization gptq \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --kv-cache-dtype fp8 \
    --enforce-eager \
    --language-model-only \
    --reasoning-parser qwen3 \
    --no-enable-prefix-caching \
    --default-chat-template-kwargs '{"enable_thinking":false,"preserve_thinking":true}' \
    --host 0.0.0.0 --port ${PORT}
EOF

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
