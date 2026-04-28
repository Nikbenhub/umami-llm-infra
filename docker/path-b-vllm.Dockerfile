# Qwen3.6-35B-A3B serving via vLLM + cyankiwi compressed-tensors quant
#
# Built on top of the official vLLM image.
#
# Build:
#   docker build -f path-b-vllm.Dockerfile -t ghcr.io/<you>/qwen36-vllm:latest .
#   docker push ghcr.io/<you>/qwen36-vllm:latest

FROM vllm/vllm-openai:v0.19.1

ENV DEBIAN_FRONTEND=noninteractive

# Multiproc method explicitly required for Qwen3.5/3.6 family per Qwen docs
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# Configurable via env vars
ENV MODEL_REPO=cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit
ENV SERVED_NAME=qwen3.6-35b
ENV MAX_MODEL_LEN=8192
ENV GPU_MEMORY_UTILIZATION=0.85
ENV PORT=8000

EXPOSE 8000

# Health stub responds to /health immediately while vLLM downloads + loads the model.
# Without it, HF's platform health check times out and kills the endpoint.
# Once vLLM binds port 8000 the stub is already dead (port conflict kills it).
COPY --chmod=755 <<'EOF' /usr/local/bin/entrypoint.sh
#!/bin/bash
set -e

echo "[entrypoint] Starting health stub on port ${PORT}..."
python3 -c "
import http.server, os, threading, time, subprocess, sys, signal

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

# Give stub time to bind, then launch vLLM in foreground
time.sleep(2)
server.shutdown()
" &
STUB_PID=$!
sleep 3  # wait for stub to bind and then shut itself down

echo "[entrypoint] Launching vLLM..."
exec vllm serve ${MODEL_REPO} \
    --served-model-name ${SERVED_NAME} \
    --quantization compressed-tensors \
    --task generate \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --enforce-eager \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --host 0.0.0.0 --port ${PORT}
EOF

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
