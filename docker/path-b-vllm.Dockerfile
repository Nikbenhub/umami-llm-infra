# Qwen3.6-35B-A3B serving via vLLM + cyankiwi AWQ-4bit
#
# Built on top of the official vLLM image. Adds the launch command and
# all the workarounds documented in the deployment guide for the
# qwen3_5_moe / hybrid Gated DeltaNet architecture issues.
#
# Build:
#   docker build -f path-b-vllm.Dockerfile -t ghcr.io/<you>/qwen36-vllm:latest .
#   docker push ghcr.io/<you>/qwen36-vllm:latest
#
# Run locally:
#   docker run --gpus all -p 8000:8000 \
#     -v $PWD/models:/root/.cache/huggingface \
#     -e HF_TOKEN=$HF_TOKEN \
#     --shm-size=8gb \
#     ghcr.io/<you>/qwen36-vllm:latest

# Pin to a specific vLLM version. >= 0.19.0 required for qwen3_5_moe.
# 0.19.1 has Qwen3.5 GDN and AWQ-Marlin fixes.
FROM vllm/vllm-openai:v0.19.1

ENV DEBIAN_FRONTEND=noninteractive

# Multiproc method explicitly required for Qwen3.5/3.6 family per Qwen docs
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# Set this to "0" if V1 backend crashes on Qwen3 MoE (issue #24436)
# ENV VLLM_USE_V1=0

# Configurable via env vars
ENV MODEL_REPO=cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit
ENV SERVED_NAME=qwen3.6-35b
# 36K covers our largest billing prompt (~34K tokens). fp8 KV is required to fit
# on A10G 24GB — bf16 KV would need ~4GB for 36K context vs ~2GB for fp8.
ENV MAX_MODEL_LEN=36000
ENV GPU_MEMORY_UTILIZATION=0.85
ENV KV_CACHE_DTYPE=fp8
ENV PORT=8000

EXPOSE 8000

# Override the upstream entrypoint with our locked-down launch command.
# Notable defaults that diverge from "default" vLLM advice:
#   --enforce-eager         (issue #38486, #35743 — CUDA graphs crash with mamba cache)
#   --no-enable-prefix-caching (issue #26201 — prefix cache corrupts tool calls on hybrid GDN)
#   no --speculative-config (MTP is net-negative on Ada per benchmarks)
#   no --enable-chunked-prefill (issue #22616 — slow on high context for MoE)
#   --gpu-memory-utilization 0.85 (issue #37121 — vLLM over-allocates KV ~7x for hybrid)
#   removed --language-model-only (wrong for chat/instruct models, breaks chat template)
ENTRYPOINT ["/bin/sh", "-c", "exec vllm serve ${MODEL_REPO} \
    --served-model-name ${SERVED_NAME} \
    --quantization awq_marlin \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --kv-cache-dtype ${KV_CACHE_DTYPE} \
    --enforce-eager \
    --reasoning-parser qwen3 \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --default-chat-template-kwargs '{\"enable_thinking\":false,\"preserve_thinking\":true}' \
    --host 0.0.0.0 --port ${PORT}"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1
