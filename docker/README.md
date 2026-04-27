# Qwen3.6-35B-A3B Docker images

Two Dockerfiles for serving Qwen3.6-35B-A3B with single-stream UX-critical workloads:

- `path-a-llamacpp.Dockerfile` — llama.cpp + `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS`
- `path-b-vllm.Dockerfile` — vLLM + `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`

Both are built for **Ada Lovelace (sm_89)** — works on RTX 4000 SFF Ada (Hetzner GEX44), NVIDIA L4 (HF Endpoints), RTX 4090, RTX 4080, etc. They will not work on Hopper or Blackwell without rebuild (change `CMAKE_CUDA_ARCHITECTURES` to 90 / 100).

See `qwen36-deployment-guide.md` for full context, gotchas, and reasoning behind the launch flags.

## Build and push to ghcr.io

The included GitHub Actions workflow builds both images on push to `main` and publishes to:

- `ghcr.io/<your-org>/qwen36-llamacpp:latest`
- `ghcr.io/<your-org>/qwen36-vllm:latest`

To set this up:

1. Create a GitHub repo (or push these files to an existing one)
2. Push to `main`
3. The workflow will build and tag images automatically
4. Make the resulting `ghcr.io` packages public if you want to pull without auth

To build manually (much faster than CI for first iterations):

```bash
docker build -f docker/path-a-llamacpp.Dockerfile -t ghcr.io/<you>/qwen36-llamacpp:latest docker/
docker build -f docker/path-b-vllm.Dockerfile -t ghcr.io/<you>/qwen36-vllm:latest docker/

# Auth to ghcr.io with a Personal Access Token (read:packages, write:packages)
echo $GHCR_PAT | docker login ghcr.io -u <you> --password-stdin

docker push ghcr.io/<you>/qwen36-llamacpp:latest
docker push ghcr.io/<you>/qwen36-vllm:latest
```

## Run on HF Inference Endpoints (L4 GPU)

For each path, create an endpoint at https://ui.endpoints.huggingface.co/:

1. **Task**: Custom Container
2. **Container image URL**: `ghcr.io/<you>/qwen36-llamacpp:latest` (or `qwen36-vllm:latest`)
3. **Container port**: `8000`
4. **GPU**: L4
5. **Min replicas / max replicas**: 1 / 1 (single-stream test)
6. **Environment variables**:
   - `HF_TOKEN`: your HF read token (so the container can pull the model)
   - For Path A only: `CONTEXT_LEN=16384` (or `32768` to take advantage of L4's 24 GB)
   - For Path B only: `MAX_MODEL_LEN=16384` (or `32768` similarly)
7. **Health check path**: `/health`
8. **Health check start period**: 600 seconds (model download takes time on cold start)
9. **Request timeout**: 600 seconds (long enough for chains to complete)

If your ghcr.io image is private, also set:
- **Container registry username**: `<your-github-username>`
- **Container registry password**: a GitHub PAT with `read:packages`

## Run locally / on Hetzner GEX44

Same image, same command:

```bash
# Path A
docker run -d \
  --name qwen36-llamacpp \
  --gpus all \
  -p 8000:8000 \
  -v /opt/models:/models \
  -e HF_TOKEN=$HF_TOKEN \
  --restart unless-stopped \
  ghcr.io/<you>/qwen36-llamacpp:latest

# Path B
docker run -d \
  --name qwen36-vllm \
  --gpus all \
  -p 8000:8000 \
  -v /opt/models:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  --shm-size=8gb \
  --restart unless-stopped \
  ghcr.io/<you>/qwen36-vllm:latest
```

Mount a host directory for the model cache so you only download the 14–17 GB once.

## Verifying the container actually works

After the container is running, check:

```bash
# 1. Health
curl http://localhost:8000/health

# 2. Model loaded?
curl http://localhost:8000/v1/models

# 3. Smoke test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b",
    "messages": [{"role": "user", "content": "Reply with the word OK and nothing else."}],
    "max_tokens": 5
  }'

# Path B specifically: check that AWQ-Marlin loaded (not slow triton fallback)
docker logs qwen36-vllm 2>&1 | grep -i marlin

# Path A specifically: check VRAM matches expectations
nvidia-smi
```

## Customizing without rebuilding

Both images expose configuration via env vars so you don't need to rebuild for tuning:

| Path A env var | Default | Purpose |
|---|---|---|
| `HF_REPO` | `unsloth/Qwen3.6-35B-A3B-GGUF` | Switch quant author |
| `HF_QUANT` | `UD-IQ4_XS` | Switch quant level (e.g. `UD-Q3_K_XL`, `UD-IQ3_XXS`) |
| `CONTEXT_LEN` | `16384` | Adjust per available VRAM |
| `PORT` | `8000` | Server port |

| Path B env var | Default | Purpose |
|---|---|---|
| `MODEL_REPO` | `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | Switch repo |
| `SERVED_NAME` | `qwen3.6-35b` | OpenAI API model identifier |
| `MAX_MODEL_LEN` | `16384` | Adjust per available VRAM |
| `GPU_MEMORY_UTILIZATION` | `0.85` | Lower if you OOM at startup |
| `PORT` | `8000` | Server port |

## Image sizes (approximate)

- `qwen36-llamacpp`: ~3 GB (CUDA runtime + llama.cpp binary + Python for hf_hub_cli)
- `qwen36-vllm`: ~14 GB (full vLLM image)

Models are NOT baked into the image — they're downloaded at first run. This keeps images portable and small, but means a ~3-5 minute cold start for the first container on a new node. Mount a persistent volume to `/models` (Path A) or `/root/.cache/huggingface` (Path B) to avoid re-downloading.

## When to rebuild

Rebuild Path A when:
- llama.cpp ships a release with `qwen3_5_moe` perf improvements (track issue #19345)
- You need a newer build to pick up bugfixes (e.g. KV cache, CUDA graph fixes)
- Update `LLAMA_REF` build arg to a specific commit for reproducibility

Rebuild Path B when:
- vLLM ships a version that fixes prefix caching for hybrid models (#26201) or KV over-allocation (#37121)
- Update the `FROM vllm/vllm-openai:vX.Y.Z` line and rebuild
