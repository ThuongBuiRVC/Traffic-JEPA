# Latent Painter - UTE / Traffic-JEPA — AI City Challenge 2026 Track 2.
# One image, two Python environments (they cannot share pins):
#   * main  (system python)  torch 2.12.0+cu130 — the VQA pipeline + caption client
#   * serve (.venv-serving)  vllm 0.26.0 (torch 2.11.0) — the Qwen3-VL caption server
# CUDA 13.0 to match torch==2.12.0+cu130.
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3-pip python3.12-venv git ffmpeg libgl1 libglib2.0-0 \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /workspace/Traffic-JEPA

# --- main environment: VQA pipeline + caption client (layer-cached on requirements) ---
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages \
        --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt

# --- project code (data/, checkpoints/*.pt, and the venvs are excluded by .dockerignore) ---
COPY . .
RUN chmod +x scripts/*.sh serving/*.sh entrypoint.sh

# --- caption server environment (vllm) at .venv-serving. Baked in so the image is ready to
#     caption offline; set --build-arg BUILD_SERVING=0 to skip it and let the first caption
#     run build it instead (scripts/serve_lib.sh does this automatically). ---
ARG BUILD_SERVING=1
RUN if [ "$BUILD_SERVING" = "1" ]; then bash scripts/setup_serving.sh; fi

# torch.hub (V-JEPA 2.1) and HuggingFace (Llama-3.2-1B, EmbeddingGemma, Qwen3-VL-8B) download here.
ENV HF_HOME=/workspace/.cache/huggingface TORCH_HOME=/workspace/.cache/torch

# HF_TOKEN is required at runtime (Llama-3.2-1B + EmbeddingGemma are gated). Mount the data and,
# optionally, a pre-downloaded checkpoints/ directory:
#   docker run --gpus all -e HF_TOKEN=hf_xxx \
#       -v $PWD/data:/workspace/Traffic-JEPA/data \
#       -v $PWD/checkpoints:/workspace/Traffic-JEPA/checkpoints \
#       traffic-jepa inference          # VQA + captions -> submissions/
ENTRYPOINT ["bash", "entrypoint.sh"]
CMD ["inference"]
