# Traffic-JEPA runtime. CUDA 13.0 to match the wheels torch==2.12.0+cu130 was built against.
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3-pip python3.12-venv git ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /workspace/Traffic-JEPA

# Python deps first (layer cache). torch/vision from the CUDA 13.0 index.
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        -r requirements.txt

# Project code. Data is mounted at run time; weights are pulled from the Hub by entrypoint.sh.
COPY traffic_jepa/ traffic_jepa/
COPY configs/ configs/
COPY checkpoints/ checkpoints/
COPY scripts/ scripts/
COPY entrypoint.sh README.md ./
RUN chmod +x scripts/*.sh entrypoint.sh

# HuggingFace + torch.hub caches (V-JEPA 2 / Llama / Gemma download here on first run).
ENV HF_HOME=/workspace/.cache/huggingface TORCH_HOME=/workspace/.cache/torch
# HF_TOKEN must be provided at runtime (meta-llama/Llama-3.2-1B is gated):
#   docker run --gpus all -e HF_TOKEN=hf_xxx \
#       -v /path/to/wts_data:/workspace/Traffic-JEPA/data \
#       traffic-jepa inference   # or: train
ENTRYPOINT ["bash", "entrypoint.sh"]
CMD ["inference"]
