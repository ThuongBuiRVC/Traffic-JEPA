# Caption server

The caption stage can load Qwen3-VL itself, but then it generates one segment at a time. A served
endpoint batches requests across its own queue, so the 414 segments go out concurrently and the run
finishes in a fraction of the time. This serves the base model and both adapters at once.

## Start

```bash
cd serving
HF_TOKEN=hf_xxxxx docker compose up -d
```

Or without Docker:

```bash
pip install vllm
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --served-model-name qwen3-vl-8b --port 8100 --api-key not-needed \
    --enable-prefix-caching --gpu-memory-utilization 0.92 --max-model-len 8192 \
    --enable-lora --max-lora-rank 16 \
    --lora-modules caption_lora=../checkpoints/caption_lora \
                   caption_lora_mm=../checkpoints/caption_lora_mm
```

## Check

```bash
curl -s http://localhost:8100/v1/models -H "Authorization: Bearer not-needed"
```

Three names have to come back: `qwen3-vl-8b`, `caption_lora`, `caption_lora_mm`. The caption stage
refuses to start if any is missing, rather than silently captioning with the wrong weights.

## Use it

```bash
export CAPTION_SERVER=http://localhost:8100/v1
bash scripts/05_caption.sh          # or: mm | base
```

Every mode picks its own served name. `lora` calls `caption_lora` and falls back to `qwen3-vl-8b`
for the segments with no VQA answers, because that adapter is text-only. `mm` calls
`caption_lora_mm` throughout. `base` calls `qwen3-vl-8b`.

`CAPTION_WORKERS` sets how many segments are in flight, 8 by default. Raise it if the GPU is idle,
lower it if the server starts rejecting requests.

## VRAM

The server holds the 8B model in bf16 plus both adapters, so about 20 GB, and it keeps holding it
until you stop it. Leave it running for the whole caption run and stop it before training anything
else on the same GPU.
