# Caption server

The caption stage can load Qwen3-VL itself, but then it generates one segment at a time. A served
endpoint batches requests, so the segments go out concurrently and the run finishes much sooner.
One server holds the base model and both adapters, so it covers all three caption modes.

## Start

```bash
pip install vllm
bash serving/start.sh
```

It runs in the foreground on port 8100. `PORT`, `GPU_UTIL` and `MAX_LEN` override the defaults.

## Check

```bash
curl -s http://localhost:8100/v1/models -H "Authorization: Bearer not-needed"
```

Three names have to come back: `qwen3-vl-8b`, `caption_lora`, `caption_lora_mm`. The caption stage
refuses to start if any is missing, rather than captioning with the wrong weights.

## Use it

```bash
export CAPTION_SERVER=http://localhost:8100/v1
bash scripts/05_caption.sh          # or: mm | base
```

Each mode picks its own served name. `lora` calls `caption_lora` and falls back to `qwen3-vl-8b` for
the segments with no VQA answers, because that adapter is text-only. `mm` calls `caption_lora_mm`
throughout. `base` calls `qwen3-vl-8b`.

`CAPTION_WORKERS` sets how many segments are in flight, 8 by default. Raise it if the GPU is idle,
lower it if the server starts refusing requests.

## VRAM

The server holds the 8B model in bf16 plus both adapters, about 20 GB, and keeps holding it until
you stop it. Leave it up for the whole caption run and stop it before training on the same GPU.
