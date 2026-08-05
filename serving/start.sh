#!/usr/bin/env bash
# Serve Qwen3-VL-8B with both caption adapters on one endpoint.
#
#   bash serving/start.sh            # foreground on :8100
#   PORT=8200 bash serving/start.sh  # somewhere else
#
# Then, in another shell:
#   export CAPTION_SERVER=http://localhost:8100/v1
#   bash scripts/05_caption.sh
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8100}"
MODEL="${CAPTION_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"

command -v vllm >/dev/null || { echo "ERROR: vllm not installed. pip install vllm"; exit 1; }
for a in caption_lora caption_lora_mm; do
  [ -f "$here/checkpoints/$a/adapter_config.json" ] || {
    echo "ERROR: $here/checkpoints/$a is missing. hf download ThuongBuiRVC/Traffic-JEPA --local-dir checkpoints/"
    exit 1; }
done

exec vllm serve "$MODEL" \
  --served-model-name qwen3-vl-8b \
  --port "$PORT" \
  --api-key not-needed \
  --enable-prefix-caching \
  --gpu-memory-utilization "${GPU_UTIL:-0.92}" \
  --max-model-len "${MAX_LEN:-8192}" \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules \
      "caption_lora=$here/checkpoints/caption_lora" \
      "caption_lora_mm=$here/checkpoints/caption_lora_mm"
