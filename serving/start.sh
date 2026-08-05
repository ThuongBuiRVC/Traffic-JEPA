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
TJ_ROOT="$here"
PORT="${PORT:-8100}"
MODEL="${CAPTION_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"

# vllm lives in its own virtualenv, see scripts/setup_serving.sh
source "$here/scripts/serve_lib.sh"
VLLM="${VLLM_BIN:-$(vllm_bin || true)}"
[ -n "$VLLM" ] || {
  echo "ERROR: no vllm found. Build its environment: bash scripts/setup_serving.sh"; exit 1; }
# the served names stay fixed, the directories behind them follow CAPTION_LORA / CAPTION_LORA_MM
LORA="${CAPTION_LORA:-$here/checkpoints/caption_lora}"
LORA_MM="${CAPTION_LORA_MM:-$here/checkpoints/caption_lora_mm}"
for d in "$LORA" "$LORA_MM"; do
  [ -f "$d/adapter_config.json" ] || {
    echo "ERROR: $d is not a LoRA adapter directory."
    echo "       hf download ThuongBuiRVC/Traffic-JEPA --local-dir checkpoints/"
    exit 1; }
done
echo "serving caption_lora=$LORA"
echo "serving caption_lora_mm=$LORA_MM"

exec "$VLLM" serve "$MODEL" \
  --served-model-name qwen3-vl-8b \
  --port "$PORT" \
  --api-key not-needed \
  --enable-prefix-caching \
  --gpu-memory-utilization "${GPU_UTIL:-0.92}" \
  --max-model-len "${MAX_LEN:-8192}" \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules \
      "caption_lora=$LORA" \
      "caption_lora_mm=$LORA_MM"
