#!/usr/bin/env bash
# Container entrypoint. Two modes, each taking an optional caption mode (lora | base | mm):
#   inference   VQA + captions -> both submissions   (default)
#   train       trains the VQA predictor + the caption LoRA
# Anything else runs verbatim.
#   docker run --gpus all -e HF_TOKEN=hf_xxx -v ... traffic-jepa inference
#   docker run --gpus all -e HF_TOKEN=hf_xxx -v ... traffic-jepa inference mm
#   docker run --gpus all -e HF_TOKEN=hf_xxx -v ... traffic-jepa train
set -euo pipefail

# Weights are not baked into the image. Pull them once into the mounted checkpoints/ dir, and
# check every one of them so a partial directory is completed rather than left broken.
for f in model_best.pt caption_lora/adapter_model.safetensors caption_lora_mm/adapter_model.safetensors; do
  if [ ! -f "checkpoints/$f" ]; then
    echo "== checkpoints/$f missing, fetching the weights from the Hub =="
    hf download ThuongBuiRVC/Traffic-JEPA --local-dir checkpoints/
    break
  fi
done

case "${1:-inference}" in
  inference) shift; exec bash scripts/inference.sh "$@" ;;
  train)     shift; exec bash scripts/train.sh "$@" ;;
  *)         exec "$@" ;;
esac
