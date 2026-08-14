#!/usr/bin/env bash
# Container entrypoint for the Latent Painter - UTE / Traffic-JEPA submission.
#
#   inference [lora|mm|base]   VQA + captions -> submissions/   (default: lora)
#   vqa                        SubTask2 only -> submissions/submission_final.json
#   caption   [lora|mm|base]   SubTask1 only (needs VQA output already there)
#   train                      retrain the VQA predictor + caption LoRA (simulation only)
#   <anything else>            run verbatim (e.g. `bash`)
#
# Weights download from the Hub on first run unless checkpoints/ is mounted. HF_TOKEN is
# required (Llama-3.2-1B + EmbeddingGemma are gated).
set -euo pipefail
cd /workspace/Traffic-JEPA
CMD="${1:-inference}"

# --- checkpoints: pull from the Hub when the VQA weights are not already mounted ---
if [ "$CMD" != "caption" ] && [ ! -f checkpoints/model_best.pt ]; then
  echo "== checkpoints/model_best.pt not found -> hf download ThuongBuiRVC/Traffic-JEPA =="
  hf download ThuongBuiRVC/Traffic-JEPA --local-dir checkpoints/
fi

# --- V-JEPA 2.1 hub URL fix: upstream ships VJEPA_BASE_URL pointing at a localhost test server.
#     Warm the hub repo (the weight fetch fails fast on localhost), then rewrite the cached URL. ---
patch_vjepa() {
  python -c "import torch; torch.hub.load('facebookresearch/vjepa2','vjepa2_1_vit_large_384',trust_repo=True,pretrained=True)" >/dev/null 2>&1 || true
  find "${TORCH_HOME:-$HOME/.cache/torch}/hub" -name backbones.py -path "*vjepa2*" 2>/dev/null \
    | xargs -r sed -i 's#http://localhost:8300#https://dl.fbaipublicfiles.com/vjepa2#g' || true
}

case "$CMD" in
  inference) shift; patch_vjepa; exec bash scripts/inference.sh "$@" ;;
  vqa)       patch_vjepa; exec bash scripts/04_submit_test.sh "${CKPT:-$PWD/checkpoints/model_best.pt}" ;;
  caption)   shift; exec bash scripts/05_caption.sh "$@" ;;
  train)     patch_vjepa; exec bash scripts/train.sh ;;
  *)         exec "$@" ;;
esac
