#!/usr/bin/env bash
# Train both models from raw data: the VQA predictor, then the caption LoRA.
#
#   bash scripts/train.sh            # text-only caption LoRA, our highest-scoring mode
#   bash scripts/train.sh mm         # caption LoRA on facts + simulation frames, the paper's method
#
# Writes checkpoints only. Build the submissions afterwards with scripts/inference.sh, which the
# last line prints ready to paste.
set -euo pipefail
here="$(dirname "$0")"
source "$here/env.sh"
MODE=lora
case "${1:-}" in
  lora|base|mm) MODE="$1"; shift ;;
esac
if [ "$MODE" = "mm" ]; then
  CAP_RUN="$TJ_ROOT/runs/caption_lora_mm"
else
  CAP_RUN="$TJ_ROOT/runs/caption_lora"
fi

echo "== [1/4] cut the simulation clips and build the manifest =="
bash "$here/01_manifest_sim.sh"

echo "== [2/4] encode the V-JEPA latents and the option vectors =="
bash "$here/02_preprocess.sh"

echo "== [3/4] train the VQA predictor =="
bash "$here/03_train.sh"

echo "== [4/4] train the caption LoRA ($MODE) =="
if [ "$MODE" = "mm" ]; then
  $PY -m traffic_jepa.captioning_mm.preprocess --split train --out "$CAPTION_MM" \
    --annot "$SYNWTS/annotations" --videos "$SYNWTS/videos"
  $PY -m traffic_jepa.captioning_mm.train --data "$CAPTION_MM/manifests" --out "$CAP_RUN"
else
  $PY -m traffic_jepa.captioning.build_data --annot "$SYNWTS/annotations" \
    --out "$DATA/processed/caption_lora"
  $PY -m traffic_jepa.captioning.train_lora --data "$DATA/processed/caption_lora" --out "$CAP_RUN"
fi

LORA_VAR=$([ "$MODE" = "mm" ] && echo CAPTION_LORA_MM || echo CAPTION_LORA)
echo
echo "== training done =="
echo "   VQA      $RUNDIR/model_latest.pt"
echo "   caption  $CAP_RUN/adapter"
echo
echo "   build the submissions with what you just trained:"
echo "     CKPT=$RUNDIR/model_latest.pt $LORA_VAR=$CAP_RUN/adapter bash scripts/inference.sh $MODE"
