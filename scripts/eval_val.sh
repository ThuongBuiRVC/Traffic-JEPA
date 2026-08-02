#!/usr/bin/env bash
# Optional: measure real-val accuracy. Not part of the main pipeline (the submission
# does not depend on it). Re-evaluates the trained checkpoint on real, then graph-decodes
# the real-val split and prints per-category accuracy.
source "$(dirname "$0")/env.sh"
CKPT="${1:-$RUNDIR/model_best.pt}"; RD="$(dirname "$CKPT")"
echo "== re-eval on real -> scored predictions =="
$PY -m traffic_jepa.evaluation.reeval "$RD" --cache "$CACHE" --domain real --save_scores \
  --out_name validation_scored_predictions.jsonl
echo "== graph decode on real-val =="
$PY -m traffic_jepa.postprocess.graph_decode \
  --sim_index "$CACHE/index_sim.jsonl" --real_index "$CACHE/index_real.jsonl" \
  --pred_name validation_scored_predictions.jsonl --out "$TJ_ROOT/runs/validation_decoded.jsonl" $DECODE_ARGS "$RD"
echo "== eval + decode-val done =="
