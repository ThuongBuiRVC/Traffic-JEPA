#!/usr/bin/env bash
# Public-test inference -> model scores -> graph/temporal decode -> VQA submission.
# Fails loudly if the test videos are not found (no silent all-fallback submission).
#
#   bash scripts/04_submit_test.sh                             # the released checkpoints/model_best.pt
#   bash scripts/04_submit_test.sh runs/<run>/model_latest.pt  # one you trained
source "$(dirname "$0")/env.sh"

CKPT="${1:-$TJ_ROOT/checkpoints/model_best.pt}"
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }

# The model is rebuilt from the training config, so run_args.json must sit next to the
# checkpoint. Training writes it; the shipped checkpoint uses configs/train_args.json.
RD="$(dirname "$CKPT")"
[ -f "$RD/run_args.json" ] || cp -f "$TJ_ROOT/configs/train_args.json" "$RD/run_args.json"

# Decoder sim tables. A copy ships in checkpoints/ so a run needs no preprocess; the full
# pipeline rebuilds it under $CACHE.
SIM_INDEX="$CACHE/index_sim.jsonl"
[ -f "$SIM_INDEX" ] || SIM_INDEX="$TJ_ROOT/checkpoints/index_sim.jsonl"

$PY -m traffic_jepa.inference.submit \
  --test "$TEST_VQA" --route "$TJ_ROOT/configs/route.json" \
  --bbox-root "$TEST_BBOX" --videos-root "$TEST_VIDEOS" \
  --sim_index "$SIM_INDEX" \
  --ckpt "$CKPT" --llama meta-llama/Llama-3.2-1B --gemma google/embeddinggemma-300m \
  --raw_out "$SUBS/submission_raw.json" --out "$SUBS/submission_final.json" \
  --batch_size 64 $DECODE_ARGS
echo "== VQA submission -> $SUBS/submission_final.json =="
