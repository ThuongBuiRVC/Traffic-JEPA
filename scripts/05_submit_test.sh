#!/usr/bin/env bash
# Public-test inference -> model scores -> graph/temporal postprocess -> final submission.
# Fails loudly if the test videos are not found (no silent all-fallback submission).
source "$(dirname "$0")/env.sh"
CKPT="${1:-$RUNDIR/model_best.pt}"
$PY -m traffic_jepa.inference.submit --test "$TJ_ROOT/configs/WTS_VQA_PUBLIC_TEST.json" --route "$TJ_ROOT/configs/route.json" \
  --bbox-root "$TEST_BBOX" --videos-root "$TEST_VIDEOS" \
  --sim_index "$CACHE/index_sim.jsonl" \
  --ckpt "$CKPT" --llama meta-llama/Llama-3.2-1B --gemma google/embeddinggemma-300m \
  --raw_out "$SUBS/submission_raw.json" --out "$SUBS/submission_final.json" \
  --batch_size 64 $DECODE_ARGS
