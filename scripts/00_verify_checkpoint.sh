#!/usr/bin/env bash
# Run inference + graph decode with the shipped checkpoint and compare the
# submission against checkpoints/reference_submission.json. No training needed.
source "$(dirname "$0")/env.sh"
CKPT="$TJ_ROOT/checkpoints/model_best.pt"
INFDIR="$TJ_ROOT/runs/verify"; mkdir -p "$INFDIR"
cp -f "$CKPT" "$INFDIR/model_best.pt"
cp -f "$TJ_ROOT/configs/train_args.json" "$INFDIR/run_args.json"

# The decoder's sim tables come from index_sim.jsonl. A copy is shipped in checkpoints/ so
# verify runs standalone (no preprocess needed); the full pipeline rebuilds it under $CACHE.
SIM_INDEX="$TJ_ROOT/checkpoints/index_sim.jsonl"; [ -f "$SIM_INDEX" ] || SIM_INDEX="$CACHE/index_sim.jsonl"

echo "== inference + postprocess on public test =="
$PY -m traffic_jepa.inference.submit --test "$TJ_ROOT/configs/WTS_VQA_PUBLIC_TEST.json" --route "$TJ_ROOT/configs/route.json" \
  --bbox-root "$TEST_BBOX" --videos-root "$TEST_VIDEOS" \
  --sim_index "$SIM_INDEX" \
  --ckpt "$INFDIR/model_best.pt" --llama meta-llama/Llama-3.2-1B --gemma google/embeddinggemma-300m \
  --raw_out "$SUBS/verify_raw.json" --out "$SUBS/verify_final.json" \
  --batch_size 64 $DECODE_ARGS
echo "== compare against the reference =="
$PY -m traffic_jepa.evaluation.compare_submissions "$TJ_ROOT/checkpoints/reference_submission.json" "$SUBS/verify_final.json"
echo "== verify done =="
