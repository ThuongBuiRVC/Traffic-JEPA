#!/usr/bin/env bash
# VQA answers -> Qwen3-VL-8B (+ LoRA) -> caption submission.
#
#   bash scripts/05_caption.sh              # lora: facts -> caption, our highest-scoring mode
#   bash scripts/05_caption.sh mm           # mm: facts + frames on every segment, the paper's method
#   bash scripts/05_caption.sh base         # lora without the adapter, a baseline
#   bash scripts/05_caption.sh --check       # any flag works with or without a mode
#
# The step serves the model itself and stops the server when it is done. CAPTION_SERVER points it
# at an endpoint that is already up, CAPTION_NO_SERVE=1 loads the model in-process instead.
export TJ_NEED_HF_TOKEN=0
source "$(dirname "$0")/env.sh"
source "$(dirname "$0")/serve_lib.sh"
MODE=lora
case "${1:-}" in
  lora|base|mm) MODE="$1"; shift ;;
esac

# --check only inspects the inputs, so it must not pull up a server
CHECK_ONLY=0
for a in "$@"; do case "$a" in --check) CHECK_ONLY=1 ;; esac; done

if [ "$MODE" = "mm" ]; then
  MANIFEST="$CAPTION_MM/manifests/test.jsonl"
  # cut the test frames once; --check must not trigger the build
  if [ ! -f "$MANIFEST" ] && [ "${1:-}" != "--check" ]; then
    echo "== no test manifest yet -> cutting frames =="
    $PY -m traffic_jepa.captioning_mm.preprocess --split test \
      --out "$CAPTION_MM" --wts-root "$WTS_ROOT" --bbox-root "$TEST_BBOX" \
      --vqa "$SUBS/submission_final.json" --test "$TEST_VQA"
  fi
  [ "$CHECK_ONLY" = "1" ] || { serve_auto; trap serve_stop EXIT; }
  $PY -m traffic_jepa.captioning_mm.generate \
    --manifest "$MANIFEST" \
    --model "$CAPTION_MODEL" \
    --lora "$CAPTION_LORA_MM" \
    ${CAPTION_SERVER:+--server "$CAPTION_SERVER" --workers "$CAPTION_WORKERS"} \
    --out "$SUBS/caption_submission_mm.json" "$@"
  exit
fi

[ "$CHECK_ONLY" = "1" ] || { serve_auto; trap serve_stop EXIT; }
$PY -m traffic_jepa.captioning.generate \
  --mode "$MODE" \
  --model "$CAPTION_MODEL" \
  --lora "$CAPTION_LORA" \
  --vqa "$SUBS/submission_final.json" --test "$TEST_VQA" \
  --wts-root "$WTS_ROOT" \
  ${CAPTION_SERVER:+--server "$CAPTION_SERVER" --workers "$CAPTION_WORKERS"} \
  --out "$SUBS/caption_submission.json" "$@"
