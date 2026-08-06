#!/usr/bin/env bash
# VQA answers -> Qwen3-VL-8B (+ LoRA) -> caption submission.
#
#   bash scripts/05_caption.sh              # lora: facts -> caption, our highest-scoring mode
#   bash scripts/05_caption.sh mm           # mm: facts + frames on every segment, the paper's method
#   bash scripts/05_caption.sh base         # lora without the adapter, a baseline
#   bash scripts/05_caption.sh --check       # any flag works with or without a mode
#
#   bash scripts/05_caption.sh runs/caption_lora        # a LoRA you trained
#   bash scripts/05_caption.sh mm runs/caption_lora_mm  # the mode comes first
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

# The adapter to caption with, either as a leading directory the way 04_submit_test.sh takes the
# checkpoint, or as --lora DIR. It has to be picked up here, because the server is what loads it.
LORA_ARG=""
case "${1:-}" in
  ""|-*) ;;
  *) LORA_ARG="$1"; shift ;;
esac
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --lora)   LORA_ARG="${2:-}"; shift 2 ;;
    --lora=*) LORA_ARG="${1#--lora=}"; shift ;;
    *)        ARGS+=("$1"); shift ;;
  esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}
if [ -n "$LORA_ARG" ]; then
  if [ "$MODE" = "mm" ]; then export CAPTION_LORA_MM="$LORA_ARG"; else export CAPTION_LORA="$LORA_ARG"; fi
fi

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
