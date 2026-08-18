#!/usr/bin/env bash
# Both submissions in one go: VQA answers, then the captions written from them.
#
#   bash scripts/inference.sh            # lora captions, our highest-scoring mode
#   bash scripts/inference.sh mm         # captions the way the paper describes
#   bash scripts/inference.sh base       # lora without the adapter, a baseline
#
# The caption step serves Qwen3-VL itself and stops the server when it is done, so this is one
# command. CKPT overrides the VQA checkpoint, CAPTION_LORA / CAPTION_LORA_MM the caption adapters.
# Stage ordering is intentional: caption generation reads the decoded VQA answers produced by
# 04_submit_test.sh, so the two stages must not run concurrently.
set -euo pipefail
here="$(dirname "$0")"
source "$here/env.sh"
MODE=lora
case "${1:-}" in
  lora|base|mm) MODE="$1"; shift ;;
esac

echo "== [1/2] VQA =="
bash "$here/04_submit_test.sh" "${CKPT:-$TJ_ROOT/checkpoints/model_best.pt}"

echo "== [2/2] captions ($MODE) =="
bash "$here/05_caption.sh" "$MODE"

echo
echo "== done =="
echo "   SubTask2  $SUBS/submission_final.json"
if [ "$MODE" = "mm" ]; then
  echo "   SubTask1  $SUBS/caption_submission_mm.json"
else
  echo "   SubTask1  $SUBS/caption_submission.json"
fi
