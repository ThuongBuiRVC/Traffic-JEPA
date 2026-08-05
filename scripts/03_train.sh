#!/usr/bin/env bash

source "$(dirname "$0")/env.sh"
$PY -m traffic_jepa.training.train \
  --spatial_pool 4 \
  --predictor meta-llama/Llama-3.2-1B --predictor_layers 6 --cache "$CACHE" \
  --seed 31 --epochs 10 --eval_every 1 \
  --batch_size 2 --grad_accum 2 --lr 0.00012 --backbone_lr 0.0 --backbone_wd 0.05 \
  --dropout 0.12 --token_dropout 0.08 --temperature 0.07 \
  --scheduler cosine --warmup_steps 300 --min_lr_scale 0.08 \
  --grad_clip 1.0 --workers 6 --out "$RUNDIR" "$@"
echo "== train done -> $RUNDIR/model_latest.pt (the last epoch) =="
