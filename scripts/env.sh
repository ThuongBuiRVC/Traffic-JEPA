#!/usr/bin/env bash
# Shared environment for all step scripts. `source` this from each NN_*.sh.
# All paths are absolute so every stage resolves regardless of the caller's cwd.
#
# Override any of these before running a step, e.g.:
#   export TJ_DATA=/mnt/wts_data   # if your data lives elsewhere
set -euo pipefail

# repo root = parent of this scripts/ dir (works when sourced from a bash step script)
_ENV_SRC="${BASH_SOURCE[0]:-${(%):-%x}}"; _ENV_SRC="${_ENV_SRC:-$0}"
TJ_ROOT="$(cd "$(dirname "$_ENV_SRC")/.." && pwd)"
export TJ_ROOT

# --- where the data lives (defaults to Traffic-JEPA/data; override with TJ_DATA) ---
export DATA="${TJ_DATA:-$TJ_ROOT/data}"
export SYNWTS="${SYNWTS:-$DATA/synwts/data}"                 # SynWTS raw: annotations/ videos/
# The public test set lives under one root. Both stages read out of it, so there is a single
# directory to place and the two paths below follow from it.
export WTS_ROOT="${WTS_ROOT:-$DATA/test}"                    # holds videos/ + annotations/
export TEST_VIDEOS="${TEST_VIDEOS:-$WTS_ROOT/videos/test/public}"
export TEST_BBOX="${TEST_BBOX:-$WTS_ROOT/annotations}"       # holds bbox_generated/ + bbox_annotated/
export TEST_VQA="${TEST_VQA:-$WTS_ROOT/WTS_VQA_PUBLIC_TEST.json}"   # the public-test questions

# --- produced artifacts (all overridable so a clean run can use a separate workspace) ---
export SIMQA="${TJ_SIMQA:-$DATA/processed/sim_qa_vljepa16}"    # cut clips + sim manifests
export CACHE="${TJ_CACHE:-$DATA/processed/cache_vljepa16_8f}"  # V-JEPA latents + Gemma vecs + index
export RUNDIR="${TJ_RUNDIR:-$TJ_ROOT/runs/traffic_jepa_world_model}"
export SUBS="${TJ_SUBS:-$TJ_ROOT/submissions}"

# --- caption stage ---
export CAPTION_LORA="${TJ_CAPTION_LORA:-$TJ_ROOT/checkpoints/caption_lora}"
export CAPTION_LORA_MM="${TJ_CAPTION_LORA_MM:-$TJ_ROOT/checkpoints/caption_lora_mm}"
export CAPTION_MM="${TJ_CAPTION_MM:-$DATA/processed/caption_mm}"   # grounded variant: frames + manifests
export CAPTION_MODEL="${CAPTION_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
# Empty means the caption step serves the model itself. Set this to a vLLM endpoint that is
# already up to caption through that one instead.
export CAPTION_SERVER="${CAPTION_SERVER:-}"
export CAPTION_WORKERS="${CAPTION_WORKERS:-8}"

# --- python / package ---
export PYTHONPATH="$TJ_ROOT:${PYTHONPATH:-}"
export VLJEPA_DATA="$DATA"          # preprocess reads manifests from $DATA/processed/*_qa_vljepa16/_review
export VLJEPA_ROOT="$DATA"          # clip_16 paths are absolute, so this is only a harmless base
export VLJEPA_CACHE="$CACHE"
export VLJEPA_OUT="$CACHE"
PY="${PY:-$(command -v python || command -v python3)}"

# graph-decode hyper-parameters, shared by the decode steps
DECODE_ARGS="--gamma 0.02 --gamma_qphase 0.12 --alpha 0.8 --tau 0.32 --relation_cap 2.0 \
  --w_dist 1.3 --w_pos 0.9 --w_orientation 0.6 --w_behavior 0.8 --w_gaze 1.0 \
  --temporal --temporal_beta 0.8 --temporal_stay 0.2 \
  --temporal_categories pedestrian_behavior,pedestrian_orientation,pedestrian_position,pedestrian_gaze"

# Llama-3.2-1B and embeddinggemma-300m are both gated on HuggingFace. The caption stage uses
# neither, so its step sets TJ_NEED_HF_TOKEN=0 and runs without a token.
if [ "${TJ_NEED_HF_TOKEN:-1}" = "1" ]; then
  [ -n "${HF_TOKEN:-}" ] || {
    echo "ERROR: export HF_TOKEN (needs meta-llama/Llama-3.2-1B and google/embeddinggemma-300m access)"
    exit 1; }
fi

mkdir -p "$SUBS" "$(dirname "$RUNDIR")"
