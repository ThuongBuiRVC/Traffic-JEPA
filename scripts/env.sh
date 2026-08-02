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
export WOVEN="${WOVEN:-$DATA/woven-traffic/internal}"        # real WTS raw: video/ caption/
export TEST_VIDEOS="${TEST_VIDEOS:-$DATA/raw/public_test_videos}"
export TEST_BBOX="${TEST_BBOX:-$DATA/raw/public_test_annotations}"

# --- produced artifacts (all overridable so a clean run can use a separate workspace) ---
export SIMQA="${TJ_SIMQA:-$DATA/processed/sim_qa_vljepa16}"    # cut clips + sim manifests
export REALQA="${TJ_REALQA:-$DATA/processed/real_qa_vljepa16}" # cut clips + real manifests
export CACHE="${TJ_CACHE:-$DATA/processed/cache_vljepa16_8f}"  # V-JEPA latents + Gemma vecs + index
export RUNDIR="${TJ_RUNDIR:-$TJ_ROOT/runs/traffic_jepa_world_model}"
export SUBS="${TJ_SUBS:-$TJ_ROOT/submissions}"

# --- python / package ---
export PYTHONPATH="$TJ_ROOT:${PYTHONPATH:-}"
export VLJEPA_DATA="$DATA"          # preprocess reads manifests from $DATA/processed/*_qa_vljepa16/_review
export VLJEPA_ROOT="$DATA"          # clip_16 paths are absolute, so this is only a harmless base
export VLJEPA_CACHE="$CACHE"
export VLJEPA_OUT="$CACHE"
PY="${PY:-python}"

# graph-decode hyper-parameters, shared by the decode steps
DECODE_ARGS="--gamma 0.02 --gamma_qphase 0.12 --alpha 0.8 --tau 0.32 --relation_cap 2.0 \
  --w_dist 1.3 --w_pos 0.9 --w_orientation 0.6 --w_behavior 0.8 --w_gaze 1.0 \
  --temporal --temporal_beta 0.8 --temporal_stay 0.2 \
  --temporal_categories pedestrian_behavior,pedestrian_orientation,pedestrian_position,pedestrian_gaze"

# Llama-3.2-1B is gated on HuggingFace.
[ -n "${HF_TOKEN:-}" ] || { echo "ERROR: export HF_TOKEN (needs meta-llama/Llama-3.2-1B access)"; exit 1; }

mkdir -p "$SUBS" "$(dirname "$RUNDIR")"
