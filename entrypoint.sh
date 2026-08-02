#!/usr/bin/env bash
# Container entrypoint. `verify` (default) checks the shipped checkpoint; `all` retrains
# from raw data. Anything else runs verbatim.
#   docker run --gpus all -e HF_TOKEN=hf_xxx -v /path/wts:/workspace/Traffic-JEPA/data traffic-jepa verify
#   docker run --gpus all -e HF_TOKEN=hf_xxx -v /path/wts:/workspace/Traffic-JEPA/data traffic-jepa all
set -euo pipefail
case "${1:-verify}" in
  verify) exec bash scripts/00_verify_checkpoint.sh ;;
  all)    exec bash scripts/run_all.sh ;;
  *)      exec "$@" ;;
esac
