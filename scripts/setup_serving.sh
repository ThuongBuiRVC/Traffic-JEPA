#!/usr/bin/env bash
# Build the virtualenv the caption server runs in, at .venv-serving/.
#
#   bash scripts/setup_serving.sh
#
# vllm pins torch==2.11.0 and caps numpy below 2.4, so installing it into the main environment
# would move the pins that produced the top-1 result. It gets its own environment instead. The
# server is a separate process reached over HTTP, so nothing in the repo imports it.
#
# It downloads several GB and takes a while. Delete .venv-serving/ to undo it, the main
# environment is untouched either way.
set -euo pipefail
TJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${TJ_SERVE_VENV:-$TJ_ROOT/.venv-serving}"

if [ -x "$VENV/bin/vllm" ]; then
  echo "== already built: $VENV =="
  "$VENV/bin/vllm" --version 2>/dev/null || true
  exit 0
fi

echo "== building the caption server environment at $VENV =="
python3 -m venv "$VENV" 2>/dev/null || {
  echo "ERROR: python3 -m venv failed. On Debian/Ubuntu: apt install python3-venv"
  exit 1; }

"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$TJ_ROOT/requirements-serving.txt"

[ -x "$VENV/bin/vllm" ] || { echo "ERROR: vllm did not land in $VENV/bin"; exit 1; }
echo "== done, $("$VENV/bin/vllm" --version 2>/dev/null | head -1) =="
echo "   scripts/05_caption.sh picks this up on its own from now on."
