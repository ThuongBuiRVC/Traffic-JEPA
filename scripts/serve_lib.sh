#!/usr/bin/env bash
# Bring up the caption server on demand, so the caption step never falls back to the slow
# in-process path by accident. Sourced by 05_caption.sh and by serving/start.sh.
#
# The server runs from its own virtualenv (.venv-serving), because vllm pins torch==2.11.0 and
# the main environment is pinned at 2.12.0. They never have to agree: the server is a separate
# process and the caption step talks to it over HTTP.
#
#   CAPTION_SERVER=...      already pointed at an endpoint -> use it, start nothing
#   CAPTION_NO_SERVE=1      force the in-process path, no server
#   CAPTION_KEEP_SERVER=1   leave the server up after the step, for repeated runs
#   TJ_SERVE_VENV=...       somewhere other than .venv-serving
#   TJ_NO_AUTO_SETUP=1      never build the virtualenv, only report that it is missing

_SERVE_PID=""

# path to the vllm binary, serving virtualenv first, then whatever is on PATH
vllm_bin() {
  local venv="${TJ_SERVE_VENV:-$TJ_ROOT/.venv-serving}"
  [ -x "$venv/bin/vllm" ] && { echo "$venv/bin/vllm"; return 0; }
  command -v vllm 2>/dev/null && return 0
  return 1
}

serve_stop() {
  [ -n "$_SERVE_PID" ] || return 0
  [ "${CAPTION_KEEP_SERVER:-0}" = "1" ] && {
    echo "== caption server left up on :$_SERVE_PORT (pid $_SERVE_PID) =="
    return 0; }
  echo "== stopping the caption server =="
  kill "$_SERVE_PID" 2>/dev/null || true
  wait "$_SERVE_PID" 2>/dev/null || true
  _SERVE_PID=""
}

_serve_up() {  # is a server answering on port $1
  curl -sf --max-time 3 "http://localhost:$1/v1/models" \
    -H "Authorization: Bearer not-needed" >/dev/null 2>&1
}

serve_auto() {
  [ -n "${CAPTION_SERVER:-}" ] && { echo "== captioning through $CAPTION_SERVER =="; return 0; }
  [ "${CAPTION_NO_SERVE:-0}" = "1" ] && return 0

  local port="${PORT:-8100}"
  _SERVE_PORT="$port"

  if _serve_up "$port"; then
    echo "== reusing the caption server already on :$port =="
    export CAPTION_SERVER="http://localhost:$port/v1"
    export CAPTION_WORKERS="${CAPTION_WORKERS:-8}"
    return 0
  fi

  if ! vllm_bin >/dev/null; then
    if [ "${TJ_NO_AUTO_SETUP:-0}" = "1" ]; then
      echo "== no caption server environment, captioning in-process: hours, not minutes"
      echo "   build it with: bash scripts/setup_serving.sh"
      return 0
    fi
    echo "== no caption server environment yet, building it once at ${TJ_SERVE_VENV:-$TJ_ROOT/.venv-serving}"
    echo "   several GB, and it leaves the main environment alone. Ctrl-C to skip."
    bash "$TJ_ROOT/scripts/setup_serving.sh" || {
      echo "== could not build it, captioning in-process: hours, not minutes"
      return 0; }
  fi

  local bin log waited=0
  bin="$(vllm_bin)"
  log="${TMPDIR:-/tmp}/traffic_jepa_vllm_$port.log"
  echo "== starting the caption server on :$port, takes a few minutes to load =="
  echo "   log: $log"
  VLLM_BIN="$bin" PORT="$port" bash "$TJ_ROOT/serving/start.sh" >"$log" 2>&1 &
  _SERVE_PID=$!

  until _serve_up "$port"; do
    kill -0 "$_SERVE_PID" 2>/dev/null || {
      echo "ERROR: the caption server exited before it was ready. Last lines of $log:"
      tail -n 25 "$log"
      _SERVE_PID=""
      exit 1; }
    [ "$waited" -ge 900 ] && {
      echo "ERROR: the caption server was not ready after 15 minutes, see $log"
      serve_stop
      exit 1; }
    sleep 5; waited=$((waited + 5))
  done
  echo "== caption server ready after ${waited}s =="

  export CAPTION_SERVER="http://localhost:$port/v1"
  export CAPTION_WORKERS="${CAPTION_WORKERS:-8}"
}
