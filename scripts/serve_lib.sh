#!/usr/bin/env bash
# Bring up the caption server on demand, so the caption step never falls back to the slow
# in-process path by accident. Sourced by 05_caption.sh.
#
#   CAPTION_SERVER=...      already pointed at an endpoint -> use it, start nothing
#   CAPTION_NO_SERVE=1      force the in-process path
#   CAPTION_KEEP_SERVER=1   leave the server up after the step, for repeated runs

_SERVE_PID=""

serve_stop() {
  [ -n "$_SERVE_PID" ] || return 0
  [ "${CAPTION_KEEP_SERVER:-0}" = "1" ] && {
    echo "== caption server left up on :${_SERVE_PORT} (pid $_SERVE_PID) =="
    return 0; }
  echo "== stopping the caption server =="
  kill "$_SERVE_PID" 2>/dev/null || true
  wait "$_SERVE_PID" 2>/dev/null || true
  _SERVE_PID=""
}

_serve_up() {  # is a server answering on $1
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
  else
    command -v vllm >/dev/null || {
      echo "== vllm is not installed, captioning in-process: one segment at a time, hours not minutes"
      echo "   pip install vllm   (or set CAPTION_NO_SERVE=1 to silence this)"
      return 0; }

    local log="${TMPDIR:-/tmp}/traffic_jepa_vllm_$port.log"
    echo "== starting the caption server on :$port, takes a few minutes to load =="
    echo "   log: $log"
    PORT="$port" bash "$TJ_ROOT/serving/start.sh" >"$log" 2>&1 &
    _SERVE_PID=$!

    local waited=0
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
  fi

  export CAPTION_SERVER="http://localhost:$port/v1"
  export CAPTION_WORKERS="${CAPTION_WORKERS:-8}"
}
