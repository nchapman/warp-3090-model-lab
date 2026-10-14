#!/bin/bash
# tail_run.sh — inspect lab runs without guessing where anything went.
# Usage: tail_run.sh [name-prefix]     (no arg = latest run)
#   - cmd.txt: exact invocation
#   - events.jsonl: structured lifecycle (SERVER_HEALTHY / SERVER_STUCK / ...)
#   - server.log tail (if the run had a server)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS_DIR=${RUNS_DIR:-$ROOT/runs}
shopt -s nullglob
if [[ $# -ge 1 ]]; then
  dirs=("$RUNS_DIR/$1".*)
else
  dirs=("$RUNS_DIR"/*)
fi
[[ ${#dirs[@]} -gt 0 ]] || { echo "no runs under $RUNS_DIR"; exit 1; }
latest=$(ls -1d "${dirs[@]}" | sort | tail -1)
echo "== $latest"
[[ -f "$latest/cmd.txt" ]] && cat "$latest/cmd.txt"
echo "-- events:"
cat "$latest/events.jsonl" 2>/dev/null
# server logs live wherever the launching script pointed them; find the path
# from the SERVER_LAUNCH event instead of guessing
slog=$(grep '"type": "SERVER_LAUNCH"' "$latest/events.jsonl" 2>/dev/null | tail -1 \
      | grep -o '"log": "[^"]*"' | cut -d'"' -f4)
if [[ -n ${slog:-} && -f $slog ]]; then
  echo "-- $slog (last 10):"; tail -n 10 "$slog" | cut -c1-160
fi
