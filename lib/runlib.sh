#!/bin/bash
# lib/runlib.sh — standard run lifecycle, logging, and server health.
#
# Source from any lab script:
#   source "$(dirname "$0")/../lib/runlib.sh"
#
# Contract (README "Run logging standard" is the user-facing doc):
#   - every run gets runs/<name>.<UTC ts>/ with cmd.txt + events.jsonl
#   - events are structured (one JSON object per line), also echoed to stderr
#   - inputs are validated by name BEFORE anything launches (require_file)
#   - servers are launched with a recorded PID and die by that PID — never
#     pkill-scatter, which has killed the wrong shell twice in this project
#   - health waits are watchdogged: log idle + GPU 0% for two consecutive
#     probes => SERVER_STUCK with a process snapshot, kill, loud failure

RUNLIB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR=${RUNS_DIR:-$RUNLIB_ROOT/runs}

# run_init <name>  — call FIRST, before any output paths are used.
run_init() {
  RUN_NAME=${1:?run_init needs a name}
  export RUN_NAME
  RUN_DIR="$RUNS_DIR/$RUN_NAME.$(date -u +%Y%m%dT%H%M%SZ).$$"
  mkdir -p "$RUN_DIR"
  RUN_EVENTS="$RUN_DIR/events.jsonl"
  {
    echo "# utc: $(date -u +%FT%TZ)"
    echo "# host: $(hostname)"
    echo "# cmd: ${RUN_CMD:-$0 $(printf '%q ' "$@")}"
  } > "$RUN_DIR/cmd.txt"
  log_event RUN_START "name=$RUN_NAME" "dir=$RUN_DIR"
}

# log_event <TYPE> [key=value]... — JSON to events.jsonl, human line to stderr.
log_event() {
  local type=$1; shift
  python3 - "$RUN_EVENTS" "$type" "$@" <<'PY' >&2
import json, os, sys, time
path, etype, kv = sys.argv[1], sys.argv[2], sys.argv[3:]
obj = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
       + f".{int(time.time()*1000)%1000:03d}Z", "type": etype}
for item in kv:
    k, _, v = item.partition("=")
    obj[k] = v
with open(path, "a") as f:
    f.write(json.dumps(obj) + "\n")
print(f"[{obj['ts']}] [{os.environ.get('RUN_NAME', 'run')}] {etype} {' '.join(kv)}", file=sys.stderr)
PY
}

# require_file <path>... — preflight: every path must exist and be readable.
require_file() {
  local f
  for f in "$@"; do
    if [[ ! -r $f ]]; then
      log_event PREFLIGHT_FAIL "path=$f"
      echo "PREFLIGHT FAIL: missing or unreadable: $f" >&2
      exit 1
    fi
  done
  log_event PREFLIGHT_OK "files=$#"
}

# launch_server <logfile> <cmd...> — nohup, record PID, log it.
launch_server() {
  SERVER_LOG=$1; shift
  nohup "$@" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  log_event SERVER_LAUNCH "pid=$SERVER_PID" "log=$SERVER_LOG"
}

# wait_healthy <port> [max_wait_s=900] [stuck_s=180]
#   Polls /health every 5s. Fails fast on: timeout (SERVER_FAILED + log tail)
#   or stuck watchdog (SERVER_STUCK: log idle > stuck_s AND GPU util 0% on
#   two consecutive probes — a compiling/loading server keeps one of them busy).
wait_healthy() {
  local port=$1 max=${2:-900} stuck=${3:-180}
  local start=$SECONDS idle=0
  while :; do
    if curl -s -m 3 "localhost:$port/health" >/dev/null 2>&1; then
      log_event SERVER_HEALTHY "elapsed=$((SECONDS-start))s" "port=$port"
      return 0
    fi
    local elapsed=$((SECONDS-start))
    if (( elapsed >= max )); then
      log_event SERVER_FAILED "waited=${elapsed}s" "port=$port"
      tail -n 5 "$SERVER_LOG" >&2
      stop_server
      return 1
    fi
    local mtime now gpu
    mtime=$(stat -c %Y "$SERVER_LOG" 2>/dev/null || echo 0)
    now=$(date +%s)
    if (( now - mtime > stuck )); then
      gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
      if [[ ${gpu:-1} == 0 ]]; then
        idle=$((idle+1))
        if (( idle >= 2 )); then
          log_event SERVER_STUCK "log_idle=$((now-mtime))s" "gpu_util=${gpu}pct" "waited=${elapsed}s"
          server_snapshot >&2
          stop_server
          return 1
        fi
      else
        idle=0
      fi
    fi
    sleep 5
  done
}

# server_snapshot — process tree state for diagnosis.
server_snapshot() {
  [[ -n ${SERVER_PID:-} ]] || return 0
  ps -o pid,ppid,etime,pcpu,stat,comm -p "$SERVER_PID" $(pgrep -P "$SERVER_PID" 2>/dev/null | tr '\n' ' ') 2>/dev/null
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1
}

# stop_server — TERM, up to 20s grace, then KILL; children too. By PID only.
stop_server() {
  [[ -n ${SERVER_PID:-} ]] || return 0
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    SERVER_PID=""; return 0   # already gone — idempotent
  fi
  # TERM the parent AND its children (contract: servers die by PID, children too)
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  pkill -TERM -P "$SERVER_PID" 2>/dev/null || true
  local i
  for i in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    pkill -KILL -P "$SERVER_PID" 2>/dev/null || true
    log_event SERVER_STOPPED "pid=$SERVER_PID" "signal=KILL"
  else
    log_event SERVER_STOPPED "pid=$SERVER_PID" "signal=TERM"
  fi
}

finish_ok()  { log_event RUN_DONE  "status=ok" "$@"; }
finish_fail() { log_event RUN_FAILED "status=fail" "$@"; exit 1; }
