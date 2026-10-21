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
  RUN_FINISHED=0
  trap '_runlib_on_exit' EXIT
}

# log_event <TYPE> [key=value]... — JSON to events.jsonl, human line to stderr.
# Accepts both "k1=v1" "k2=v2" and the legacy packed form "k1=v1 k2=v2" —
# a packed arg is split only when EVERY whitespace token looks like key=value,
# so free-text values (reason=..., args=...) stay one field.
log_event() {
  local type=$1; shift
  python3 - "$RUN_EVENTS" "$type" "$@" <<'PY' >&2
import json, os, re, sys, time
path, etype, kv = sys.argv[1], sys.argv[2], sys.argv[3:]
t = time.time()  # one clock read: gmtime() and the ms field must agree
obj = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t))
       + f".{int(t*1000)%1000:03d}Z", "type": etype}
parts = []
for item in kv:
    toks = item.split()
    if len(toks) > 1 and all(re.match(r"[A-Za-z_][A-Za-z0-9_]*=", s) for s in toks):
        parts.extend(toks)
    else:
        parts.append(item)
for item in parts:
    k, _, v = item.partition("=")
    obj[k] = v
with open(path, "a") as f:
    f.write(json.dumps(obj) + "\n")
print(f"[{obj['ts']}] [{os.environ.get('RUN_NAME', 'run')}] {etype} {' '.join(parts)}", file=sys.stderr)
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
#   Refuses if the port is already serving: a stale leaked server would answer
#   health checks and silently absorb the run's traffic.
launch_server() {
  SERVER_LOG=$1; shift
  local args=("$@") ptc=""
  for ((i=0; i<${#args[@]}; i++)); do
    if [[ ${args[i]} == "--port" ]]; then ptc=${args[i+1]:-}; break; fi
    if [[ ${args[i]} == --port=* ]]; then ptc=${args[i]#--port=}; break; fi
  done
  if [[ -n $ptc ]] && curl -s -m 2 "localhost:$ptc/health" >/dev/null 2>&1; then
    log_event SERVER_PORT_BUSY "port=$ptc"
    echo "launch_server: port $ptc already serving — stop it first (recorded PIDs live in runs/*/events.jsonl)" >&2
    return 1
  fi
  nohup "$@" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  # a killed/interrupted script must not leak the nohup'd server; INT/TERM
  # also EXIT (cleanup alone would let the script continue headless)
  SERVER_STOPPED=0
  trap '_runlib_on_exit' EXIT
  trap '[[ ${SERVER_PID:-} && $SERVER_STOPPED == 0 ]] && stop_server; exit 130' INT
  trap '[[ ${SERVER_PID:-} && $SERVER_STOPPED == 0 ]] && stop_server; exit 143' TERM
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
  SERVER_STOPPED=1   # suppress the EXIT-trap cleanup double-fire
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

RUN_FINISHED=0   # finish_ok/finish_fail set this; the EXIT trap logs any other exit

finish_ok()  { RUN_FINISHED=1; log_event RUN_DONE  "status=ok" "$@"; }
finish_fail() { RUN_FINISHED=1; stop_server 2>/dev/null || true; log_event RUN_FAILED "status=fail" "$@"; exit 1; }

# composed EXIT handler: server cleanup (launch_server arms this too) plus a
# terminal event for deaths that bypass finish_* (set -e kills, unhandled errors)
_runlib_on_exit() {
  local rc=$?
  [[ ${SERVER_PID:-} && ${SERVER_STOPPED:-1} == 0 ]] && stop_server
  if [[ ${RUN_FINISHED:-0} == 0 && $rc -ne 0 ]]; then
    log_event RUN_FAILED "status=fail unexpected_exit=$rc"
  fi
}

# Liveness watchdog: while a long foreground command runs, fail the stage if its
# log stops growing (silent stall — process alive, no progress). Epoch-boundary
# checkpoint/val windows can pause logging for 15-20 min, so the default
# staleness budget is 30 min (LADDER_STALE_MIN overrides).
# GB10 unified memory: a job's model memory is invisible to RSS (GPU-side
# allocations) and therefore to the kernel OOM killer's scoring — under global
# pressure it shoots random userspace (2026-10-17: pipewire/tailscaled died
# mid-quant-run while the 67GB-holder showed rss=21MB). /proc/meminfo DOES
# see unified-memory allocations, so guard on MemAvailable and kill OUR child
# before the kernel OOM fires. Floor leaves the desktop session alive.
mem_guard() {  # mem_guard <pid> <floor_mb>
  local pid=$1 floor=$2 avail
  while kill -0 "$pid" 2>/dev/null; do
    avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 999999)
    if (( avail < floor )); then
      log_event MEMGUARD_KILL "mem_avail_mb=$avail floor_mb=$floor pid=$pid"
      kill -TERM "$pid" 2>/dev/null || true
      sleep 10
      pkill -KILL -P "$pid" 2>/dev/null || true
      kill -KILL "$pid" 2>/dev/null || true
      return 1
    fi
    sleep 5
  done
  return 0
}

run_with_watchdog() {  # run_with_watchdog <logfile> <cmd...>
  local log=$1; shift
  local max_rounds=$(( ${LADDER_STALE_MIN:-30} / 2 ))
  "$@" >> "$log" 2>&1 &
  local cmd_pid=$!
  local guard_pid=""
  if [[ -n ${MEM_GUARD_MB:-} ]]; then
    mem_guard "$cmd_pid" "$MEM_GUARD_MB" &
    guard_pid=$!
  fi
  local last_size=-1 stale_rounds=0
  while kill -0 "$cmd_pid" 2>/dev/null; do
    sleep 120
    kill -0 "$cmd_pid" 2>/dev/null || break
    local size; size=$(stat -c %s "$log" 2>/dev/null || echo 0)
    if (( size == last_size )); then
      stale_rounds=$(( stale_rounds + 1 ))
      if (( stale_rounds >= max_rounds )); then
        log_event WATCHDOG_FAIL "stage_log_stale_min=$(( stale_rounds * 2 )) log=$log pid=$cmd_pid"
        kill -TERM "$cmd_pid" 2>/dev/null || true
        sleep 30
        pkill -KILL -P "$cmd_pid" 2>/dev/null || true
        kill -KILL "$cmd_pid" 2>/dev/null || true
        wait "$cmd_pid" 2>/dev/null
        return 1
      fi
    else
      stale_rounds=0
    fi
    last_size=$size
  done
  [[ -n "$guard_pid" ]] && kill "$guard_pid" 2>/dev/null || true
  wait "$cmd_pid"
}
