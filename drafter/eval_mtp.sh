#!/bin/bash
# eval_mtp.sh — native-MTP baseline on the same ladder as eval_ladder.sh.
# Usage: eval_mtp.sh <eval_name> <port> [num_spec_tokens]
# Serves the verifier (QUANT env, default: published 4b cascade quant) with
# its native MTP head via --speculative-config, fires the fixed 20-prompt
# ladder (data/ladder-20.jsonl), computes the INTERVAL-WEIGHTED acceptance —
# identical protocol to eval_ladder.sh so τ columns are comparable.
# Standard logging via lib/runlib.sh (run dir + events + watchdogged health).
# Env: VENV (drafter venv bin, default: repo .venv-dflash/bin)
#      QUANT (verifier to serve, default: published 4b cascade quant)
#      LADDER (prompts jsonl, default: data/ladder-20.jsonl)
#      EXTRA_SERVE_ARGS (e.g. "--enforce-eager" — MTP + cudagraph capture has
#        wedged on GB10 before; eager dodges it at some throughput cost)
#      EVAL_DIR / LOG / OUT / RESULTS_CSV (working paths)
# Standard HF cache only — no custom VLLM_CACHE_ROOT.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/lib/runlib.sh"
NAME=$1; PORT=$2; SPEC=${3:-3}
[[ $SPEC =~ ^[0-9]+$ && $SPEC -ge 1 && $SPEC -le 8 ]] \
  || { echo "MTP k must be 1-8, got: $SPEC" >&2; exit 1; }
EVAL_DIR=${EVAL_DIR:-/tmp/eval-mtp-$NAME}
LOG=${LOG:-/tmp/eval-mtp-$NAME.log}
OUT=${OUT:-/tmp/eval-mtp-$NAME.out}
RESULTS_CSV=${RESULTS_CSV:-$ROOT/results/results.csv}
VENV=${VENV:-$ROOT/.venv-dflash/bin}
QUANT=${QUANT:-nchapman/Qwen3.5-4B-W4A16-GPTQ}
LADDER=${LADDER:-$ROOT/data/ladder-20.jsonl}

RUN_CMD="eval_mtp.sh $*" run_init "mtp-$NAME"
require_file "$LADDER"
rm -rf "$EVAL_DIR"; mkdir -p "$EVAL_DIR" "$(dirname "$RESULTS_CSV")"
: > "$OUT"   # per-run artifact: truncate so the success gate can't see a stale TAU

launch_server "$LOG" env PATH="$VENV:/usr/local/cuda/bin:$PATH" \
  "$VENV/vllm" serve "$QUANT" --port "$PORT" --max-model-len 8192 --gpu-memory-utilization "${GPU_MEM_UTIL:-0.6}" \
  --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $SPEC}" \
  ${EXTRA_SERVE_ARGS:-} \
  || finish_fail "stage=server-launch"
wait_healthy "$PORT" || finish_fail "stage=health"

# identical ladder protocol to eval_ladder.sh: temp 1.0 / top_p 0.95 / top_k 20,
# max 250 tok, conc 8
TEMP=${TEMP:-1.0}
log_event PROMPTS_START "n=20 temp=$TEMP"
"$VENV/python" - "$PORT" "$LADDER" "$TEMP" << 'PY' >> "$OUT" 2>&1
import concurrent.futures, json, sys, urllib.request
port, ladder, temp = sys.argv[1], sys.argv[2], float(sys.argv[3])
prompts = [json.loads(l) for l in open(ladder)]
def hit(p):
    body = json.dumps({"messages": p["messages"], "max_tokens": 250,
                       "temperature": temp, "top_p": 0.95, "top_k": 20}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"].get("content", "")[:40]
with concurrent.futures.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(hit, prompts))
print("n =", len(res), "nonempty =", sum(1 for t in res if t))
PY
log_event PROMPTS_DONE

# wait for metric flush: server log quiet for two consecutive 5s probes (max 60s)
_prev=-1; _quiet=0; _t0=$SECONDS
while (( SECONDS - _t0 < 60 )); do
  _n=$(wc -l < "$LOG" 2>/dev/null || echo 0)
  if [[ $_n == $_prev ]]; then _quiet=$((_quiet+1)); else _quiet=0; fi
  (( _quiet >= 2 )) && break
  _prev=$_n; sleep 5
done
"$VENV/python" - "$LOG" "$NAME" "$RESULTS_CSV" "mtp-k$SPEC" "$TEMP" << 'PYMETRIC' >> "$OUT" 2>&1
import csv, re, sys
log, name, csv_path, spec, temp = sys.argv[1:6]
temp = float(temp)
pat = re.compile(r"Mean acceptance length: ([0-9.]+),.*Accepted: (\d+) tokens")
a = t = 0
for line in open(log, errors="ignore"):
    m = pat.search(line)
    if m:
        acc, tok = float(m.group(1)), int(m.group(2))
        a += acc * tok; t += tok
tau = a / t if t else 0.0
if t == 0:
    sys.exit(f"no acceptance lines in server log — flush missed or format drift: name={name}")
print(f"TAU {name} {tau:.3f} accepted={t}")
row = [name, f"spec={spec}", str(t), f"{tau:.3f}", f"20prompt temp{temp} interval-weighted"]
new = True
try:
    rows = list(csv.reader(open(csv_path)))
    new = not any(r and r[0] == name and len(r) > 4 and re.search(r"temp([0-9.]+)", r[4]) and float(re.search(r"temp([0-9.]+)", r[4]).group(1)) == temp for r in rows)
except FileNotFoundError:
    rows = []
if new:
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)
PYMETRIC
stop_server
tau_line=$(grep "^TAU " "$OUT" | tail -1)
if [[ -n $tau_line ]]; then
  finish_ok "tau=${tau_line#TAU }"
else
  finish_fail "stage=metric"
fi
