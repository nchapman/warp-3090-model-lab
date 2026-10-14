#!/bin/bash
# eval_ladder.sh — the standard drafter evaluation.
# Usage: eval_ladder.sh <draft_dir> <eval_name> <spec_tokens> <port>
#   spec_tokens: e.g. 15 for b16-native, 7 for b8 serve-down, or "native"
#   draft_dir must contain config.json + model.safetensors (config.py optional)
# Serves the drafter + verifier in vLLM, fires the fixed 20-prompt ladder
# (data/ladder-20.jsonl), computes the INTERVAL-WEIGHTED acceptance (the
# standard — last-line reads are banned, they are straggler-skewed), appends
# a row to $RESULTS_CSV.
# Standard logging via lib/runlib.sh: run dir + events.jsonl + watchdogged
# server health (see README "Run logging standard").
# Env: VENV (drafter venv bin, default: repo .venv-dflash/bin)
#      QUANT / QUANT_PATH (verifier, default: published 4b cascade quant)
#      VERIFIER_PATH (repoint the config's verifier before serving)
#      LADDER (prompts jsonl, default: data/ladder-20.jsonl)
#      EXTRA_SERVE_ARGS (e.g. "--enforce-eager" to dodge cudagraph hangs)
#      EVAL_DIR / LOG / OUT / RESULTS_CSV (working paths)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/lib/runlib.sh"
SRC=$1; NAME=$2; SPEC=$3; PORT=$4
EVAL_DIR=${EVAL_DIR:-$ROOT/eval_$NAME}
LOG=${LOG:-$ROOT/vllm_eval_${NAME}.log}
OUT=${OUT:-$ROOT/eval_${NAME}.out}
RESULTS_CSV=${RESULTS_CSV:-$ROOT/results/results.csv}
VENV=${VENV:-$ROOT/.venv-dflash/bin}
QUANT=${QUANT_PATH:-${QUANT:-nchapman/Qwen3.5-4B-W4A16-GPTQ}}
LADDER=${LADDER:-$ROOT/data/ladder-20.jsonl}

RUN_CMD="eval_ladder.sh $*" run_init "ladder-$NAME"

# preflight: fail on wrong-shaped inputs BEFORE touching the GPU
require_file "$SRC/config.json" "$SRC/model.safetensors" "$LADDER"

rm -rf "$EVAL_DIR"; mkdir -p "$EVAL_DIR" "$(dirname "$RESULTS_CSV")"
: > "$OUT"   # per-run artifact: truncate so the success gate can't see a stale TAU
cp "$SRC"/config.json "$SRC"/model.safetensors "$EVAL_DIR"/
cp "$SRC"/config.py "$EVAL_DIR"/ 2>/dev/null || true
log_event FILES_STAGED "eval_dir=$EVAL_DIR"

"$VENV/python" - "$EVAL_DIR/config.json" "$SPEC" "${VERIFIER_PATH:-}" "$QUANT" << 'PYCFG'
import json, sys, os
p, spec = sys.argv[1], sys.argv[2]
c = json.load(open(p))
if os.environ.get("VERIFIER_PATH"):
    c["speculators_config"]["verifier"]["name_or_path"] = os.environ["VERIFIER_PATH"]
if spec != "native":
    for m in c["speculators_config"]["proposal_methods"]:
        if m.get("speculative_tokens"):
            m["speculative_tokens"] = int(spec)
json.dump(c, open(p, "w"), indent=1)
PYCFG

launch_server "$LOG" env PATH="$VENV:/usr/local/cuda/bin:$PATH" \
  "$VENV/vllm" serve "$EVAL_DIR" --port "$PORT" --max-model-len 8192 \
  ${EXTRA_SERVE_ARGS:-} \
  || finish_fail "stage=server-launch"
wait_healthy "$PORT" || finish_fail "stage=health"

# DFlash2 convention: temp 1.0 / top_p 0.95 / top_k 20, max 250 tok, conc 8.
# TEMP selects the regime: 0.7 = serving point (checkpoint SELECTION happens
# here — see AGENTS.md); 0 = greedy, the low-variance stability instrument
# (repeat spread ~0.05); 1.0 = heavy-tailed (0.34 spread on identical inputs)
# — never rank checkpoints on small n at t>=0.7 without repeats.
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
"$VENV/python" - "$LOG" "$NAME" "$RESULTS_CSV" "$SPEC" "$TEMP" << 'PYMETRIC' >> "$OUT" 2>&1
import csv, re, sys
log, name, csv_path, spec, temp = sys.argv[1:6]
pat = re.compile(r"Mean acceptance length: ([0-9.]+),.*Accepted: (\d+) tokens")
a = t = 0
for line in open(log, errors="ignore"):
    m = pat.search(line)
    if m:
        acc, tok = float(m.group(1)), int(m.group(2))
        a += acc * tok; t += tok
tau = a / t if t else 0.0
print(f"TAU {name} {tau:.3f} accepted={t}")
row = [name, f"spec={spec}", str(t), f"{tau:.3f}", f"20prompt temp{temp} interval-weighted"]
new = True
try:
    rows = list(csv.reader(open(csv_path)))
    new = not any(r and r[0] == name and len(r) > 4 and f"temp{temp}" in r[4] for r in rows)
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
