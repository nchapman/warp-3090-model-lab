#!/bin/bash
# eval_ladder.sh — the standard drafter evaluation.
# Usage: eval_ladder.sh <draft_dir> <eval_name> <spec_tokens> <port>
#   spec_tokens: e.g. 15 for b16-native, 7 for b8 serve-down, or "native"
# Serves the drafter + verifier in vLLM, fires the fixed 20-prompt ladder
# (data/ladder-20.jsonl), computes the INTERVAL-WEIGHTED acceptance (the
# standard — last-line reads are banned, they are straggler-skewed), appends
# a row to $RESULTS_CSV.
# Env: VENV (drafter venv bin, default: repo .venv-dflash/bin)
#      QUANT (verifier, default: published 4b cascade quant)
#      VERIFIER_PATH (repoint the config's verifier before serving)
#      LADDER (prompts jsonl, default: data/ladder-20.jsonl)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC=$1; NAME=$2; SPEC=$3; PORT=$4
EVAL_DIR=${EVAL_DIR:-$ROOT/eval_$NAME}
LOG=${LOG:-$ROOT/vllm_eval_${NAME}.log}
OUT=$ROOT/eval_${NAME}.out
RESULTS_CSV=${RESULTS_CSV:-$ROOT/results/results.csv}
VENV=${VENV:-$ROOT/.venv-dflash/bin}
QUANT=${QUANT_PATH:-${QUANT:-nchapman/Qwen3.5-4B-W4A16-GPTQ}}
LADDER=${LADDER:-$ROOT/data/ladder-20.jsonl}

rm -rf "$EVAL_DIR"; mkdir -p "$EVAL_DIR" "$(dirname "$RESULTS_CSV")"
cp "$SRC"/config.json "$SRC"/model.safetensors "$EVAL_DIR"/
cp "$SRC"/config.py "$EVAL_DIR"/ 2>/dev/null || true

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

nohup env PATH="$VENV:/usr/local/cuda/bin:$PATH" \
  "$VENV/vllm" serve "$EVAL_DIR" --port "$PORT" --max-model-len 8192 \
  > "$LOG" 2>&1 &

for i in $(seq 1 60); do
  sleep 15
  curl -s -m 3 "localhost:$PORT/health" >/dev/null 2>&1 && break
done
if ! curl -s -m 3 "localhost:$PORT/health" >/dev/null 2>&1; then
  echo "SERVER_FAILED $NAME" > "$OUT"; tail -n 5 "$LOG" >> "$OUT"; exit 1
fi

# DFlash2 convention: temp 1.0 / top_p 0.95 / top_k 20, max 250 tok, conc 8
"$VENV/python" - "$PORT" "$LADDER" << 'PY' >> "$OUT" 2>&1
import concurrent.futures, json, sys, urllib.request
port, ladder = sys.argv[1], sys.argv[2]
prompts = [json.loads(l) for l in open(ladder)]
def hit(p):
    body = json.dumps({"messages": p["messages"], "max_tokens": 250,
                       "temperature": 1.0, "top_p": 0.95, "top_k": 20}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"].get("content", "")[:40]
with concurrent.futures.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(hit, prompts))
print("n =", len(res), "nonempty =", sum(1 for t in res if t))
PY

sleep 15
"$VENV/python" - "$LOG" "$NAME" "$RESULTS_CSV" "$SPEC" << 'PYMETRIC' >> "$OUT" 2>&1
import csv, re, sys
log, name, csv_path, spec = sys.argv[1:5]
pat = re.compile(r"Mean acceptance length: ([0-9.]+),.*Accepted: (\d+) tokens")
a = t = 0
for line in open(log, errors="ignore"):
    m = pat.search(line)
    if m:
        acc, tok = float(m.group(1)), int(m.group(2))
        a += acc * tok; t += tok
tau = a / t if t else 0.0
print(f"TAU {name} {tau:.3f} accepted={t}")
row = [name, f"spec={spec}", str(t), f"{tau:.3f}", "20prompt temp1.0 interval-weighted"]
new = True
try:
    rows = list(csv.reader(open(csv_path)))
    new = not any(r and r[0] == name for r in rows)
except FileNotFoundError:
    rows = []
if new:
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)
PYMETRIC

pkill -f "vllm ser[v]e"
echo "EVAL_ONE_DONE $NAME" >> "$OUT"
