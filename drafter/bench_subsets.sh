#!/usr/bin/env bash
# bench_subsets.sh — RedHatAI 9-subset acceptance profile for a drafter (or
# native MTP baseline), temp 0, one server load.
#
#   bash drafter/bench_subsets.sh <mtp|drafter_path> <name> [port]
#
# Serves the stack once, then drives each subset's prompts sequentially
# (conc 8, max 250 tok) and computes per-subset interval-weighted τ from the
# vLLM "Mean acceptance length" log window for that subset. Emits a markdown
# table (stdout + runs/<name>/summary.md) and per-subset CSV rows.
#
# Env:
#   TEMP        sampling temperature (default 0 — the historical profile's
#               regime; category ranking is the signal, not absolute τ)
#   SPEC        drafter speculative tokens (default 15) / MTP k (default 3)
#   SUBSET_N    cap prompts per subset (default: all)
#   DRAFTER/verifier serve params: VERIFIER_PATH, QUANT, EXTRA_SERVE_ARGS as
#   in eval_ladder.sh / eval_mtp.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/lib/runlib.sh"

DRAFTER=$1; NAME=$2; PORT=${3:-8300}
TEMP=${TEMP:-0}
SPEC=${SPEC:-$([[ $DRAFTER == mtp ]] && echo 3 || echo 15)}
SUBSET_N=${SUBSET_N:-0}
VENV=${VENV:-$ROOT/.venv-dflash/bin}
QUANT=${QUANT:-nchapman/Qwen3.5-4B-W4A16-GPTQ}
VERIFIER_PATH=${VERIFIER_PATH:-$QUANT}   # staged configs may carry stale pod-era absolute paths

RUN_CMD="bench_subsets.sh $* (TEMP=$TEMP SPEC=$SPEC SUBSET_N=$SUBSET_N)" \
  run_init "bench-$NAME"

RESULTS_CSV=${RESULTS_CSV:-$RUN_DIR/results.csv}
OUT="$RUN_DIR/bench.out"; : > "$OUT"

# resolve the benchmark into the standard HF cache; download AT the pinned
# revision (the pin must constrain the bytes, not just be logged)
STAGE_OUT="$RUN_DIR/staging.txt"
"$VENV/python" - > "$STAGE_OUT" << 'PY' || finish_fail "stage=dataset"
import sys
from huggingface_hub import hf_hub_download, HfApi
sha = HfApi().dataset_info("RedHatAI/speculator_benchmarks").sha
print(f"PIN {sha}")
for s in ["qa","HumanEval","math_reasoning","tool_call","rag","question",
          "writing","summarization","translation"]:
    print(hf_hub_download("RedHatAI/speculator_benchmarks", f"{s}.jsonl",
                          repo_type="dataset", revision=sha))
PY
mapfile -t SUBSET_FILES < "$STAGE_OUT"
(( ${#SUBSET_FILES[@]} == 10 )) \
  || finish_fail "stage=dataset lines=${#SUBSET_FILES[@]} (need 10: PIN + 9 subsets)"
PIN=${SUBSET_FILES[0]#PIN }
NSUB=$(( ${#SUBSET_FILES[@]} - 1 ))
log_event DATASET_PIN "revision=$PIN subsets=$NSUB sub_n=$SUBSET_N"

# ---- serve once ----
if [[ $DRAFTER == mtp ]]; then
  launch_server "$RUN_DIR/server.log" env PATH="$VENV:/usr/local/cuda/bin:$PATH" \
    "$VENV/vllm" serve "$QUANT" --port "$PORT" --max-model-len 8192 \
    --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $SPEC}" \
    ${EXTRA_SERVE_ARGS:-} || finish_fail "stage=server-launch"
else
  [[ -f "$DRAFTER/model.safetensors" && -f "$DRAFTER/config.json" ]] \
    || finish_fail "stage=preflight drafter=$DRAFTER"
  EVAL_DIR="$RUN_DIR/serve"; mkdir -p "$EVAL_DIR"
  cp "$DRAFTER"/config.json "$DRAFTER"/model.safetensors "$EVAL_DIR"/
  cp "$DRAFTER"/config.py "$EVAL_DIR"/ 2>/dev/null || true
  "$VENV/python" - "$EVAL_DIR/config.json" "$SPEC" "$VERIFIER_PATH" << 'PYCFG'
import json, sys
p, spec, verifier = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(p))
c["speculators_config"]["verifier"]["name_or_path"] = verifier
for m in c["speculators_config"]["proposal_methods"]:
    if m.get("speculative_tokens"):
        m["speculative_tokens"] = int(spec)
json.dump(c, open(p, "w"), indent=1)
PYCFG
  launch_server "$RUN_DIR/server.log" env PATH="$VENV:/usr/local/cuda/bin:$PATH" \
    "$VENV/vllm" serve "$EVAL_DIR" --port "$PORT" --max-model-len 8192 \
    ${EXTRA_SERVE_ARGS:-} || finish_fail "stage=server-launch"
fi
wait_healthy "$PORT" || finish_fail "stage=health"
LOG="$RUN_DIR/server.log"

mkdir -p "$(dirname "$RESULTS_CSV")"
: > "$RESULTS_CSV"
echo "subset,tau,accepted" >> "$RESULTS_CSV"

# ---- per-subset: drive prompts, flush, parse the log window ----
for f in "${SUBSET_FILES[@]:1}"; do
  s=$(basename "$f" .jsonl)
  off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  log_event SUBSET_START "subset=$s file=$f"
  "$VENV/python" - "$PORT" "$f" "$TEMP" "$SUBSET_N" >> "$OUT" 2>&1 << 'PY' || finish_fail "stage=prompts subset=$s"
import concurrent.futures, json, sys, urllib.request
port, f, temp, cap = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
rows = [json.loads(l) for l in open(f)]
if cap: rows = rows[:cap]
def hit(r):
    body = json.dumps({"messages": [{"role": "user", "content": r["prompt"]}],
                       "max_tokens": 250, "temperature": temp,
                       "top_p": 0.95, "top_k": 20}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return 1 if json.load(r)["choices"][0]["message"].get("content") else 0
with concurrent.futures.ThreadPoolExecutor(8) as ex:
    print(f"n={len(rows)} nonempty={sum(ex.map(hit, rows))}")
PY
  # metric flush: log quiet for two consecutive 5s probes (max 60s)
  _prev=-1; _quiet=0; _t0=$SECONDS
  while (( SECONDS - _t0 < 60 )); do
    _n=$(wc -l < "$LOG" 2>/dev/null || echo 0)
    if [[ $_n == $_prev ]]; then _quiet=$((_quiet+1)); else _quiet=0; fi
    (( _quiet >= 2 )) && break
    _prev=$_n; sleep 5
  done
  "$VENV/python" - "$LOG" "$off" "$s" "$RESULTS_CSV" >> "$OUT" 2>&1 << 'PYTAU' || finish_fail "stage=metric subset=$s"
import csv, re, sys
log, off, s, csv_path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
pat = re.compile(r"Mean acceptance length: ([0-9.]+),.*Accepted: (\d+) tokens")
a = t = 0
with open(log, "rb") as f:
    f.seek(off)   # byte offset: immune to CR-rich logs vs universal-newline counts
    for raw in f:
        m = pat.search(raw.decode(errors="ignore"))
        if m: a += float(m.group(1)) * int(m.group(2)); t += int(m.group(2))
tau = a / t if t else 0.0
if t == 0:
    sys.exit(f"no acceptance lines in log window — flush missed or format drift: subset={s}")
print(f"SUBSET {s} {tau:.3f} accepted={t}")
with open(csv_path, "a", newline="") as f:
    csv.writer(f).writerow([s, f"{tau:.3f}", t])
PYTAU
  log_event SUBSET_DONE "subset=$s $(grep "^SUBSET $s " "$OUT" | tail -1 | cut -d' ' -f3-)"
done

stop_server

# ---- summary ----
{
  echo "## $NSUB-subset profile — $NAME (temp $TEMP, interval-weighted τ)"
  echo
  echo "| subset | τ | accepted tokens |"
  echo "|---|---|---|"
  tail -n +2 "$RESULTS_CSV" | awk -F, '{printf "| %s | **%s** | %s |\n", $1, $2, $3}'
} | tee "$RUN_DIR/summary.md"
w=$(tail -n +2 "$RESULTS_CSV" | awk -F, '$3 > 0 {a+=$2*$3; t+=$3} END {if (t) printf "%.3f", a/t}')
finish_ok "weighted_tau=$w subsets=$NSUB temp=$TEMP pin=$PIN"
