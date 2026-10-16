#!/usr/bin/env bash
# regen_pool.sh <model-key> [port] — the standing per-model-size regen pass.
#
# Doctrine: the context pool is the shared invariant; every target model
# regenerates the final assistant turn ON-POLICY with its DEPLOYED QUANT
# (drafter must match its target). One pass per model size: 0.8b pilot now,
# 4b corpus recovery when 4B ladder work resumes, 9b/27b later.
#
# Serving: MTP speculative decoding ON by default (MTP_K, default 3) —
# spec-decode is lossless, so the corpus distribution is identical, only
# wall-clock improves. MTP+cudagraph capture has wedged on GB10 before:
# if startup hangs, EXTRA_SERVE_ARGS="--enforce-eager". MTP_K=0 disables.
# MEMORY: GB10 is UNIFIED (121 GiB shared CPU+GPU) — vLLM's default 0.9
# gpu-mem-util reserves ~109 GiB and OOM-kills the whole userspace once
# client load arrives (2026-10-16: pipewire/llama-swap/session died at
# conc 128). 0.6 leaves the OS ~48 GiB and is still a massive KV cache
# for the 0.8B; GPU_MEM_UTIL overrides.
#
# Resumable: regen_out doubles as the resume log (skips done ids); rerun
# after any interruption. Outputs land next to the pool:
#   regen_<key>.jsonl        {id, response} resume log (+ .errors.jsonl)
#   regen_<key>_final.jsonl  merged corpus: context + on-policy final,
#                            subset re-attached (mix arms need it)
#
# Env: REGEN_CONC (default 16), EXTRA_SERVE_ARGS (e.g. --enforce-eager),
#      POOL (override input contexts).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/lib/runlib.sh"

KEY=${1:?usage: regen_pool.sh <model-key> [port]}
PORT=${2:-8138}
VENV=${VENV:-$ROOT/.venv-dflash/bin}
POOL=${POOL:-$(cd "$ROOT/.." && pwd)/llm-compressor/artifacts/v4_data/regen_all.jsonl}
OUTDIR=$(dirname "$POOL")
REGEN_OUT=$OUTDIR/regen_${KEY}.jsonl
MERGED_OUT=$OUTDIR/regen_${KEY}_final.jsonl
CONC=${REGEN_CONC:-16}
MTP_K=${MTP_K:-3}   # MTP speculative tokens (0 = off)
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}  # vLLM share of UNIFIED memory

run_init "regen_pool_${KEY}"
LOG=$RUNS_DIR/vllm_regen_${KEY}.log

# quant to serve: models.py hub id, else the local-only cascade dir
QUANT=$("$VENV/python" -c "
import sys; sys.path.insert(0, '$ROOT')
from models import resolve
_, info = resolve('$KEY')
q = info.get('quant_repo')
print(q if q else '$ROOT/Qwen3.5-0.8B-W4A16-GPTQ-cascade')" | tail -1)
log_event QUANT_RESOLVED "key=$KEY quant=$QUANT"

if [[ ! -e $QUANT ]]; then finish_fail "stage=quant-missing quant=$QUANT"; fi

MTP_ARGS=()
if (( MTP_K > 0 )); then
  MTP_ARGS=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP_K}}")
fi
launch_server "$LOG" env PATH="$VENV:/usr/local/cuda/bin:$PATH" \
  "$VENV/vllm" serve "$QUANT" --port "$PORT" --max-model-len 8192 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  "${MTP_ARGS[@]}" \
  ${EXTRA_SERVE_ARGS:-} \
  || finish_fail "stage=server-launch"
wait_healthy "$PORT" || finish_fail "stage=health"

# regen + merge in one resume-safe step (sampling from models.SERVING_PARAMS;
# pipeline knobs: seed 42, 640-tok cap — v1 corpus conventions)
run_with_watchdog "$RUN_DIR/regen.log" \
  "$VENV/python" "$ROOT/drafter/regen.py" \
  "$POOL" "$REGEN_OUT" "$MERGED_OUT" \
  --model "$KEY" --endpoint "http://localhost:${PORT}/v1/chat/completions" \
  --conc "$CONC" \
  || { stop_server; finish_fail "stage=regen"; }

# re-attach subset labels (merge writes {id, messages} only)
"$VENV/python" - "$POOL" "$MERGED_OUT" <<'PY'
import json, sys, os, tempfile
pool = {json.loads(l)["id"]: json.loads(l).get("subset") for l in open(sys.argv[1])}
tmp = sys.argv[2] + ".tmp"
n = miss = 0
with open(tmp, "w") as f:
    for line in open(sys.argv[2]):
        r = json.loads(line)
        r["subset"] = pool.get(r["id"], "?")
        miss += r["subset"] == "?"
        f.write(json.dumps(r) + "\n"); n += 1
os.replace(tmp, sys.argv[2])
print(f"subset re-attached: {n} rows, {miss} unknown")
PY

stop_server

# coverage summary by subset (the number that bit us: response rates)
"$VENV/python" - "$POOL" "$REGEN_OUT" <<'PY'
import json, sys
from collections import Counter
pool_sub, done = Counter(), set()
for l in open(sys.argv[1]):
    pool_sub[json.loads(l).get("subset", "?")] += 1
for l in open(sys.argv[2]):
    done.add(json.loads(l)["id"])
print(f"pool {sum(pool_sub.values())} | regen ok {len(done)} "
      f"({len(done)/sum(pool_sub.values()):.1%})")
PY
log_event REGEN_DONE "merged=$MERGED_OUT rows_ok=$(wc -l < "$REGEN_OUT")"
