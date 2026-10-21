#!/usr/bin/env bash
# regen_calib_self.sh [port] — self-authored calib arm (H2): the 4B BASE
# regenerates the final assistant turn of the frozen cascade-v1 calib.
#
# Why the base, not a quant: the quant doesn't exist yet, and the author-swap
# A/B vs gdnfix-qwenfinals (flash-authored finals, same contexts/sampling)
# wants "the model being quantized" as the author. Contexts byte-identical
# (regen_calib_finals.py contract); sampling = SERVING_PARAMS (t 0.7 /
# top_p 0.95 / top_k 20 — deployment point, same as the qwenfinals arm).
#
# Resumable (per-row context-sha skip in the python). GB10 unified memory:
# GPU_MEM_UTIL=0.6 doctrine. MTP off (data-gen saturates; MTP loses there).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/lib/runlib.sh"

PORT=${1:-8141}
VENV=${VENV:-$ROOT/.venv-dflash/bin}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
CALIB=${CALIB:-$ROOT/data/calib-cascade-v1.jsonl}
OUT=${OUT:-$ROOT/data/calib-cascade-4bself.jsonl}
CONC=${CONC:-96}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}

run_init "regen_calib_4bself"
LOG=$RUNS_DIR/vllm_serve_4bbase.log

SNAP=$("$VENV/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))" | tail -1)
[[ -e $SNAP ]] || finish_fail "stage=model-missing snap=$SNAP"
log_event MODEL_RESOLVED "model=$MODEL snap=$SNAP"

# no --max-model-len: native 262144 covers every calib context (~127K max)
# + 16K gen; KV fits easily inside 0.6 util on this GQA layout
launch_server "$LOG" env PATH="$VENV:/usr/local/cuda/bin:$PATH" \
  "$VENV/vllm" serve "$SNAP" --port "$PORT" --served-model-name "$MODEL" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  ${EXTRA_SERVE_ARGS:-} \
  || finish_fail "stage=server-launch"
wait_healthy "$PORT" || finish_fail "stage=health"

# smoke 3 rows first, always
"$VENV/python" "$ROOT/experiments/quants/regen_calib_finals.py" \
  --calib "$CALIB" --out "${OUT%.jsonl}.smoke.jsonl" --limit 3 \
  --api-base "http://localhost:${PORT}/v1" \
  --model "$MODEL" --concurrency 3 --max-tokens 16384 \
  || { stop_server; finish_fail "stage=smoke"; }
"$VENV/python" - "${OUT%.jsonl}.smoke.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
assert len(rows) == 3, f"expected 3 smoke rows, got {len(rows)}"
for r in rows:
    fin = r["conversations"][-1]
    assert fin["role"] == "assistant" and fin["content"].strip(), "bad final"
    assert r["conversations"][:-1], "context must be non-empty"
print(f"smoke OK: 3/3 rows, generator={rows[0]['generator']}, "
      f"final lens={[len(r['conversations'][-1]['content']) for r in rows]}")
PY

# full pass (resumable; python self-reports via lablog run_done/run_fail)
run_with_watchdog "$RUN_DIR/regen.log" \
  "$VENV/python" "$ROOT/experiments/quants/regen_calib_finals.py" \
  --calib "$CALIB" --out "$OUT" \
  --api-base "http://localhost:${PORT}/v1" \
  --model "$MODEL" --concurrency "$CONC" --max-tokens 16384 \
  || { stop_server; finish_fail "stage=regen"; }

stop_server

# receipts: row count + sha + sidecar
N=$(wc -l < "$OUT")
[[ $N -eq 999 ]] || { log_event ROW_COUNT_BAD "rows=$N"; finish_fail "stage=count rows=$N"; }
sha256sum "$OUT" "${OUT%.jsonl}.provenance.json"
log_event WRAPPER_DONE "rows=$N out=$OUT"
echo "DONE: $N rows -> $OUT"
