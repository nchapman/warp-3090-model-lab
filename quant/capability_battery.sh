#!/bin/bash
# quant/capability_battery.sh — paired base-vs-quant capability battery.
#
# Runs lm-eval (vLLM backend, same serving path as deployment) over the
# frozen task list for BOTH the BF16 base and a quant dir, then emits
# paired deltas with bootstrap CIs via quant/battery_report.py.
#
# Doctrine (README): capability evals are CLIFF DETECTORS — greedy (t=0),
# paired deltas, >=5pt cliffs only, never rankings. COMPLETION prompts
# (task-native, no chat template) so both sides see identical formatting
# and models complete stubs directly instead of answering conversationally
# (a chat-formatted humaneval prompt yields prose + fenced code that the
# exec grader rejects). Reasoning handled harness-side via lm-eval's
# thinking-model mode: task stop sequences are WITHHELD from vLLM (so
# "\ndef"/"\nif" inside <think> can't truncate generation mid-reasoning),
# everything up to the last </think> is stripped post-hoc, then task stops
# apply to the answer text only — both sides identically. MAX_GEN_TOKS
# (4096) gives think-phase headroom; an answer that never arrives within
# it is graded a miss on both sides alike (honest signal, not artifact).
# Rejected alternatives (measured 2026-10-17, 0.8B): /no_think suffix in
# completion mode degenerates ("/ / /" loop); chat template with thinking
# ON never closes the think loop within budget (base 0.8B loops to cap on
# every doc).
#
# usage: capability_battery.sh --model 0.8b|4b [--base PATH] --quant DIR \
#           [--tasks a,b,c] [--limit N] [--tag X] [--sides base|quant|both]
#           [--allow-code-eval]
#   --model   models.py key; resolves the BF16 base hub id (--base overrides)
#   --tasks   comma list; default = the frozen battery (see DEFAULT_TASKS)
#   --limit   per-task doc cap (smokes only — never for recorded numbers)
#   --allow-code-eval  opt in to executing model-generated code (pass@1
#                      tasks need it; see HF_ALLOW_CODE_EVAL below)
# env: GPU_MEM_UTIL (0.6 GB10 shared-box ceiling)  MAX_MODEL_LEN (8192)
#      MEM_GUARD_MB (16384)  MAX_GEN_TOKS (4096)  EXTRA_LMEVAL_ARGS ("")
#      HF_ALLOW_CODE_EVAL (0; also settable via --allow-code-eval)
set -euo pipefail
source "$(dirname "$0")/../lib/runlib.sh"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LM_EVAL="$ROOT/.venv-dflash/bin/lm-eval"          # vllm 0.31.0 lives here
LMEVAL_VER=$($ROOT/.venv-dflash/bin/python -c "import lm_eval; print(lm_eval.__version__)")
PY="$ROOT/.venv-quant/bin/python"
DEFAULT_TASKS="humaneval_plus,mbpp_plus,ifeval,minerva_math500,gpqa_diamond_cot_zeroshot"

MODEL_KEY=4b BASE="" QUANT="" TASKS=$DEFAULT_TASKS LIMIT="" TAG="" SIDES=both ALLOW_CODE_EVAL=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL_KEY=$2; shift 2;;
    --base)  BASE=$2; shift 2;;
    --quant) QUANT=$2; shift 2;;
    --tasks) TASKS=$2; shift 2;;
    --limit) LIMIT=$2; shift 2;;
    --tag)   TAG=$2; shift 2;;
    --sides) SIDES=$2; shift 2;;
    --allow-code-eval) ALLOW_CODE_EVAL=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[[ -n $QUANT ]] || { echo "--quant DIR is required" >&2; exit 2; }
case $SIDES in
  both|base|quant) ;;
  *) echo "--sides must be base|quant|both (got: $SIDES)" >&2; exit 2;;
esac

if [[ -z $BASE ]]; then
  BASE=$("$PY" -c "import sys; sys.path.insert(0, '$ROOT'); from models import resolve; print(resolve(sys.argv[1])[0])" "$MODEL_KEY")
fi

# vLLM's spawned EngineCore must find venv-provided build tools (flashinfer JIT
# needs ninja) — invoking the lm-eval binary by path does NOT put them on PATH.
export PATH="$ROOT/.venv-dflash/bin:$PATH"

# pass@1 tasks execute untrusted model-generated code in a local subprocess
# (lm-eval code_eval, the standard HumanEval practice — no docker here; on a
# pod the pod container IS the sandbox). Opt-in, never default: pass
# --allow-code-eval (or export HF_ALLOW_CODE_EVAL=1) or the run fails fast
# when the task list needs it.
if [[ $ALLOW_CODE_EVAL == 1 || ${HF_ALLOW_CODE_EVAL:-0} == 1 ]]; then
  export HF_ALLOW_CODE_EVAL=1
  UNSAFE_CODE_ARGS=(--confirm_run_unsafe_code)
elif [[ ",$TASKS," == *humaneval* || ",$TASKS," == *mbpp* ]]; then
  echo "task list includes exec-graded tasks (humaneval/mbpp pass@1):" >&2
  echo "rerun with --allow-code-eval to opt in to executing model-generated code" >&2
  exit 2
else
  UNSAFE_CODE_ARGS=()
fi

GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_GEN_TOKS=${MAX_GEN_TOKS:-4096}
export MEM_GUARD_MB=${MEM_GUARD_MB:-16384}

# Thinking-model handling (see header): strip reasoning post-hoc, never
# truncate mid-think. Truncation-by-budget failures (answer never reached)
# then surface honestly as graded misses on both sides alike.
THINK_ARGS="enable_thinking=True,think_end_token=</think>"

# lm-eval 0.4.13 postprocess_generated_text() does
# `generation.split(think_end_token)[-1].lstrip()` — the lstrip is
# UNCONDITIONAL, so any code-completion that legitimately starts with
# indentation (humaneval stub bodies) loses it the moment think_end_token
# is set, the glued program hits IndentationError, and pass@1 collapses
# to 0/0 (measured). Patch it out idempotently (same in-venv patch
# pattern as our speculators patches); logged as an event so the run
# record shows which lm-eval behavior produced the numbers.
patch_lmeval_think_strip() {
  local f status
  f=$("$ROOT/.venv-dflash/bin/python" -c "import lm_eval.models.utils as u; print(u.__file__)" 2>/dev/null | tail -1)
  if [[ -z $f || ! -f $f ]]; then
    log_event THINK_STRIP_PATCH probe-failed
    echo "cannot locate lm_eval/models/utils.py in .venv-dflash (venv broken?)" >&2
    return 1
  fi
  if grep -q "BATTERY: preserve leading ws" "$f"; then
    status=already
  elif grep -q "\[-1\].lstrip()" "$f"; then
    "$ROOT/.venv-dflash/bin/python" - "$f" <<'EOF'
import os, sys
p = sys.argv[1]
s = open(p).read()
old = "generation = generation.split(think_end_token)[-1].lstrip()"
new = "generation = generation.split(think_end_token)[-1]  # BATTERY: preserve leading ws"
assert s.count(old) == 1, f"expected exactly 1 site, found {s.count(old)}"
tmp = p + ".batterytmp"  # atomic: never leave a truncated shared module
open(tmp, "w").write(s.replace(old, new))
os.replace(tmp, p)
EOF
    status=patched
  else
    status=missing   # upstream changed — think handling needs review
  fi
  log_event THINK_STRIP_PATCH "$status" "file=$f"
  [[ $status != missing ]]
}

run_init "battery-${TAG:-$MODEL_KEY}"
require_file "$LM_EVAL" "$QUANT/config.json"
patch_lmeval_think_strip
log_event BATTERY_CONFIG "model=$MODEL_KEY" "base=$BASE" "quant=$QUANT" \
  "tasks=$TASKS" "limit=${LIMIT:-full}" "gpu_mem_util=$GPU_MEM_UTIL" \
  "max_gen_toks=$MAX_GEN_TOKS" "think_strip=</think>" "allow_code_eval=${HF_ALLOW_CODE_EVAL:-0}" \
  "lm_eval=$LMEVAL_VER" "leading_ws=preserved"

run_side() {  # run_side <name> <model_path> [extra model_args]
  local side=$1 path=$2 extra=${3:-}
  local out="$RUN_DIR/raw/$side"
  mkdir -p "$out"
  local margs="pretrained=$path,gpu_memory_utilization=$GPU_MEM_UTIL,max_model_len=$MAX_MODEL_LEN,$THINK_ARGS"
  [[ -n $extra ]] && margs="$margs,$extra"
  local args=(--model vllm --model_args "$margs" --tasks "$TASKS"
              --batch_size auto --output_path "$out" --log_samples
              --seed 0
              --gen_kwargs temperature=0,max_gen_toks=$MAX_GEN_TOKS)
  [[ ${#UNSAFE_CODE_ARGS[@]} -gt 0 ]] && args+=("${UNSAFE_CODE_ARGS[@]}")
  [[ -n $LIMIT ]] && args+=(--limit "$LIMIT")
  [[ -n ${EXTRA_LMEVAL_ARGS:-} ]] && read -r -a extra_args <<< "$EXTRA_LMEVAL_ARGS" && args+=("${extra_args[@]}")
  log_event LMEVAL_START "side=$side" "path=$path"
  run_with_watchdog "$RUN_DIR/$side.log" "$LM_EVAL" "${args[@]}" \
    || { finish_fail "stage=$side"; }
  log_event LMEVAL_DONE "side=$side"
}

[[ $SIDES == both || $SIDES == base ]]  && run_side base  "$BASE" "dtype=bfloat16"
[[ $SIDES == both || $SIDES == quant ]] && run_side quant "$QUANT" ""

if [[ $SIDES == both ]]; then
  REPORT="$RUN_DIR/battery-report.json"
  if ! "$PY" "$ROOT/quant/battery_report.py" --base "$RUN_DIR/raw/base" \
      --quant "$RUN_DIR/raw/quant" --json "$REPORT" 2>&1 | tee "$RUN_DIR/report.txt" >&2; then
    finish_fail "stage=report"
  fi
  WORST=$("$PY" -c "
import json, sys
r = json.load(open(sys.argv[1]))
w = r.get('worst') or {}
print(f\"{w.get('task','-')}:{w.get('delta_pt',0):+.1f}pt\")" "$REPORT")
  finish_ok "tasks=$TASKS" "limit=${LIMIT:-full}" "worst=$WORST" "report=$REPORT"
else
  finish_ok "tasks=$TASKS" "limit=${LIMIT:-full}" "sides=$SIDES" "note=report skipped (single side)"
fi
