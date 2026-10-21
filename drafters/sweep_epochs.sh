#!/usr/bin/env bash
# sweep_epochs.sh — checkpoint-selection sweep for DFlash2 drafters.
#
# Runs the serving-temperature ladder across every per-epoch checkpoint dir,
# then extra repeats on the top contenders, and emits a ranked table with
# pooled n. One command = the full selection campaign from AGENTS.md.
#
#   bash drafters/sweep_epochs.sh <checkpoints_parent> <sweep_name> [spec] [port]
#
# Env:
#   TEMP          sampling temperature (default 0.7 — the selection regime)
#   SWEEP_TOPK    contenders promoted to round 2 (default 2)
#   SWEEP_REPEATS extra runs per promoted contender (default 3 -> n=4 pooled)
#   VERIFIER_PATH verifier model id/path (default: the 4B W4A16 quant)
#   EXTRA_SERVE_ARGS passed through to eval_ladder.sh
#
# Artifacts: runs/sweep-<name>.<ts>.<pid>/ with events.jsonl, results.csv
# (all ladder rows), and summary.md (the ranked table). Each epoch eval is a
# child ladder-* run dir, recorded in SWEEP_EPOCH events.
#
# Runtime: ~6.5 min per eval on GB10 — 6 epochs + 2x3 repeats ≈ 80 min.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/lib/runlib.sh"

PARENT=$1; NAME=$2; SPEC=${3:-15}; PORT=${4:-8200}
[[ $SPEC =~ ^[0-9]+$ && $SPEC -ge 1 && $SPEC -le 32 ]] \
  || { echo "SPEC must be 1-32 (speculative tokens), got: $SPEC" >&2; exit 1; }
TEMP=${TEMP:-0.7}
SWEEP_TOPK=${SWEEP_TOPK:-2}
SWEEP_REPEATS=${SWEEP_REPEATS:-3}
(( SWEEP_REPEATS >= 1 )) || { echo "SWEEP_REPEATS>=1 required — 0 would allow single-draw winners" >&2; exit 1; }
(( SWEEP_TOPK >= 1 )) || { echo "SWEEP_TOPK>=1 required — 0 would burn round 1 with no promotion" >&2; exit 1; }
VERIFIER_PATH=${VERIFIER_PATH:-nchapman/Qwen3.5-4B-W4A16-GPTQ}
VENV=${VENV:-$ROOT/.venv-dflash/bin}

RUN_CMD="sweep_epochs.sh $* (TEMP=$TEMP TOPK=$SWEEP_TOPK REPEATS=$SWEEP_REPEATS verifier=$VERIFIER_PATH)" \
  run_init "sweep-$NAME"

[[ -d "$PARENT" ]] || { echo "not a dir: $PARENT" >&2; exit 1; }
epochs=()
for d in "$PARENT"/*/; do
  d=${d%/}
  # Only numeric per-epoch dirs: skip sub-epoch symlinks (epochN_stepM),
  # epochN_end links, and checkpoint_best — crash insurance, not sweep candidates.
  [[ "$(basename "$d")" =~ ^[0-9]+$ ]] || continue
  [[ -f "$d/model.safetensors" ]] && epochs+=("$d")
done
(( ${#epochs[@]} >= 2 )) || { log_event SWEEP_FAILED "reason=need>=2 checkpoint dirs with model.safetensors"; finish_fail "reason=no-epochs"; }
log_event SWEEP_START "epochs=${#epochs[@]} temp=$TEMP"
SWEEP_FAILED=0

# results.csv lives in the sweep run dir — self-contained, no /tmp litter
RESULTS_CSV="$RUN_DIR/results.csv"

tau_of() { awk -F, -v l="$1" '$1==l {v=$4} END {print v}' "$RESULTS_CSV"; }
mean_of() { grep "^$1," "$RESULTS_CSV" | awk -F, '{s+=$4; n++} END {printf "%.3f %d", s/n, n}'; }

run_epoch() {  # run_epoch <dir> <label>
  local child="$RUN_DIR/child-$2.log"
  if ! VERIFIER_PATH="$VERIFIER_PATH" RESULTS_CSV="$RESULTS_CSV" TEMP="$TEMP" \
      bash "$ROOT/drafters/eval_ladder.sh" "$1" "$2" "$SPEC" "$PORT" >"$child" 2>&1; then
    log_event SWEEP_EPOCH_FAILED "label=$2 dir=$1 child_log=$child"
    echo "  $2: FAILED (see $child)"
    SWEEP_FAILED=$(( ${SWEEP_FAILED:-0} + 1 ))
    return 0
  fi
  local t; t=$(tau_of "$2")
  log_event SWEEP_EPOCH "label=$2 tau=$t dir=$1"
  echo "  $2: τ=$t"
}

echo "== round 1: one ladder per epoch (temp $TEMP) =="
for d in "${epochs[@]}"; do run_epoch "$d" "$NAME-$(basename "$d")"; done
[[ -s "$RESULTS_CSV" ]] || finish_fail "reason=no-round1-results (every epoch eval failed — see child-*.log)"

echo "== round 2: $SWEEP_REPEATS extra runs on top $SWEEP_TOPK =="
mapfile -t top < <(awk -F, -v p="$NAME-" 'index($1, p)==1 {print $4, $1}' "$RESULTS_CSV" \
                  | sort -rn | head -"$SWEEP_TOPK" | awk '{print $2}')
for label in "${top[@]}"; do
  d=""
  for e in "${epochs[@]}"; do [[ "$NAME-$(basename "$e")" == "$label" ]] && d=$e; done
  [[ -n "$d" ]] || continue
  for ((r=1; r<=SWEEP_REPEATS; r++)); do run_epoch "$d" "${label}r$r"; done
  # rename repeat rows so pooled stats key on the base label (exact string
  # match, no regex interpolation of the label)
  "$VENV/python" - "$RESULTS_CSV" "$label" << 'PYREN' || finish_fail "stage=rename label=$label"
import os, sys
csv_path, label = sys.argv[1], sys.argv[2]
lines = open(csv_path).readlines()
out = []
for l in lines:
    head = l.split(",", 1)[0]
    if head.startswith(label + "r") and head[len(label)+1:].isdigit():
        l = label + "," + l.split(",", 1)[1]
    out.append(l)
tmp = csv_path + ".tmp"
with open(tmp, "w") as f:
    f.writelines(out)
os.replace(tmp, csv_path)   # atomic — never truncate the campaign's only CSV
PYREN
done

echo "== ranked (pooled means, temp $TEMP) =="
{
  echo "## Checkpoint sweep — $NAME (temp $TEMP, ladder-20 interval-weighted τ)"
  echo
  echo "| rank | checkpoint | τ mean | n |"
  echo "|---|---|---|---|"
  awk -F, -v p="$NAME-" 'index($1, p)==1 {s[$1]+=$4; n[$1]++} END {for (k in s) printf "%s %.3f %d\n", k, s[k]/n[k], n[k]}' \
    "$RESULTS_CSV" | sort -k2 -rn | awk '{printf "| %d | %s | **%s** | %s |\n", NR, $1, $2, $3}'
} | tee "$RUN_DIR/summary.md"

winner=$(awk -F, -v p="$NAME-" -v minn=$((1 + SWEEP_REPEATS)) 'index($1, p)==1 {s[$1]+=$4; n[$1]++} END {for (k in s) if (n[k] >= minn) printf "%.4f %s\n", s[k]/n[k], k}' \
  "$RESULTS_CSV" | sort -rn | head -1 | awk '{print $2}')
[[ -z "$winner" ]] && finish_fail "reason=no-label-with-n>=$((1 + SWEEP_REPEATS)) (lucky single draws are not a selection) failed=$SWEEP_FAILED"
finish_ok "winner=$winner epochs=${#epochs[@]} temp=$TEMP failed=${SWEEP_FAILED:-0}"
