#!/usr/bin/env bash
# Data-scaling ladder rung: slice -> prep -> extract -> train -> sweep -> clean.
#
# Fixed champion recipe (v4 armB: lr 6e-4 cosine 6ep, gamma 7, taps_v4, fp8
# states, sdpa attention on GB10); the ONLY variable is conversation count.
# Each rung's number = best-epoch tau at serving temp (t=0.7, pooled n=4) via
# sweep_epochs.sh — the same selection procedure that produced the 260M
# anchor (3.79). States are deleted after training (checkpoints are 1.3GB).
#
# Usage:
#   bash drafter/data_ladder.sh <tokens_millions> [name] [port]
#     LADDER_BENCH=1   benchmark batch caps {16,64,128 rows} on 300 rows
#                      before extraction, run at the winner
#     LADDER_KEEP_STATES=1   skip the states cleanup
set -euo pipefail
cd "$(dirname "$0")/.."
source lib/runlib.sh

MILL=${1:?usage: data_ladder.sh <tokens_millions> [name] [port]}
NAME=${2:-r${MILL}m}
PORT=${3:-8210}
VENV=.venv-dflash
QUANT=${QUANT:-../llm-compressor/Qwen3.5-4B-W4A16-GPTQ-cascade}
SRC=${SRC:-../llm-compressor/artifacts/v4_data/regen_all.jsonl}
WORK=${LADDER_WORK:-../llm-compressor/ladder}
mkdir -p "$WORK"
run_init "ladder-$NAME"
trap 'finish_fail "interrupted"' INT TERM

[[ -d "$QUANT" ]] || finish_fail "quant dir missing: $QUANT"
[[ -f "$SRC" ]] || finish_fail "corpus missing: $SRC"
[[ $MILL =~ ^[0-9]+$ && $MILL -ge 1 ]] || finish_fail "tokens_millions >= 1 required"
log_event LADDER_RUNG_START "name=$NAME budget=${MILL}M work=$WORK"

# states need ~15.4KB/token (4B fp8 6-tap); fail before hours of work if thin
need_gb=$(( MILL * 16 ))  # KB->GB with headroom
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
(( free_gb >= need_gb )) || finish_fail "disk: need ~${need_gb}GB, have ${free_gb}GB"

# GPU must be free — GB10's nvidia-smi reports [N/A] for memory, so check
# for processes holding the device instead
used_pids=$(fuser /dev/nvidia0 2>/dev/null | tr -s ' ')
[[ -z $used_pids ]] || finish_fail "gpu busy: pids=$used_pids hold /dev/nvidia0"

# 1) slice — nested, conv-grouped, deterministic
if [[ ! -f "$WORK/rung_$NAME.jsonl" ]]; then
  "$VENV/python" drafter/slice_rungs.py --jsonl "$SRC" \
    --tokens $(( MILL * 1000000 )) --out "$WORK/rung_$NAME.jsonl" \
    --quant "$QUANT" > "$RUN_DIR/slice.log" 2>&1 || finish_fail "stage=slice"
fi
tail -1 "$RUN_DIR/slice.log" || true
[[ -s "$WORK/rung_$NAME.jsonl" ]] || finish_fail "empty rung slice"

# 2) tokenize
if [[ ! -f "$WORK/ds_$NAME/state.json" ]]; then
  "$VENV/python" drafter/prep_dataset.py --jsonl "$WORK/rung_$NAME.jsonl" \
    --out "$WORK/ds_$NAME" --quant "$QUANT" > "$RUN_DIR/prep.log" 2>&1 \
    || finish_fail "stage=prep"
fi

# 3) draft init — v4 6-tap convention (the local draft_init_dflash2 is the
#    4-tap v1-era init and must NOT be reused)
if [[ ! -f "$WORK/draft_init_v4/model.safetensors" ]]; then
  "$VENV/python" drafter/bootstrap_draft.py --model 4b --quant "$QUANT" \
    --out "$WORK/draft_init_v4" > "$RUN_DIR/bootstrap.log" 2>&1 \
    || finish_fail "stage=bootstrap"
fi

# 4) optional batch-cap benchmark on 300 rows (fresh ds only)
BATCH_ROWS=${LADDER_ROWS:-64}
BATCH_TOKENS=${LADDER_TOKENS:-65536}
if [[ ${LADDER_BENCH:-0} == 1 ]]; then
  bench="$RUN_DIR/bench.log"; : > "$bench"
  best=0
  while read -r bt br; do
    rm -rf "$WORK/bench_tmp"
    t0=$SECONDS
    "$VENV/python" drafter/extract_states.py --dataset "$WORK/ds_$NAME" \
      --verifier "$QUANT" --model 4b --out "$WORK/bench_tmp" --limit 300 \
      --max-batch-tokens "$bt" --max-batch-rows "$br" \
      > "$RUN_DIR/bench_${br}.log" 2>&1 || finish_fail "stage=bench cfg=$br"
    rate=$(grep -o '[0-9.]* rows/s' "$RUN_DIR/bench_${br}.log" | tail -1)
    rate=${rate% rows/s}
    echo "batch_tokens=$bt batch_rows=$br rate=${rate:-0}" | tee -a "$bench"
    awk -v r="${rate:-0}" -v b="$best" 'BEGIN{exit !(r>b)}' && best=$rate && BATCH_ROWS=$br && BATCH_TOKENS=$bt
  done << 'EOF'
16384 16
65536 64
131072 128
EOF
  rm -rf "$WORK/bench_tmp"
  log_event LADDER_BENCH "winner_rows=$BATCH_ROWS winner_tokens=$BATCH_TOKENS best_rate=$best"
fi

# 5) extract states (resume-safe; parity gate on 20 files)
nrows=$(wc -l < "$WORK/rung_$NAME.jsonl")
nhs=$(ls "$WORK/ds_$NAME/hidden_states" 2>/dev/null | wc -l)
if (( nhs < nrows )); then
  "$VENV/python" drafter/extract_states.py --dataset "$WORK/ds_$NAME" \
    --verifier "$QUANT" --model 4b --out "$WORK/ds_$NAME" \
    --max-batch-tokens "$BATCH_TOKENS" --max-batch-rows "$BATCH_ROWS" \
    --resume --parity 20 > "$RUN_DIR/extract.log" 2>&1 \
    || finish_fail "stage=extract"
fi

# 6) train — champion recipe, sdpa on GB10, per-epoch checkpoints
if [[ ! -d "$WORK/draft_$NAME/5" ]]; then
  "$VENV/bin/speculators" train \
    --speculator-type dflash2 --draft-attn-impl sdpa \
    --from-pretrained "$WORK/draft_init_v4" \
    --verifier-name-or-path "$QUANT" --data-path "$WORK/ds_$NAME" \
    --epochs 6 --lr 6e-4 --scheduler-type cosine --scheduler-warmup-ratio 0.04 \
    --dflash-decay-gamma 7 --max-anchors 512 --checkpoint-freq 1 \
    --train-data-ratio 0.950606 --log-freq 10 --save-path "$WORK/draft_$NAME" \
    > "$RUN_DIR/train.log" 2>&1 || finish_fail "stage=train"
fi

# 7) sweep at serving temp — best epoch, pooled n=4 (1 + 3 repeats)
TEMP=0.7 SWEEP_TOPK=1 SWEEP_REPEATS=3 \
  bash drafter/sweep_epochs.sh "$WORK/draft_$NAME" "lad_$NAME" 15 "$PORT" \
  > "$RUN_DIR/sweep.log" 2>&1 || finish_fail "stage=sweep"

# 8) record + cleanup states
tau=$(sed -n 's/^| 1 | \([^|]*\) | \*\*\([0-9.]*\)\*\* | \([0-9]*\) |$/\1 \2 \3/p' "$RUN_DIR/sweep.log" | head -1)
[[ -n $tau ]] || finish_fail "stage=record (no ranked row in sweep log)"
echo "$NAME,$MILL,$tau" >> "$WORK/ladder_results.csv"
log_event LADDER_RESULT "name=$NAME tokens=${MILL}M ckpt_tau_n=$(echo "$tau" | tr ' ' '/')"
if [[ ${LADDER_KEEP_STATES:-0} != 1 ]]; then
  freed=$(du -sb "$WORK/ds_$NAME/hidden_states" 2>/dev/null | cut -f1 || echo 0)
  rm -rf "$WORK/ds_$NAME/hidden_states"
  log_event LADDER_CLEANUP "freed_bytes=$freed"
fi
finish_ok "rung=$NAME winner_tau=$tau"
