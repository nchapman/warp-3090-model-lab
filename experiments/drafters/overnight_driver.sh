#!/bin/bash
# overnight_driver v3 — control-arm terminal pipeline:
#   wait ctrl trainer -> wait ladder drain -> best epoch -> final ladders
#   + bench924 -> stage everything -> PIPELINE_TERMINAL. (No OPD: budget.)
cd /root/warp-3090-quant-lab
LOG=/root/overnight.log
STAGE=/root/stage
say(){ echo "$(date -u +%m-%d\ %H:%M:%S) $*" >> "$LOG"; }
say "driver v3 start (ctrl arm)"

mean_for(){  # ctrl epoch*_end name -> mean tau of its 4 reps
  awk -v p="opt4b_CTRL_$1_r" 'index($2,p)==1{s+=$3;n++} END{if(n>=4)printf "%.4f",s/n}' /root/opt4b_taus.txt
}
ctrl_ckpts(){
  for d in draft_ctrl4b/epoch*_end/; do
    [ -f "${d}config.json" ] && [ -f "${d}model.safetensors" ] && printf "%s\n" "$d"
  done | xargs -r stat -c "%Y %n" | sort -n | awk '{print $2}' | xargs -r -n1 basename
}
stage_ckpt(){
  local n; n=$(basename "$1")
  [ -f "$STAGE/$n.tar.zst" ] && return 0
  tar -I "zstd -T8" -cf "$STAGE/$n.tar.zst" -C "$(dirname "$1")" "$n" \
    && say "staged $n" || say "stage FAILED $n"
}

# ---- Phase A: wait ctrl trainer terminal; stage checkpoints as they appear ----
CTRL_PID=$(cat /root/ctrl.pid)
while kill -0 "$CTRL_PID" 2>/dev/null; do
  sleep 180
  for d in draft_ctrl4b/*/; do [ -f "${d}model.safetensors" ] && stage_ckpt "$d"; done
done
say "ctrl trainer terminal"
for d in draft_ctrl4b/*/; do [ -f "${d}model.safetensors" ] && stage_ckpt "$d"; done

# ---- Phase B: wait ladder drain ----
while true; do
  pend=0
  for ck in draft_ctrl4b/epoch*_end/; do
    [ -f "${ck}config.json" ] && [ -f "${ck}model.safetensors" ] || continue
    grep -qxF "$(basename "$ck")" /root/ladder_done_ctrl.txt || pend=1
  done
  [ $pend -eq 0 ] && break
  sleep 60
done
say "ctrl ladders done"
cp /root/opt4b_taus.txt "$STAGE/" 2>/dev/null

BEST_NAME=$(for c in $(ctrl_ckpts); do m=$(mean_for "$c"); [ -n "$m" ] && echo "$m $c"; done \
  | sort -rn | head -1 | cut -d' ' -f2)
say "best ctrl epoch: ${BEST_NAME:-NONE}"

# ---- Phase C: final ladders + bench924 on best ctrl epoch ----
if [ -n "$BEST_NAME" ]; then
  SHIP="draft_ctrl4b/$BEST_NAME"
  say "ship candidate: $SHIP"
  export TEMP=0.7 VERIFIER_PATH=nchapman/Qwen3.5-4B-W4A16-GPTQ GPU_MEM_UTIL=0.6
  for rep in 1 2 3 4; do
    if bash drafters/eval_ladder.sh "$SHIP" "opt4b_CTRLFINAL_r${rep}" 15 8151 >> "$LOG" 2>&1; then
      say "ctrl final r$rep: $(grep '^TAU' "eval_opt4b_CTRLFINAL_r${rep}.out" 2>/dev/null | tail -1)"
    else
      say "ctrl final r$rep FAILED"
    fi
  done
  if bash drafters/bench_subsets.sh "$SHIP" opt4b_ctrl 8152 >> "$LOG" 2>&1; then
    say "bench924 done"; cp runs/opt4b_ctrl*/summary.md "$STAGE/" 2>/dev/null
  else
    say "bench924 FAILED"
  fi
else
  say "no complete ctrl tau table"
fi

# ---- Phase D: stage everything, manifest, terminal ----
for d in draft_ctrl4b/*/; do stage_ckpt "$d"; done 2>/dev/null
cp -r runs "$STAGE/runs_snapshot" 2>/dev/null
cp draft_opt4b/run.yaml draft_opt4b/train_command.txt draft_ctrl4b_train_command.txt "$STAGE/" 2>/dev/null
for f in draft_ctrl4b/run.yaml draft_ctrl4b/train_command.txt; do cp "$f" "$STAGE/" 2>/dev/null; done
cp /root/chain_sft.log /root/chain_ctrl.log /root/parity.log /root/ladder_watch.log /root/opt4b_taus.txt "$STAGE/" 2>/dev/null
( cd "$STAGE" && sha256sum ./*.tar.zst ./hs_states_manifest.sha256 > STAGE_MANIFEST.sha256 )
say "PIPELINE TERMINAL (ctrl arm) — assets staged, awaiting pull confirm"
touch /root/PIPELINE_TERMINAL
