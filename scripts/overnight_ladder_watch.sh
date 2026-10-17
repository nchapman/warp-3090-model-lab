#!/bin/bash
# ladder_watch v3 — control-arm per-epoch ladders (draft_ctrl4b, t=0.7 n=4).
cd /root/warp-3090-quant-lab
export TEMP=0.7 VERIFIER_PATH=nchapman/Qwen3.5-4B-W4A16-GPTQ GPU_MEM_UTIL=0.6
DONE=/root/ladder_done_ctrl.txt; touch $DONE
CTRL_PID=$(cat /root/ctrl.pid)
log(){ echo "$(date -u +%H:%M:%S) $*" >> /root/ladder_watch.log; }

run_ladders(){  # $1=ckpt dir name, $2=eval name prefix
  local ck=$1 tag=$2 rep
  for rep in 1 2 3 4; do
    if bash drafter/eval_ladder.sh "draft_ctrl4b/$ck" "${tag}_r${rep}" 15 8151 >> /root/ladder_watch.log 2>&1; then
      log "  $tag r$rep: $(grep '^TAU' eval_${tag}_r${rep}.out 2>/dev/null | tail -1)"
    else
      log "  $tag r$rep FAILED"
    fi
  done
  grep -h '^TAU' eval_${tag}_r*.out 2>/dev/null >> /root/opt4b_taus.txt
  echo "$ck" >> $DONE
}

while true; do
  for ck in draft_ctrl4b/epoch*_end/; do
    [ -f "${ck}config.json" ] && [ -f "${ck}model.safetensors" ] || continue
    name=$(basename "$ck")
    grep -qxF "$name" $DONE && continue
    log "ctrl epoch ladder: $name"
    run_ladders "$name" "opt4b_CTRL_${name}"
  done
  if ! kill -0 "$CTRL_PID" 2>/dev/null; then
    pend=0
    for ck in draft_ctrl4b/epoch*_end/; do
      [ -f "${ck}config.json" ] && [ -f "${ck}model.safetensors" ] || continue
      grep -qxF "$(basename "$ck")" $DONE || pend=1
    done
    [ $pend -eq 0 ] && { log "watcher3: ctrl trainer done, all epoch ckpts evaluated"; exit 0; }
  fi
  sleep 60
done
