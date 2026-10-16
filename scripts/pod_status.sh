#!/usr/bin/env bash
# pod_status.sh — one-shot machine utilization snapshot (run ON the pod).
# The standard check-in companion: GPU util/mem/power, GPU processes (zombie
# spotting), load, RAM, disk, plus the newest run event for stage context.
# For run logs use scripts/tail_run.sh; this is machine-level only.
set -u
echo "== $(date -u +%FT%TZ)  up $(uptime -p | cut -d' ' -f2-)"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader \
  | awk -F', ' '{printf "gpu    %s  util %s  mem %s/%s  %s  %s\n",$1,$2,$3,$4,$5,$6}'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
  | while IFS=', ' read -r pid mem; do
      printf "proc   pid=%s mem=%s cmd=%s\n" "$pid" "$mem" "$(ps -o comm= -p "$pid" 2>/dev/null)"
    done
printf 'load   %s  ram %s/%sGB  disk %s\n' \
  "$(cut -d' ' -f1-3 /proc/loadavg)" \
  "$(free -g | awk 'NR==2{print $3}')" "$(free -g | awk 'NR==2{print $2}')" \
  "$(df -h / | awk 'NR==2{print $4" free"}')"
latest=$(ls -t /root/warp-3090-quant-lab/runs/*/events.jsonl 2>/dev/null | head -1)
[[ -n $latest ]] && printf "run    %s\n       %s\n" \
  "$(basename "$(dirname "$latest")")" "$(tail -1 "$latest" | cut -c1-160)"
