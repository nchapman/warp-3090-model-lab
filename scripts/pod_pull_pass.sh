#!/bin/bash
# pull_pass.sh — one local sync pass from the pod (run repeatedly, chained)
# Verifies tar integrity each pass; when PIPELINE_TERMINAL + manifest verify,
# touches /root/PULL_CONFIRMED on the pod (green light for local ship+teardown)
cd /home/nchapman/Code/warp-3090-quant-lab || exit 1
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS="-i $KEY -p 30117"
POD=root@38.80.152.146
mkdir -p pod_snapshot/stage pod_snapshot/evals

rsync -e "ssh $SSH_OPTS" --partial --timeout=300 "$POD:/root/stage/" pod_snapshot/stage/ || echo "WARN stage rsync incomplete"
rsync -e "ssh $SSH_OPTS" --timeout=120 "$POD":/root/chain_sft.log "$POD":/root/parity.log "$POD":/root/ladder_watch.log "$POD":/root/overnight.log "$POD":/root/opt4b_taus.txt pod_snapshot/ 2>/dev/null
rsync -e "ssh $SSH_OPTS" --timeout=120 "$POD:/root/warp-3090-quant-lab/eval_opt4b_*.out" pod_snapshot/evals/ 2>/dev/null
rsync -e "ssh $SSH_OPTS" --timeout=120 --exclude '*.safetensors' "$POD:/root/warp-3090-quant-lab/runs/" pod_snapshot/runs/ 2>/dev/null

echo "--- tar integrity ---"
bad=0
for f in pod_snapshot/stage/*.tar.zst; do
  [ -e "$f" ] || continue
  zstd -t "$f" 2>/dev/null || { echo "CORRUPT: $f"; bad=1; }
done
[ "$bad" = 0 ] && echo "all present tars OK"

echo "--- pod status ---"
ssh $SSH_OPTS "$POD" 'tail -5 /root/overnight.log 2>/dev/null; tail -1 /root/chain_sft.log | grep -oE "epoch=[0-9]+|global_step=[0-9]+" | tail -2; tail -8 /root/opt4b_taus.txt 2>/dev/null; ls /root/PIPELINE_TERMINAL >/dev/null 2>&1 && echo TERMINAL_SET'

if ssh $SSH_OPTS "$POD" test -f /root/PIPELINE_TERMINAL; then
  if ( cd pod_snapshot/stage && sha256sum -c STAGE_MANIFEST.sha256 >/dev/null 2>&1 ); then
    ssh $SSH_OPTS "$POD" touch /root/PULL_CONFIRMED && echo "PULL_CONFIRMED set"
  else
    echo "terminal but manifest not yet verified — pass again"
  fi
fi
