#!/bin/bash
# Goal verify: campaign grid recorded, 27b synced + sha-verified, pods stopped.
# Exits 0 only when every local receipt exists and both campaign pods are EXITED.
cd /home/nchapman/Code/warp-3090-quant-lab-moe || exit 1
fail=""

R=results/campaign-20261019-grid.md
for tag in "ARM1 long8k" "ARM2 if3x" "ARM3 4bself" "CHAMPION re-anchor"; do
  grep -q "$tag" "$R" 2>/dev/null || fail="$fail report-missing[$tag]"
done
n=$(grep 'VERDICT:' "$R" 2>/dev/null | grep -vc '(pending)') || true
[ "${n:-0}" -ge 3 ] || fail="$fail verdicts-incomplete($n/3)"

D=artifacts/Qwen3.8-27B-W4A16-GPTQ-qwenfinals
if [ -s "$D/SHA256SUMS" ] && ls "$D"/*.log "$D"/*probe* >/dev/null 2>&1; then :; else
  fail="$fail 27b-receipts-incomplete"
fi
# published artifact: today's 27b build lives in the PRIVATE holding repo
# (incumbent cascade-v1 repo stays canonical — campaign verdict 2026-10-19);
# verify its model.safetensors LFS sha256 matches the local receipt sha
sha=$(grep -o '[0-9a-f]\{64\}' "$D/SHA256SUMS" 2>/dev/null | head -1)
lfs=$(curl -sS -m 20 "https://huggingface.co/api/models/nchapman/Qwen3.8-27B-W4A16-GPTQ-qwenfinals/tree/main" \
  | grep -o '"lfs":{[^}]*}' | grep -o '"oid":"sha256:[0-9a-f]*"' | head -4)
if [ -n "$sha" ] && echo "$lfs" | grep -q "$sha"; then :; else
  fail="$fail hf-not-published-or-sha-mismatch"
fi

if [ -s .pi/goal-pod-ids ]; then
  . ~/.config/secrets.env
  resp=$(curl -sS -m 20 "https://api.runpod.io/v2/pods" \
    -H "Authorization: Bearer $RUNPOD_API_KEY") || fail="$fail api-error"
  for id in $(cat .pi/goal-pod-ids); do
    st=$(echo "$resp" | .venv-quant/bin/python -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print('PARSE_ERR'); raise SystemExit
for p in d.get('pods', []):
    if p.get('id') == '$id':
        print(p.get('status', '?')); break
else:
    print('NOT_FOUND')
")
    [ "$st" = "EXITED" ] || fail="$fail pod-not-stopped[$id:$st]"
  done
else
  fail="$fail pod-ids-not-recorded"
fi

if [ -n "$fail" ]; then echo "GOAL INCOMPLETE:$fail"; exit 1; fi
echo "GOAL COMPLETE: grid recorded + 3 verdicts + 27b sha-verified + pods EXITED"
