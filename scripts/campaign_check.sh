#!/usr/bin/env bash
# campaign_check.sh — verify the confirmation campaign is complete.
# Exits 0 only when every deliverable exists.
set -u
cd "$(dirname "$0")/.."
o=0
for b in mtp-a mtp-b ep6-a ep6-b; do
  grep -h '"type": "RUN_DONE".*"weighted_tau"' runs/bench-$b.*/events.jsonl >/dev/null 2>&1 \
    || { echo "missing: bench $b"; o=1; }
done
git diff --quiet && git diff --cached --quiet || { echo "dirty tree"; o=1; }
[ -z "$(git log origin/main..HEAD --oneline)" ] || { echo "unpushed commits"; o=1; }
git ls-files --error-unmatch drafter/bench_subsets.sh >/dev/null 2>&1 \
  || { echo "bench_subsets.sh not committed"; o=1; }
grep -q 'Confirmed category profile' RESULTS.md || { echo "RESULTS.md missing confirmed profile section"; o=1; }
grep -q '^| math_reasoning |' RESULTS.md || { echo "RESULTS.md missing subset table rows"; o=1; }
grep -q 'Category profile (RedHatAI speculator_benchmarks, temp 0, n=2)' hf/dflash2/README.md \
  || { echo "card missing category section"; o=1; }
card_sha=$(grep -o '[0-9a-f]\{64\}' hf/dflash2/README.md | head -1)
hub_sha=$(.venv-quant/bin/python -c "
from huggingface_hub import HfApi
for f in HfApi().model_info('nchapman/Qwen3.5-4B-DFlash2', files_metadata=True).siblings:
    if f.rfilename == 'model.safetensors' and getattr(f, 'lfs', None):
        print(f.lfs.sha256); break" 2>/dev/null)
[ -n "$card_sha" ] && [ "$card_sha" = "$hub_sha" ] \
  || { echo "card sha ($card_sha) != Hub model sha ($hub_sha)"; o=1; }
grep -h '"weighted_tau": "[^"]*subsets=9' runs/bench-ep6-a.*/events.jsonl >/dev/null 2>&1 \
  || { echo "bench events missing subsets=9"; o=1; }
n=$(grep -l '"type": "RUN_DONE".*"status": "ok"' runs/probe.*/events.jsonl 2>/dev/null | wc -l)
[ "$n" -ge 2 ] || { echo "probe confirmations: $n/2"; o=1; }
grep -h '"type": "RUN_DONE".*"status": "ok"' runs/regression.*/events.jsonl >/dev/null 2>&1 \
  || { echo "regression not passed"; o=1; }
echo "campaign: benches=$(ls runs/ 2>/dev/null | grep -c '^bench-') probes=$n/2 tree=$([ $o -eq 0 ] && echo CLEAN || echo INCOMPLETE)"
exit $o
