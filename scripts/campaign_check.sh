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
# data-based assertions: weighted tau from the clean runs must agree across
# draws and match the numbers published in RESULTS.md and the card
calc() { awk -F, 'NR>1 {a+=$2*$3; t+=$3} END {if (t) printf "%.2f", a/t}' "$1"; }
m_c=$(calc results/bench-mtp-c.csv 2>/dev/null); m_d=$(calc results/bench-mtp-d.csv 2>/dev/null)
e_c=$(calc results/bench-ep6-c.csv 2>/dev/null); e_d=$(calc results/bench-ep6-d.csv 2>/dev/null)
[ -s "results/bench-ep6-c.csv" ] && [ -s "results/bench-mtp-c.csv" ] || { echo "clean-run CSVs missing from results/"; o=1; }
if [ -n "$m_c" ] && [ -n "$m_d" ]; then
  ok=$(awk -v a="$m_c" -v b="$m_d" 'BEGIN {print (a-b<0.05 && b-a<0.05) ? 1 : 0}')
  [ "$ok" = 1 ] || { echo "mtp draws disagree: $m_c vs $m_d"; o=1; }
fi
if [ -n "$e_c" ] && [ -n "$e_d" ]; then
  ok=$(awk -v a="$e_c" -v b="$e_d" 'BEGIN {print (a-b<0.05 && b-a<0.05) ? 1 : 0}')
  [ "$ok" = 1 ] || { echo "ep6 draws disagree: $e_c vs $e_d"; o=1; }
fi
pool() { awk -F, 'FNR>1 {a+=$2*$3; t+=$3} END {if (t) printf "%.2f", a/t}' "$1" "$2"; }
pub=$(pool results/bench-ep6-c.csv results/bench-ep6-d.csv)
[ -n "$pub" ] && grep -q "\*\*$pub\*\*" RESULTS.md && grep -q "\*\*$pub\*\*" hf/dflash2/README.md \
  || { echo "published weighted tau ($pub) not found in RESULTS.md/card"; o=1; }
mpub=$(pool results/bench-mtp-c.csv results/bench-mtp-d.csv)
[ -n "$mpub" ] && grep -q "\*\*$mpub\*\*" RESULTS.md \
  || { echo "published MTP weighted tau ($mpub) not found in RESULTS.md"; o=1; }
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
# dataset revision actually used must be the published pin
grep -h '2ae86affa2cb97a972b7fc681dd51c04fbff083e' runs/bench-ep6-c.*/events.jsonl runs/bench-ep6-d.*/events.jsonl >/dev/null 2>&1 \
  || { echo "clean runs not on the published dataset pin"; o=1; }
echo "campaign: benches=$(ls runs/ 2>/dev/null | grep -c '^bench-') probes=$n/2 tree=$([ $o -eq 0 ] && echo CLEAN || echo INCOMPLETE)"
exit $o
