# Data-scaling ladder (2026-10-15 →, in progress; corrected 2026-10-16)

**Question:** how little training data does the DFlash2 drafter actually
need — measured, not assumed. Current sequencing: the 0.8B testbed pilot
resolves shape/mix/diversity questions cheaply first; the 4B ladder
(corpus recovery → valid rung-1 → r130 decision) resumes after.

## Corpus truth (measured 2026-10-16 — supersedes everything earlier here)

`data/context_pool.jsonl` (the "regen_all" file, sha-pinned) is the **regen
INPUT, not a corpus**: slice_a+slice_b regrouped by conversation — bare
contexts, every row ends user/tool, **zero responses**. Consequences:

- The final assistant turn of every conversation is absent from the file;
  single-turn rows (51% — ALL math + science, plus 60K chat/IF/agent)
  carry no supervision and are dropped by prep.
- The champion trained on the node-side MERGED data (contexts + on-policy
  finals). The exact merged artifact is gone with the pod, but the
  ingredients were recovered 2026-10-16 and are sha-pinned in
  `data/MANIFEST.sha256` (`regen_4b*.jsonl`, dual-homed in the main
  checkout): regenerated finals + SFT/OPD splits + 0-error receipt. Its
  supervised budget was **~150M+** (incl. ~40M on-policy finals), not
  the 115–120M estimated from the input file.
- Ladder-side training (incl. rung 1) used context-internal SOURCE turns
  only — off-policy, zero finals ever.
- Mix: file row labels (chat 65 / math 12.5 / science 8 / …) describe the
  prompt pool. Ladder-side effective supervised mix: chat ~87% / agents
  ~11% / IF ~2%, math 0, science 0. Code ≈ 6% of context content
  (measured). Champion-side mix includes regen'd finals (rate unknown,
  node logs gone).
- Lengths: mean 2,115 / median 1,676 / p90 4,588 / p99 6,207 tokens,
  uncapped (paper spec caps at 3072; DSpark reference preps ≤1024).

**τ 1.05 (rung 1) is NOT a clean dead-zone number**: trained off-policy.
At t=0.7 it sat flat at ~1.05 across all epochs (n=4) — dead zone. Its
greedy sweep *improved* over the epoch-0 control (3.407 → 4.157, both
t=0, n=1 vs n=2, `runs/sweep-sweepfix2`) — greedy gains that don't transfer to
serving are the distribution-mismatch signature (taught source-model
text, evaluated against quant text); volume was confounded with
data-kind. The rung-1 *observations* stand
(extraction parity 20/20; loss 2.90→2.76; recall@16 0.30; flat
1.04–1.05 across all 6 epochs) — the conclusion doesn't.

Recovery: per-model regen is now a standing process
(`drafters/regen_pool.sh`; SERVING_PARAMS from `models.py`; see
AGENTS.md). The 0.8B pass is running; the 4B pass (corpus recovery)
reruns when 4B work resumes — the 4B ladder must re-measure rung 1 on
regen-fixed data before r130 is decided.

## Currency ledger: one unit for "amount of data"

Canonical: **unique supervised tokens** (loss_mask=1, once per conv);
conversations as the diversity co-unit (`drafters/data_audit.py`).
Rung 1 exact: 10,001,156 nominal → 7,498,489 unique (25% prefix dup) →
**5,190,561 supervised** (1.93× exchange). Never speak in characters.

Reference conversion (their samples → est. supervised tokens): 5K
speculators example ~1.5–3M (dead); DFlash 100K-sample ablations
~20–50M (τ 4.9–5.6 greedy); DSpark 100K prompts ≤1024 ~30–60M (τ 3.41);
DFlare 270K→2.4M ~55M–1B (rising throughout). Unified: dead below ~5M
supervised (two sources), known-good 20–60M, champion 3.79 at ~150M+.

Orthogonality caveat: at matched supervised volume, short-exchange
corpora carry ~3–4× more distinct conversations. Volume and diversity
are separate axes; our shape is volume-heavy/diversity-light — exactly
what the 0.8B shape arms (prefix/merged/first) measure.

## Why the ladder exists: the provenance of "260M"

The champion corpus was never a measured requirement: the DFlash paper
trained 800K samples (recipe co-adapted there); our 260M was the full
yield of a 4TB node (~1/8 paper volume); the v3→v4 "+0.52 per 9×" was
recipe-confounded (4-tap → 6-tap). Epoch-scaling at 260M was measured
(peak ep6 at serving temp); data-scaling at fixed recipe never was.

| source | data | framing |
|---|---|---|
| DFlash2 paper (2602.06036) | 800K samples | quality recipe |
| NeMo Automodel `qwen3_dflash2.yaml` | placeholder | "PerfectBlend, regenerate first" |
| speculators `dflash2_qwen3_8b_ultrachat_online_5k.sh` | 5K samples | smoke check, explicitly "not very good" |
| rasyosef DSpark (Llama-3.2) | 100K PerfectBlend prompts, ≤1k tok | published recipe; unmeasured count (same convention as ours) |

Nobody between 800K and 5K publishes a measured requirement. The ladder
is that instrument; rung results land in `ladder/ladder_results.csv` +
`RESULTS.md`.

## Is our prep idiomatic? (checked 2026-10-15 vs paper/DSpark/DFlare/speculators)

| practice | idiom | ours |
|---|---|---|
| on-policy regen by exact deployed target | universal | yes (quant verifier; SERVING_PARAMS now in models.py) |
| temp/template consistency train↔eval | author-enforced | yes (champion proves pipeline) |
| assistant-only loss mask | universal | yes |
| packing window | 8192 | yes |
| sample shape | ONE conv per sample (DFlash author: first-turn 1.2M → τ 6.3) | NO — turn-prefix rows re-supervise (`reshape_tokens.py` fixes) |
| length | short (~1k) or capped (3072) | median 1.7k, p90 4.6k uncapped — `--max-tokens` cap added |
| min-valid-tokens filter | speculators `--minimum-valid-tokens` | missing (pilot prep should add; 0.8B regen shows ~5% loopy tails + 92% math cap-hits worth filtering) |

## Mix doctrine (checked 2026-10-15)

Nobody ablates mix (DFlash 5.4.1 is a same-data comparison). Ecosystem
converged on Open-PerfectBlend = math 39% / chat ~30% / IF 16% / code 7%;
papers use Nemotron PT V2 + CodeAlpaca. Our effective corpus is
chat-heavy, math-zero (see corpus truth) — a PerfectBlend-shaped arm and
an agent-heavy arm at matched supervised budget are the pilot's mix dial.
Real math share needs MetaMath/ORCA-style prompts regenerated on-policy
(the pool's MMLU-style ghosts regen into short pick-a-letter answers).
