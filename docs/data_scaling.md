# Data-scaling ladder (2026-10-15, in progress)

**Question:** how little training data does the 4B DFlash2 drafter actually
need? Measured — not assumed.

## Why this experiment exists: the provenance of "260M"

The champion corpus (260M tokens, 100k conversations) was never a *measured*
data requirement. The chain of evidence was:

1. **The DFlash paper (arXiv:2602.06036)** trained on **800K samples** and
   co-adapted its recipe (lr 6e-4 cosine, 6 ep, block 16) at that scale
   (research doc: "paper recipe... is co-adapted with 800K-sample training
   sets", "our conservative recipe won at 29M, theirs at 800K samples").
2. Our 260M rung was sized as the **full yield of a 4TB H200 node**
   ("144k-row full-yield ~260M tok") — the biggest rung that fit, roughly
   1/8 of the paper's data volume — extrapolated from the paper's curve,
   not from our own measurement.
3. The v3→v4 jump (+0.52 per 9x data) *appeared* to confirm "more is better"
   but was **confounded**: v3 used 4-tap `[2,16,29,32]`, v4 6-tap
   `[2,10,17,24,29,32]`, different recipe. Epoch-scaling at fixed 260M was
   measured (peak in-budget at ep6, serving temp); **data-scaling at fixed
   recipe never was**.

External reference points (checked 2026-10-15):

| source | data | their framing |
|---|---|---|
| DFlash2 paper (arXiv:2602.06036) | 800K samples | quality recipe |
| NeMo Automodel `qwen3_dflash2.yaml` | no fixed count (placeholder; params = our champion recipe lineage: 6e-4, 6ep, warmup 0.04, block 16, 5 layers) | "PerfectBlend, regenerate first" |
| vllm-project/speculators example `dflash2_qwen3_8b_ultrachat_online_5k.sh` | **5k samples** (on-policy regen, pretokenized input_ids+loss_mask, seq 8192, 5ep, lr 3e-4) | "with just 5k samples, performance will not be very good... sanity check" |
| rasyosef/train-dspark-draft-models (DSpark, Llama-3.2) | **100k Open PerfectBlend prompts**, on-policy regen (same doctrine), 3ep, <=1k-tok samples (~100M tok), 8192 packing window | published drafter recipe; no sample-count ablation (80k appears only as a Kaggle free-tier budget) — likely the anchor for our own 100k-conversation corpus size |

Notable: nobody between the paper (800K) and the smoke example (5k)
publishes a measured requirement — the ecosystem has a quality pole and a
smoke pole and nothing in between (the DSpark repo's 100k is the same
unmeasured convention, matching our champion's conversation count). Our
ladder brackets exactly that gap (rung 1 = ~their smoke scale, champion =
1/8 paper scale) with measured tau; rung 1's number is a useful public
datapoint regardless of where the knee lands.

## First measured point (rung 1, 2026-10-15)

**10M tokens (3,257 convs, 6 epochs = 60M effective): tau = 1.05 @ t=0.7,
n=4 (winner: epoch 2; all six epochs flat 1.04-1.05).**

The drafter is non-viable at this scale — acceptance pinned at chance.
This is a real data cliff, not a pipeline artifact: extraction parity
20/20; training converged smoothly (loss 2.90 -> 2.76, recall@16 up
0.274 -> 0.299); teacher-forced val accept length itself only 1.19 ->
1.21 with oracle top-16 at ~2.0 (the candidate set barely contains the
target's tokens); and the same sweep pipeline scored the champion's
epoch 0 (260M tokens seen) at 3.407 yesterday. The "smoke pole" in the
table above is therefore literally useless as a drafter, not merely
suboptimal. The knee lives somewhere in 30-100M; rung 2 (40M) is the
decisive experiment.

So "we need ~260M tokens" is an assumption inherited from the paper + disk
geometry. The ladder tests it. If the curve is flat well below 260M, the 9B
and 27B campaigns shrink accordingly (27B at 40M-equivalent = ~1.2TB states —
fits any 4TB pod un-trimmed, vs ~8TB at full corpus).

## Design

Three points, fixed champion recipe (armB: lr 6e-4 cosine 6ep, gamma 7,
6-tap fp8 states, train-ratio 0.950606, sdpa on GB10) — the only variable
is conversation count:

| rung | tokens | convs/rows | states (fp8) | where | tau @ t=0.7 (n=4) |
|---|---|---|---|---|---|
| r10 | 10.0M | 3,257 / 4,729 | 155 GB | GB10 | (running) |
| r40 | 40M | ~13k / ~19k | 620 GB | GB10 (43h) or B200 pod (~$10-15) — TBD | — |
| **anchor** | **260M** | 100k / 145,018 | 3.2 TB (pod, gone) | shipped ep6 | **3.79** |

- Rungs are **nested conversation-grouped slices** of the committed
  regenerated corpus (`regen_all.jsonl`): sorted-conv walk, smaller rungs
  are prefix-subsets of larger ones (`drafter/slice_rungs.py`).
- Each rung's number = **best-epoch tau at serving temp** via
  `sweep_epochs.sh` (t=0.7, pooled n=4) — the same selection procedure that
  produced the anchor. Never final-epoch or val-loss.
- One command per rung: `drafter/data_ladder.sh <tokens_millions> [name]`
  (slice -> prep -> bootstrap v4 init -> extract -> train -> sweep ->
  states cleanup). GB10 measured: extraction bandwidth-bound ~1-2k tok/s
  (~50% of peak on both GB10 and H200 — the machine, not the kernels);
  training ~5s/step at 8192-token batches (~2k tok/s effective).

## Decision rules (scouting pass, not a curve fit)

With tau noise ~+-0.1 at n=4, three points give a categorical read:

- **r10 ~= r40 ~= 3.79** — saturated early: corpus budget for 9B/27B is the
  smallest rung that overlaps the anchor; storage debate over.
- **r10 < r40 < 3.79, rising** — add rung(s) between r40 and 260M to locate
  the knee; 9B/27B budget from the measured slope.
- **r40 ~= 3.79 but r10 short** — knee in (10M, 40M); probe 20M only if the
  last terabyte matters for 27B.

Caveats, disclosed: the anchor was *trained* on H200 (rungs on GB10; same
code/recipe — cross-machine acceptance measured at sigma ~0.01); anchor tau
was measured on this machine, so comparisons are same-instrument.

## Status

- r10: extracted (4729/4729, parity 20/20), training 6 epochs (~13h GB10;
  resumed cleanly after a tool-timeout kill — speculators resumes from
  `epoch<n>_end`). Point expected 2026-10-15.
- r40: pending r10 (pod-vs-local fork).
- Results appended to `../llm-compressor/ladder/ladder_results.csv` and
  published into RESULTS.md when the curve is complete.

## Literature check: is our data prep idiomatic? (2026-10-15)

Verdict across DFlash paper (2602.06036), DFlare (2606.02091), speculators
docs/CLI, NeMo Automodel recipe, DSpark repo, and SpecForge issue #465:

| practice | idiom | ours |
|---|---|---|
| on-policy regen by the exact target | universal | yes (quant verifier, t=0.7 = eval temp) |
| temp/template consistency train<->eval | author-enforced (failures at tau~1.0 otherwise) | yes (champion proves pipeline) |
| assistant-only loss mask | universal | yes |
| packing window | 8192 | yes (speculators multipack) |
| sample shape | ONE conversation per sample; DFlash author's working recipe "used only the first turn" (1.2M samples, 6-layer, 3ep -> tau 6.3) | NO — turn-prefix rows re-supervise early turns (reshape `merged`/`first`/`last` fix this) |
| length | short single exchanges (~1k); paper extends 4K->32K via 1.6K-sample fine-tune | median 1.7k, p90 4.6k — acceptable, `short` arm tests the idiom |
| min-trainable-tokens filter | speculators --minimum-valid-tokens | missing (minor; add to prep) |

Data-scale reality from the field:
- DFlash paper ablations ran at 100K samples (short exchanges) with good
  results; headline models 800K. Nobody publishes below ~100K samples.
- DFlare scales 270K -> 2.4M samples (Qwen3-8B): tau rising the whole way,
  "likely to yield further gains" at the end. Our whole 10M-260M-token
  ladder sits BELOW their smallest measured point.
- SpecForge #465: multiple users with official script + official regen data
  collapsing to tau 1.02-1.05 — our exact rung-1 number. Author's
  resolutions: 175K samples = "poor" (one failure explained), 1.2M first-
  turn = tau 6.3, single-sample overfit case = teacher-forcing artifact.
  No one measured a curve; our ladder is that instrument.
- Recipe lead (champion-v2, filed with D-PACE/Draft-OPD): DFlare uses
  progressive gamma (4.5, +1/epoch) vs our fixed 7.

## Currency ledger (2026-10-15): one unit for "amount of data"

Canonical unit: UNIQUE SUPERVISED TOKENS (loss_mask=1, once per
conversation); conversations as the diversity co-unit. Measured by
drafter/data_audit.py. Rung 1 exact: 10,001,156 nominal -> 7,498,489
unique (25% prefix duplication) -> 5,190,561 supervised-unique
(nominal/supervised exchange 1.93x). Champion estimate (20K-row sample):
~45% supervised share -> ~115-120M supervised-unique.

Reference conversion (their samples -> estimated supervised tokens):
5K example ~1.5-3M (dead, matches our rung 1); DFlash 100K-sample
ablations ~20-50M (tau 4.9-5.6 greedy); DSpark 100K prompts <=1024
prepared ~30-60M (tau 3.41); DFlare 270K-2.4M ~55M-1B (rising
throughout). Unified picture: dead below ~5M supervised, known-good
20-60M, ours 3.79 at ~115-120M.

Orthogonality caveat: at matched supervised volume, short-exchange
corpora carry ~3-4x more distinct conversations than our long-conv
corpus. Supervised volume and conversation diversity are separate axes;
our shape is volume-heavy/diversity-light. The 130M-nominal pod rung
(~59-68M supervised) lands in the literature's known-good zone — the
right place to detect a shape deficit at matched volume.

## Mix doctrine in the references (2026-10-15)

Nobody ablates training-data mix — DFlash 5.4.1 "Training Data" is a
same-data EAGLE-3 comparison, not a mix study. What exists is ecosystem
convergence: DSpark repo / NeMo recipe / speculators demo / SpecForge all
use Open-PerfectBlend = math 39% (MetaMath 26 + OrcaMath 13), chat ~30%
(UltraChat 14 + UltraFeedback 12 + lmsys 4), IF 16% (UltraInteract 19 +
AutoIF 4), code 7% (evol-codealpaca), dedup'd (-88K). Papers use Nemotron
PT V2 + CodeAlpaca (+Step-3.5-Flash-SFT in DFlare), general blends with
explicit code share. Paper spec: max seq 3072 (4096 Coder) — our corpus
p90 4.6k exceeds it.

Implicit, never-ablated ecosystem belief: structured domains (math/code)
draft best (benchmarks agree: tau 6.5-7.8 math/code vs 4.4 chat).

Ours vs the default: chat-heavy 65% / math 12.5% / code ~6% (measured on a
3K-row sample: fences+pattern heuristic; chat 8%, terminal_agent 16%
within-subset) — code share matches PerfectBlend's 7% dedicated source;
the divergence is math (12.5% vs 39%) and chat (65% vs ~30%). Mix is
therefore an open dial our pilot can measure (natural vs PerfectBlend-
shaped vs agent-heavy at matched supervised tokens) before the 4B rung
commits.
