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

Notable: nobody between the paper (800K) and the smoke example (5k)
publishes a measured requirement — the ecosystem has a quality pole and a
smoke pole and nothing in between. Our ladder brackets exactly that gap
(rung 1 = ~their smoke scale, champion = 1/8 paper scale) with measured
tau; rung 1's number is a useful public datapoint regardless of where the
knee lands.

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
