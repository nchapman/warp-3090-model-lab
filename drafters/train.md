# Training the DFlash2 drafter (`speculators train`)

All commands run in the **drafter venv** (`requirements-drafter.txt` in this dir).
Verifier is always the **deployed quant** (quant-aligned doctrine).

## Hardware

| machine | role | attn impl |
|---|---|---|
| DGX Spark (GB10, sm_121) | dev, regression, small rungs | `sdpa` (flex-attention Triton kernels exceed sm_121 shared memory) |
| RunPod B200/B300 (sm_10x) | big rungs (v4/champion scale) | `simple_flex_attention` |

Cross-machine reproducibility measured: same checkpoint scored 3.250 (B200) vs
3.242 (GB10) — σ ≈ 0.01 on interval-weighted acceptance. Train locally, scale
on pods, deploy on Ampere; the acceptance number is portable, **throughput
numbers are not** (measure those on 3090-class hardware for warp-3090 claims).

## Champion recipe (v4 armB — the published τ 3.443 drafter)

Data: v4 slice (100k convs → ~145k rows, 260M tok), 6-tap fp8 states,
conv-disjoint val. Init: `drafters/bootstrap_draft.py --model 4b --block-size 16`.

```bash
speculators train \
  --speculator-type dflash2 \
  --draft-attn-impl simple_flex_attention \      # sdpa on Spark
  --from-pretrained draft_init \
  --verifier-name-or-path $QUANT --data-path ds_v4 \
  --epochs 6 --lr 6e-4 --scheduler-type cosine --scheduler-warmup-ratio 0.04 \
  --per-position-loss-weight fixed-exp-decay --dflash-decay-gamma 7 --max-anchors 512 --checkpoint-freq 1 \
  --train-data-ratio 0.97 --log-freq 10 --save-path draft_v4b
```

Then the part that matters — **ladder-eval EVERY epoch and keep the best**:

```bash
for ep in 0 1 2 3 4 5; do
  bash drafters/eval_ladder.sh draft_v4b/$ep v4b_ep$ep 15 8151
done
```

Why: the shipped champion is **epoch 6**, selected at the serving point
(ep6 3.79 vs ep4 3.68 @ t=0.7, 2.3σ at n=4 each; ep2's 3.443 was the
superseded pod-era t=1.0 pick). Acceptance at t=0.7 rose monotonically
through ep6 while **greedy inverts the ranking** (ep4 peaks, 4.20 vs 4.05) —
temperature-dependent ranking is exactly why selection is t=0.7 n≥4, never
greedy. Val loss is also an unreliable selector at high LR (improves while
τ moves either way) — select by ladder, never by val loss or finality. (`checkpoint-freq 1` is
asserted by convention: with freq=2 an odd-epoch run never saves its final epoch.)

- armA variant (serve-down-robust): `--epochs 3 --lr 1e-4` (constant), same
  everything else. armA trades 0.11 native τ for zero b8 serve-down penalty.
- Serve-down eval: `bash drafters/eval_ladder.sh draft_v4b/1 v4b_ep2_b8 7 8151`
  (b8 = 7 spec tokens; champion serve-down τ 3.131).

## Small scale (the on-ramp; Spark or one 24GB+ GPU)

`--preset small` slice → regen → prep → extract (`--model 4b`, fp8) →:

```bash
speculators train \
  --speculator-type dflash2 --draft-attn-impl sdpa \
  --from-pretrained draft_init \
  --verifier-name-or-path $QUANT --data-path ds_small \
  --per-position-loss-weight fixed-exp-decay --dflash-decay-gamma 7 \
  --epochs 3 --lr 1e-4 --checkpoint-freq 1 \
  --train-data-ratio 0.97 --save-path draft_small
```

Expected: τ ≈ 2.18 at 2.7M tokens (v2 rung) — already beats MTP k=1 (1.78).

## MTP baseline (measure it on YOUR ladder — never trust published numbers)

```bash
# serve the quant with its native MTP head (model_mtp.safetensors ships in the
# published quant repos), then:
bash drafters/eval_ladder.sh <quant_dir> mtp_k3 native 8152
```

Re-anchored numbers on our ladder: MTP k=1 1.78 / k=2 2.32 / k=3 2.71
(the blog's 3.90 was their harness + prompts, not ours).

## Known-good / known-bad (details in experiments/drafters/)

- GOOD: fp8 states (needs `patches/speculators_fp8_load_cast.patch` in this dir); epoch-2
  early stop; per-epoch ladder evals; conv-disjoint val splits.
- DEAD ENDS (do not retry): blend tilts (v5 arm D: null — math itself declined),
  selector-loss push α=4 (v5 arm E: −0.3 to −0.4, recall@16 collapses), block-16
  vs block-8 recipe flip at 29M (parity, not a lever), online states-on-the-fly
  extraction on quant verifiers (vLLM states are corrupt on W4A16).
- ETA discipline: extrapolate from MEASURED spans of the previous identical unit
  (epoch boundaries), never from first-principles rates.

## Per-position loss weighting: default vs D-PACE (opt-in)

`--per-position-loss-weight` selects the intra-block loss weighting:

- **`fixed-exp-decay` (default, champion lineage)** — static `exp(-pos/γ)`
  with `--dflash-decay-gamma 7`. This is the documented recipe; our scripts
  pin it explicitly.
- **`dpace` (experimental, opt-in)** — D-PACE weights (arXiv:2605.18810):
  dynamic per-position weights from a differentiable acceptance proxy,
  `--dpace-alpha 0.5`. Requires `--loss-fn ce`.

**FOOTGUN: the speculators 0.8.1 CLI defaults dflash to `dpace`.** Any
command that omits the flag silently trains with D-PACE — the documented
γ=7 static decay has not been what an unpinned command runs since the
library upgrade (rung 1 trained dpace-weights without anyone choosing it).
Always pin `--per-position-loss-weight` explicitly; the only 0.8B A/B so far
(3K convs, dead-zone scale, τ 1.081 vs 1.090 @ t=0.7 n=1) is uninformative —
the question is open at real scale.

## Draft-OPD post-training (experimental, opt-in)

On-policy distillation from verification boundaries (arXiv:2605.29343) —
collection → replay dataset → post-train, all scripts under `drafters/opd_*`,
trainer patch `patches/speculators_draft_opd.patch` in this dir (inert without opd_
columns in the data — the default SFT path is untouched). Full recipe and
verified semantics: `experiments/drafters/draft_opd.md`.
