# Measured results (what these scripts should reproduce)

Conventions: weightedKL = blend-weighted mean KL(base‖quant) on the frozen
probe-v4 ruler (378 rows) via `quant/probe.py`; τ = interval-weighted mean
acceptance on the 20-prompt held-out ladder (`drafter/eval_ladder.sh`), temp
1.0 / top_p 0.95 / top_k 20. **Cross-generation caveat: absolute numbers are
not comparable between 3.5 and 3.8 releases — keep tables per-model.**

## Quants (W4A16 GPTQ, cascade-v1 calib, 999×4096)

| model | weightedKL | guard dPPL | artifact |
|---|---|---|---|
| Qwen3.5-4B | **0.1087** | +0.76% | `nchapman/Qwen3.5-4B-W4A16-GPTQ` |
| Qwen3.5-9B | **0.0889** | +0.77% | `nchapman/Qwen3.5-9B-W4A16-GPTQ` |
| Qwen3.8-27B | 0.1264 † | −0.52% | `nchapman/Qwen3.8-27B-W4A16-GPTQ` |

† cross-generation (3.8); its conv_agent KL wall (2.20) is higher than 3.5-4B's
(1.60) — attribution is generation gap, not size.

Calibration ablations (4B, probe-v4): cyankiwi 0.1340 · ncblend 0.1273 ·
cascade-v1 **0.1087** · cascade-v2 0.1120 · duo AWQ+GPTQ @512×384 0.1209
(loses — agentic domains starved at 384 tokens). GPTQ beats the library-default
512×384 budget at 999×4096 everywhere it was measured.

## Drafters (DFlash2, quant-aligned, 4B)

Scaling ladder (fixed chat-heavy cascade blend, block 16, interval-weighted τ):

| rung | tokens | recipe | τ (b16) |
|---|---|---|---|
| v1 | 587k | 5ep 1e-4 | 1.53 |
| v2 | 2.73M | 3ep 1e-4 | 2.18 |
| v3 | 29M | 3ep 1e-4 | 2.74 |
| v4 final (armB 6ep) | 260M | 6e-4 cosine | 3.25 |
| **v4 champion (armB ep2)** | 260M | 6e-4 cosine, stop @2 | **3.443** |

Champion serve-down b8: 3.131. armA (1e-4, 3ep): 3.14 b16 / 3.18 b8 — the
serve-down-robust pick. MTP re-anchored on the same ladder: k=1 1.78, k=2 2.32,
k=3 **2.71** (the published "3.90" was their harness, not ours).

External anchors: champion 9-subset weighted 4.16 vs full MTP 3.26 (wins 8/9
subsets); premium 27B DFlash2 on our ladder 2.959 @ matched b8 — the 4B
champion beats it with a GPTQ verifier + serve-down handicaps stacked.

9B / 27B drafters: init configs built (models.py), never trained — 27B was a
measured NO-GO (bounded upside vs ~5× pod cost). Untested until run and
recorded here.

## Dead ends (measured — do not retry)

- Blend tilts on a peaked champion: NULL (v5 arm D — the math tilt's own target
  domain declined −0.23). No per-traffic routing story.
- Selector-loss push α=4: −0.3 to −0.4, recall@16 collapses to 0.536.
- AWQ/duo at matched 999×4096: OOM-prone (AWQ caches scale with tokens; GPTQ
  Hessians don't) and loses at library budget anyway.
- vLLM hidden-state extraction on W4A16: NaN tails + silently-wrong states
  (transformers-only extraction is mandatory, hardware-independent).
- Selecting checkpoints by val loss at high LR: val loss improves while
  acceptance declines — select by ladder, always. And select on the
  **greedy** ladder: temp-1.0 τ is heavy-tailed (0.34 spread observed on
  identical inputs); greedy repeat spread is ~0.05.


## GB10 anchor measurements (2026-10-14, runlib-era, fresh numbers only)

Ladder-20, interval-weighted τ, W4A16-GPTQ verifier unless noted; all runs
under `runs/` with full event trails (`scripts/tail_run.sh`).

Greedy (temp 0 — selection instrument):

| checkpoint | τ |
|---|---|
| epoch 4 | **4.19** (n=2: 4.218 / 4.171) |
| epoch 6 (end of training) | 4.07 |
| epoch 2 | 3.73 |
| epoch 1 | 2.95→3.52* |
| epoch 4 @ bf16 base | 4.15 |
| native MTP k=3 | 2.99 |

Temp 1.0 (deployment metric — heavy-tailed, ±0.2):

| checkpoint | τ |
|---|---|
| epoch 4 | 3.27 (n=2: 3.441 / 3.099) |
| epoch 6 | 3.25 (n=2) |
| epoch 6 @ bf16 base | 3.32 |
| epoch 2 | 3.17 (n=2) |
| native MTP k=3 | 2.66 |

\* epoch-1 temp-1.0 run measured 2.954 pre-`TEMP` support; greedy 3.516.

Conclusions: acceptance peaks at epoch 4 of 6 and declines by epoch 6
(training longer does not help); the shipped drafter
(`nchapman/Qwen3.5-4B-DFlash2`) is the epoch-4 checkpoint, selected on the
greedy ladder after the temp-1.0 instrument proved unable to rank
0.1-τ differences at small n.

Full narrative: `docs/history/dflash2_research.md`, `docs/history/calib_v2_plan.md`.
