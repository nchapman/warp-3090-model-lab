# Measured results (what these scripts should reproduce)

Conventions: weightedKL = blend-weighted mean KL(base‖quant) on the frozen
probe-v4 ruler (378 rows) via `quant/probe.py`; τ = interval-weighted mean
acceptance on the 20-prompt held-out ladder (`drafter/eval_ladder.sh`; the
ladder is general-chat only — every τ here is general-chat τ), temp
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

Scaling ladder (fixed chat-heavy cascade blend, block 16, interval-weighted τ
— pod-era single draws at t=1.0; superseded by the GB10 anchor tables below,
which measure with repeats and per-temperature spreads):

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
  acceptance declines — select by ladder, always. And measure with the
  right instrument: greedy τ (t=0) is the low-variance stability check
  (spread ~0.05), shipping selection happens at the serving point
  (t=0.7) with repeats — temp-1.0 τ is heavy-tailed (0.34 spread on
  identical inputs) and cannot rank 0.1-τ differences at small n.


## GB10 anchor measurements (2026-10-14, runlib-era, fresh numbers only)

Ladder-20, interval-weighted τ, W4A16-GPTQ verifier unless noted; all runs
under `runs/` with full event trails (`scripts/tail_run.sh`).

| checkpoint | τ @ t=0 | τ @ t=0.7 (serving) | τ @ t=1.0 |
|---|---|---|---|
| epoch 1 | 3.52 | 3.28 | 2.95 |
| epoch 2 | 3.73 | 3.49 | 3.17 (n=2) |
| epoch 4 | 4.20 (n=2) | 3.68 (n=4: 3.58-3.75) | 3.27 (n=2, spread 0.34) |
| epoch 6 — shipped | 4.05 (n=2) | **3.79** (n=4: 3.71-3.85) | 3.25 (n=2) |
| epoch 6 @ bf16 base | — | 3.75 | 3.32 |
| epoch 4 @ bf16 base | 4.15 | — | — |
| native MTP k=3 | 2.99 | 2.86 | 2.66 |
| rung 1 (10M, off-policy data) | — | 1.05 (n=4, flat all epochs) | — |

Serving-point sampling is `models.py SERVING_PARAMS` (t=0.7 / top_p 0.95 /
top_k 20) — the canonical corpus-generation and eval config (stamped into
every quant's generation_config.json). Historical note: the champion's own
v1 regen predates this and sent no top_k; rung 1's data additionally
off-policy (see AGENTS.md corpus section) — its 1.05 is recorded above as
an artifact, not a dead-zone datapoint.

Conclusions:

- **Ranking is temperature-dependent.** At t=0.7 τ rises monotonically to the
  end of training — the shipped drafter is epoch 6, selected at the serving
  point (ep6 3.79 vs ep4 3.68, 2.3 sigma at n=4 each). Under greedy the
  picture inverts: epoch 4 peaks (4.20 vs 4.05). At t=1.0 they tie.
- τ at t=1.0 is heavy-tailed (0.34 spread on identical inputs); t=0.7 spread
  ~0.1, greedy ~0.05. Small-n point estimates at t>=1 cannot rank 0.1-tau
  differences.
- The shipped `nchapman/Qwen3.5-4B-DFlash2` beats native MTP by ~+1 tau at
  every temperature.

## Confirmed category profile (2026-10-14, RedHatAI speculator_benchmarks, GB10)

`drafter/bench_subsets.sh` (dataset pin 2ae86af, temp 0, n=2 draws per
config, 924 prompts, interval-weighted τ from vLLM acceptance logs):

| subset | MTP k=3 | DFlash2 (shipped) | Δ |
|---|---|---|---|
| qa | 2.94 | **3.61** | +0.67 |
| HumanEval | 3.35 | **4.85** | +1.51 |
| math_reasoning | 3.51 | **6.64** | +3.13 |
| tool_call | 3.24 | **4.65** | +1.42 |
| rag | 3.35 | **5.03** | +1.68 |
| question | 3.12 | **4.47** | +1.36 |
| writing | 3.11 | **4.47** | +1.36 |
| summarization | 3.15 | **4.31** | +1.16 |
| translation | 2.96 | **3.04** | +0.08 |
| **weighted** | **3.21** | **4.65** | **+1.43** |
Wins every subset. Numbers above are from the second measurement pass
(fixed log-window offsets — the first pass's CR-rich-log windowing bled
neighboring subsets' tail lines; conclusions identical, shifts within
run noise). Weak tail = translation
(+0.08), consistent with the chat-heavy cascade corpus; strongest = math,
code, rag.

## Pipeline reproducibility confirmation (2026-10-14)

`quant/probe.py` re-runs against the published Hub quants on frozen
`probe-v4` reproduce the published weightedKL **exactly**: 4b 0.1087, 9b
0.0889 (dppl 3.35% / 3.23%). The 27b probe OOMs on GB10 (bf16 base + quant
will not co-reside in 128GB) — its 0.1264 stands as measured at build time;
re-confirm on a pod when convenient. (GB10 has 121 GiB UNIFIED memory — see
AGENTS.md for the vLLM gpu-mem-util rule.)

Full narrative: `docs/history/dflash2_research.md`, `docs/history/calib_v2_plan.md`.
