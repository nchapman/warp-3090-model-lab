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

35B MoE generation (Qwen3.6 / Ornith-1.5, B200 pod 10-17, same recipe;
separate table per cross-generation convention):

| model | weightedKL | probe-v4 dPPL | tail p99.9 KL | artifact |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | **0.111** | +1.33% | 18.06 | `nchapman/Qwen3.6-35B-A3B-W4A16-GPTQ` |
| Ornith-1.5-35B-A3B | **0.233** | −3.04% | 8.24 | `nchapman/Ornith-1.5-35B-A3B-W4A16-GPTQ` |

Both in family band on chat/math (where blend weight lives); both carry the
agentic-domain tail (conv_agent 1.55/1.33, swe 0.85/1.26). Ornith's 2x
weightedKL is mid-range domains + weaker base geometry (base NLL 2.38 vs
1.39). Guard-set dPPL unmeasured (pod terminated; >90GB doesn't fit GB10).

Calibration ablations (4B, probe-v4): cyankiwi 0.1340 · ncblend 0.1273 ·
cascade-v1 **0.1087** · cascade-v2 0.1120 · cascade-v3 0.1123 (recovered
2026-10-17 from cascade_v3_4b.log; both deliberate agentic re-mixes lost to
the untouched card-proportional mix) · duo AWQ+GPTQ @512×384 0.1209
(loses — agentic domains starved at 384 tokens). GPTQ beats the library-default
512×384 budget at 999×4096 everywhere it was measured.

PerfectBlend ablation (4B, 2026-10-17, GB10): the ecosystem-default blend
(mlabonne/open-perfectblend, same 999×4096 regime, role-mapped through the
model template) scored **0.1432** vs cascade-v1 0.1087 (+32%) — worst of all
GPTQ one-shots, re-anchored same-day (champion reproduces 0.1087 exactly on
this box). Uniformly worse, not just tails: chat 0.066/0.044, math
0.056/0.036, conv_agent 2.13/1.60. Only dPPL is close (+3.16 vs +3.35).
Verdict: keep cascade-v1; the general blend's missing agentic/tool coverage
is measurable, and nobody ablates this because nobody measures.

### Rollout agreement — traj-v1 ruler (2026-10-17, GB10, greedy 32-token)

`quant/traj_probe.py` on the 4B champion vs bf16 base, frozen `traj-v1.jsonl`
(301 rows; both models greedy from identical prompts, rollout = ids through
first EOS). The compounding-regime instrument: teacher-forced KL scores the
quant along the *base's* tokens; this measures what happens once the quant's
own drift feeds back. Unsloth Divergence-300-style exact@32.

| | agree@1 | agree@8 | agree@16 | exact@32 | prefix p50/p10 |
|---|---|---|---|---|---|
| **OVERALL (301)** | 0.827 | 0.542 | 0.439 | **0.239** | 10 / 0 |
| swe (25) | 0.880 | 0.000 | 0.000 | **0.000** | 2 / 0 |
| terminal_agent (50) | 0.880 | 0.360 | 0.160 | 0.080 | 4 / 0 |
| conv_agent (25) | 0.920 | 0.480 | 0.200 | 0.040 | 5 / 2 |
| ML belebele (36) | **0.444** | 0.250 | 0.194 | 0.083 | 0 / 0 |
| math (60) | 0.850 | 0.633 | 0.533 | 0.333 | 20 / 0 |
| chat (40) | 0.875 | 0.775 | 0.700 | 0.350 | 27 / 0 |
| chat_long (40) | 0.850 | 0.775 | 0.700 | 0.400 | 23 / 0 |
| instruction_following (25) | 0.960 | 0.960 | 0.960 | 0.560 | 32 / 19 |

Findings: **(1) the agentic cliff is real at 4-bit GPTQ.** swe agrees for
~2 tokens then never (exact@32 = 0/25, agree@8 = 0) — while its agree@1
(0.88) looks healthy. The conv_agent KL wall (1.60–2.20 vs chat 0.066 in
probe-v4) predicted this concentration; teacher-forced KL understates it
because the wall never compounds downstream. **(2) ML diverges at token 1**
(agree@1 0.44, median prefix 0 — worse immediately than any agentic domain;
the 10% ML weight in weightedKL hides this). **(3) even chat pays the
compounding tax**: exact@32 0.35–0.40 despite KL 0.066. **(4) IF is the
healthiest** (rigid response openings dominate the first 32 tokens).
Cost: 11 min for both sides (301 rows) on GB10 — cheap enough to run per
contender. Per-row dump (texts, prefixes) at `slices/traj-champ4b-rows.jsonl`
(local scratch, for future paired bootstrap). Sampling note: greedy ignores
the champion's stamped SERVING_PARAMS sampling flags (`do_sample=False`),
confirmed by the harness's invalid-generation-flags warning in the run log.

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

## 0.8B DFlash2 smoke (2026-10-16, GB10) — flow verified, scale floor mapped

All numbers general-chat τ @ t=0.7, n=1 (smoke-grade: single draws; treat as
floor-mapping, not selection data). Dataset = banked 0.8B regen finals
(on-policy, SERVING_PARAMS), same prep/extract/bootstrap throughout.

| supervised tokens (convs) | arm | τ |
|---|---|---|
| ~0.7M (320) | dpace-default | 1.024 |
| ~3.5M (3,000) | static γ7 | 1.081 |
| ~3.5M (3,000) | dpace α0.5 | 1.090 |
| — | MTP k=15 control (untrained head) | **2.868** |

- **Dead zone at 0.8B extends to ≥3.5M supervised** — 5× data moved τ +0.06.
  The 3.5M D-PACE A/B is therefore uninformative for the method (both arms at
  the acceptance floor; loss shaping cannot express there). 0.7M→1.024 and
  3.5M→1.08 join the measured anchors.
- **Library footgun found**: speculators 0.8.1 defaults dflash to
  `dpace` — the documented γ=7 static recipe was silently overridden in any
  unpinned command (rung 1 included). All our scripts now pin
  `--per-position-loss-weight` explicitly (champion lineage = default).
- **Draft-OPD rig verified end-to-end on real data** (collector hook on the
  true dflash verify boundary, echo-attributed assembly, replay training
  through the patched trainer with opd metrics in val logs; standard loss
  fully masked at opd positions). Boundary instrument cross-check: mean
  accepted+1 **1.100** vs ladder τ 1.081 same checkpoint. See
  `docs/draft_opd.md` for semantics + caveats.
- A 15K-conv A/B (≈17M supervised) was started and then killed: superseded
  by the full-scale 0.8B round on the Blackwell pod (RTX PRO 6000, same
  Blackwell family as GB10).

## 35B dual-quant mission (2026-10-17/18, RunPod B200 se3u5xacjmhp7j)

**Qwen3.6-35B-A3B-W4A16-GPTQ — DONE + SHIPPED.** Oneshot GPTQ (cascade-v1
999×4096 calib), run `quant-35b-35b.20261017T064500Z` (4h55m, cuDNN-SDPA-off
+ expandable_segments). Verified on pod (pack-quantized GPTQ W4A16 g128;
vision tower/MTP/shared-expert/linear-attn bf16; smoke gen OK), landed
locally `quants/Qwen3.6-35B-A3B-W4A16-GPTQ` (21G, 12 files, sha256sum -c OK).
Card live: nchapman/Qwen3.6-35B-A3B-W4A16-GPTQ (sha256 in card; shards
3fa42b79…, 1631b73b…, 68b2cabe…). Probe numbers pending (see below).

**MISSION COMPLETE 17:05Z 10-17.** Both 35B MoE W4A16 GPTQ quants shipped.

| | Qwen3.6-35B-A3B | Ornith-1.5-35B-A3B |
|---|---|---|
| run | quant-35b.20261017T065050Z.2036 | quant-35b-ornith.20261017T114909Z.13535 |
| wall | 4h55m (06:50-11:45Z) | 4h48m (11:49-16:37Z) |
| probe mean KL | 0.269 | 0.304 |
| probe weighted KL | 0.111 | 0.233 |
| probe p99.9 KL | 18.06 | 8.24 |
| probe dPPL | +1.33% (NLL 1.3895->1.4080) | -3.04% (NLL 2.3815->2.3090) |
| artifact | quants/Qwen3.6-35B-A3B-W4A16-GPTQ (12 files, sha OK) | quants/Ornith-1.5-35B-A3B-W4A16-GPTQ (12 files, 21G, sha OK) |
| card | huggingface.co/nchapman/Qwen3.6-35B-A3B-W4A16-GPTQ | huggingface.co/nchapman/Ornith-1.5-35B-A3B-W4A16-GPTQ |

Same recipe both: W4A16 GPTQ g128 oneshot, cascade-v1 999x4096 calib
(sha-pinned), snapshots Qwen 995ad96eacd9 / Ornith 10fbf86fed7e. Ornith
verify: 3 shards 20.68 GiB, compressed-tensors pack-quantized, vision tower
bf16-untouched (explicit ignore list), MTP tensors 785/785 copied, smoke
generate OK. Probe set: data/probe-v4.jsonl, 378 rows, KL_ROWS_CAP=400.
Same architecture both (qwen3_5_moe hybrid incl. gated-delta-rule; both logs
show the identical benign FLA-fallback warning) and identical ~7 min/subgraph
pace — 41 subgraphs each. Cost: pod 06:20-17:00Z = 10.7h x $7.99/h
~= **$85 total for both**. Pod se3u5xacjmhp7j terminated 17:02Z (204) after
both artifacts verified on local disk.
