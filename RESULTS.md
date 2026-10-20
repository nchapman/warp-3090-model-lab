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

### Capability battery — lm-eval cliff detection (2026-10-17/18, RunPod A100 SXM 80GB quant-lab-a100, greedy t=0)

> **Methodology caveat (2026-10-19):** ifeval/gpqa numbers in this
> section were measured under the think-cap confound (completion mode,
> 4096-token cap, mode flips dominating the delta) — decomposed in
> `results/battery-think-cap-confound-20261019.md`.
> **FINAL STANDING (2026-10-20):** numbers below are REGIME measurements
> (comp-4096), never deployment claims. Deployment-mode champion anchors:
> chat-think −7.6 / nothink −1.7 / paired-complete −3.4. Full claim-by-claim
> adjudication: `results/battery-protocol-audit-20261020.md` — read that
> before quoting anything from this section.

> Provenance corrected 2026-10-18: run logs carry `/root/.cache/huggingface`
> paths — this battery ran on the A100 pod, not GB10 (mis-labeled when the
> synced run dir was written up). Timing matters operationally: 23 min/side
> there vs ~6.5 h/side on GB10 (bandwidth-starved, 13.6 s/prompt measured
> 2026-10-18) — capability batteries are pod jobs.

`quant/capability_battery.sh battery-champion` — 4B champion quant
(nchapman/Qwen3.5-4B-W4A16-GPTQ) vs bf16 base (Qwen/Qwen3.5-4B), both
served via vLLM, lm-eval `--gen_kwargs temperature=0`, paired per-doc
bootstrap CI, cliff threshold 5pt. THINK_STRIP patch applied (strips think
blocks before scoring). Full run, no doc cap. Run:
`runs/battery-champion.20261017T234216Z.12088` (report.txt +
battery-report.json + both sides' raw lm-eval JSONs).

| task | metric | n | base | quant | Δpt | 95% CI | cliff |
|---|---|---|---|---|---|---|---|
| ifeval | prompt_level_strict_acc | 541 | 47.0 | 39.0 | **-7.9** | **[-11.5, -4.4]** | **YES** |
| gpqa_diamond_cot_zeroshot | exact_match (flexible) | 198 | 47.5 | 41.9 | -5.6 | [-13.6, +2.5] | |
| humaneval_plus | pass@1 | 164 | 53.7 | 50.0 | -3.7 | [-9.1, +1.8] | |
| mbpp_plus | pass_at_1 | 378 | 47.1 | 47.1 | +0.0 | [-5.8, +5.8] | |
| minerva_math500 | exact_match | 500 | 24.8 | 23.6 | -1.2 | [-5.6, +3.2] | |

Findings: **(1) ifeval is the only cliff — CI entirely below zero** — and
it corroborates the traj-v1 agentic cliff from an independent single-turn
instrument. **(2) Math, code, and knowledge are clean**: mbpp exactly 0.0,
minerva -1.2, humaneval -3.7 (all CI-over-zero). W4A16 GPTQ damage
concentrates in instruction-adherence regimes, not capabilities. **(3)
Cross-instrument note**: traj-v1 IF rollouts were the *healthiest*
(agree@1 0.96 — rigid openings dominate 32 tokens) while battery ifeval
full-response strict is the *worst* cliff. Both are right: the quant
tracks the base through rigid openings, then full-response adherence
degrades past them. Short-rollout agreement ≠ full-response adherence —
compare directions across instruments, never averages (standing doctrine).
**(4) gpqa -5.6 is suggestive but n=198-noisy** (CI spans zero; small-n
single-draw — treat as unproven). Tooling pins that this battery needs:
lm_eval==0.4.13 + antlr4==4.11 (THINK_STRIP patch is layout-coupled to the
lm-eval source version; a stale-but-importable lm_eval runs unpatched and
silently keeps think blocks). Metric/filter provenance: lm-eval logs the
same bare metric under multiple filters — the report pins the filter
(e.g. `exact_match,flexible-extract`), else last-logged wins by luck.

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

## GDN div-64 fix + calib-authorship A/B (2026-10-18, RunPod A100 SXM 80GB quant-lab-a100 ark6gc32cp9j8t)

**Root cause of the earlier GDN-div64 crashes: llama.cpp convert wrote
GDN in_proj as ONE fused [fused_N, K] tensor; fused_N = 3168 not div-64
→ llmcompressor group axis misaligned → per-group Hessian inversion
exploded (early layers dominated; conv_agent dppl +1.1%).** Fix =
`_gptq_ignore`-style module ignore for fused GDN in_proj in
quant/quantize.py build_recipe, letting the GDN keepouts cover it.
Both post-fix runs receipt `assert_packed_div64: OK` and the quantized
state_dict is per-group verified. Added `assert_preflight_div64` guard
to quantize.py main() (logs `PREFLIGHT_DIV64_OK`; mock-tested both
sides: N=2032 caught, N=2048 passes) — synced to pod.

Four arms, probe-v4 (378 rows, base NLL 1.402), champion = control:

| arm | mean KL | dPPL | qNLL |
|---|---|---|---|
| champion (cascade-v1, fused-GDN skip) | 0.1133 | +3.33% | 1.4487 |
| win-tail (top-25% windows) | 0.1170 | +1.79% | 1.4271 |
| **gdnfix-head (cascade-v1, author=head)** | 0.1170 | **+2.33%** | 1.4347 |
| **gdnfix-qwenfinals (qwen-authored finals)** | 0.1171 | **+0.98%** | **1.4157** |

- **Calib authorship halves dPPL at flat KL** (3.33 → 0.98%). Swe
  15.8 → 11.9; conv_agent +1.1 → **−3.2** (quant now IMPROVES agent
  domains). gdnfix-qwenfinals is the best qNLL we have measured on
  probe-v4.
- qwenfinals calib sha a4f04b7d06686478 (999 rows, Qwen-authored
  finals, truncated 4096). Artifacts local:
  ~/Code/llm-compressor/Qwen3.5-4B-W4A16-GPTQ-gdnfix-{head,qwenfinals}.
- Oddity for the record: post-fix arms sit at KL 0.117x vs champion
  0.1133 — the old fused-GDN skip damaged GDN layers LESS in KL terms
  while wrecking dPPL. KL alone under-ranks GDN damage; dPPL/NLL carry it.
- Champion remains the shipped artifact (control, never re-quantized);
  authorship result is a calib-recipe finding for the NEXT ship.

### Calib composition audit (2026-10-18, GB10 — what the 4096 window feeds GPTQ)

Reading the data, not just the sha: contexts are byte-identical between
v1 and qwenfinals (999/999 matched by context-hash; out-of-order writes
had made a naive zip mis-align and look like rewritten prompts — join by
hash, not line). Every final regenerated; both files are think-heavy by
construction, so the A/B is a clean authorship delta, not a format delta:

| | cascade-v1 | qwenfinals |
|---|---|---|
| finals with `<think>` | 923/999 (92%) | 998/999 (100%) |
| think-share of final tokens | 81% | 83% |
| final tokens (mean / p50) | 5,108 / 1,833 | 4,904 / 2,262 |

What GPTQ actually sees (4096-token window, 120-row sample, tokens):

| | context (shared) | think | answer | rows w/ >16 answer toks |
|---|---|---|---|---|
| v1 | 31% (796) | 52% (1,338) | 17% (447) | 70/120 |
| qf | 31% (835) | 59% (1,620) | 10% (269) | 54/120 |

- The authorship gain (+3.33 → +0.98 dPPL) is carried mostly by reasoning-
  token statistics, with answer share shrinking 17→10% — "Qwen-authored
  finals" is in practice "Qwen-authored reasoning". qf is the best probe
  arm we have measured; the composition shift is its likely mechanism, not
  a defect (the GGUF short-rows inversion is an instrument that cannot see
  the reasoning regime — short-context, ≤2048 tok).
- **Test for the release (characterization, not repair):** matched arms —
  same Qwen-authored finals, think-stripped/capped vs as-is, answer-share
  held constant. Directly measures whether reasoning-token hessian
  coverage is what protects this hybrid arch (GDN state compounds over
  length; think phase is the longest-horizon regime). No ecosystem calib
  covers think tokens at all (imatrix: plain text; DSpark: ≤1024 prompts).
- **Longer window lever:** regen finals are stored full-length (≤16,384) —
  widening 4096→8192 is a rebuild flag on existing data, no re-spend.
  Cost is sublinear in practice: only the ~quarter of rows over 4096 grow
  (row p90 ≈ 14–17K), so ≈ +30–50% calib tokens; Hessians are K×K
  (seq-len independent), memory flat. Probe MAX_LEN must widen in step
  or we calibrate long and measure short (same blindness as GGUF).

### Capability battery — gdnfix arms + champion cross-host control (2026-10-18, RunPod A100 SXM 80GB, greedy t=0)

> **RETRACTED (2026-10-20) — two of the three findings below.** The
> "~6pt ifeval cost of the exclusion" and the "keep in_proj IN /
> spend bits instead" implication are cross-protocol artifacts (era-1
> mode flips + era-3 cross-run mode mixing — the champion readings
> compared here are completion-era, and the whole story failed
> matched deployment-mode replication). Verdict: exclusion costs ≈0 vs
> champion (v1rebuild −6.5 vs −7.6, n=541) and is RETAINED.
> "Probe wins do not carry" is WEAKENED to untested: its selfgen
> ifeval cells were era-1 and were never rerun in deployment mode.
> Numbers below stand as comp-4096 regime measurements only.
> Master audit: `results/battery-protocol-audit-20261020.md`.

Same rig as the champion battery (humaneval_plus, mbpp_plus, ifeval,
minerva_math500, gpqa_diamond; full n; think-strip; lm-eval 0.4.13;
GPU_MEM_UTIL=0.9). All quant sides below ran on the SECOND A100 host
(216.81.245.125; artifacts rsynced 30MB/s, md5-verified 20/20 vs local
manifests — the exact probed artifacts, not rebuilds); base side = the
champion battery's base raws (old pod). **Instrument control:** champion
re-run on the new host reproduces its old-instance paired deltas to
≤0.1pt on all five tasks — cross-host pairing validated, run-to-run
variance negligible at these n.

Δ vs base (pt, 95% CI):

| task (n) | champion old pod | champion xhost | gdnfix-head | gdnfix-qf |
|---|---|---|---|---|
| gpqa (198) | −5.6 | −5.6 | −13.1 [−21.2,−5.1] ⚠️ | −7.1 [−15.2,+1.0] |
| humaneval (164) | −3.7 | −3.7 | −4.3 | −3.7 |
| ifeval (541) | −7.9 | −7.8 | **−13.7** [−17.6,−10.0] ⚠️ | **−14.6** [−18.3,−10.9] ⚠️ |
| mbpp (378) | 0.0 | 0.0 | −3.2 | −1.1 |
| math500 (500) | −1.2 | −1.2 | **+4.6** [−0.2,+9.4] | −3.8 [−8.6,+1.0] |

- **Pristine GDN in_proj buys NO capability bump — it costs ~6pt of
  ifeval.** Both gdnfix arms (different calibs, same patched quantize.py
  with the GDN exclusion) lose ~6pt more IF than champion on both
  instances' champion readings. The common factor is the exclusion
  itself: mixed-precision error redistribution, not calib. The expected
  "natural bump" measured −6pt instead.
- **Probe wins do not carry to capability.** qf = best qNLL ever measured
  (+0.98 dPPL) yet sits at the worst ifeval cell of the three; head's
  only positive anywhere is math500 +4.6 (CI borderline). Probe↔battery
  ordering is instrument-specific — record both, trust neither alone.
- **ifeval cliff is recipe-class:** champion loses 7.8–7.9pt at W4A16
  regardless of host; the GDN exclusion deepens it to ~14. Next-ship
  implication: keep in_proj IN the quantization (fused, div-64-safe) and
  spend bits instead — W8 on sensitive modules / higher group count —
  rather than excluding modules.
- **Champion stays the shipped artifact** (probe control, battery leader
  on 4/5 cells). Artifacts local: ~/Code/llm-compressor (gdnfix dirs
  + .md5 manifests); paired reports: runs/battery-*/paired-report.json
  (gdnfix-head.20261018T221434Z.3679, gdnfix-qwenfinals.20261018T224435Z.16780,
  champion-xhost.20261018T231336Z.26704).

## GGUF local showdown (2026-10-18, GB10, llama.cpp b-cuda)

Question: how do our W4A16 artifacts fare vs unsloth GGUF quants in
llama.cpp-land? Instruments (all local, `-c 16384 -ngl 99`, texts
rendered from data/probe-v4.jsonl via chat template):

- **Aggregate all-domains ppl is INVALID on this hybrid arch.** Exact-
  weights f16 scores WORSE than every quant (4.7150 > Q8 4.6708 > champion-
  dequant 4.5404 > UD-Q4_K_XL 4.5136 > Q4_K_M 4.3206) — a length-dependent
  drift in the exact-weights path masked by quant shrinkage. Never rank
  quants on aggregate llama.cpp ppl here.
- **Short-rows (194 rows ≤2048 tok) is the honest instrument:**

| file | PPL | Δ vs f16 |
|---|---|---|
| base f16 | 3.5138 | — |
| Q4_K_M (imatrix) | 3.5319 | +0.51% |
| UD-Q4_K_XL (imatrix) | 3.5372 | +0.67% |
| gdnfix-head dequant | 3.5871 | +2.09% |
| champion dequant | 3.5913 | +2.20% |
| gdnfix-qwenfinals dequant | 3.5982 | +2.40% |

Findings: **(1)** unsloth's "quality-max" UD blend is a statistical tie
with plain Q4_K_M on short rows — the imatrix+K-quant base does the work,
not the dynamic blend. **(2)** our W4A16-g128-symmetric carries a ~+2%
ppl floor in llama.cpp regardless of calib arm — that's recipe-class vs
imatrix'd K-quants, not the GDN bug. **(3)** the GDN fix is GGUF-neutral
on short rows (+2.20→+2.09): GDN damage compounds with length, which
short rows dilute and the aggregate instrument drowns — probe (teacher-
forced, reference runtime) remains the instrument for GDN. **(4)**
qwenfinals inverts across instruments (probe-best +0.98 dPPL, GGUF
+2.40 worst): see composition audit — a think-heavy calib optimizes the
reasoning regime the short instrument cannot see. **(5)** matching UD
in GGUF-land means shipping our own imatrix'd K-quant — proper-release
scope, not a W4A16 patch.

Forensics that got here (kept to prevent re-derivation): recurrent_layers
GGUF field is inert (loader never reads it); attn_norm +1.0 shift is
llama.cpp convention in BOTH files; big weights bit-exact vs HF; GPU==CPU
ppl identical (2.6324) → no kernel bug; anomaly is length-dependent in
the exact-weights path. Champion W4A16→f16 dequant: ~/gguf-eval/
dequant_champion.py (two-pass caches; unpack int32→4bit × g128 symmetric;
faithful to +0.5% tensor std vs base).

Ops notes: dequantized-W4A16 GGUFs built via convert_hf_to_gguf.py
(--outtype f16 --no-mtp) on the f16 HF render. GB10 doctrine addendum:
battery-grade vLLM serving and llama.cpp sweeps serialize on this box
(measured 18.5 vs 13.6 s/prompt contended vs solo).

## Same-box probe pass + attnbf16 precision arm (2026-10-19, H200 yav18d8q81klrr)

Full table, per-domain rows, and receipts: `results/probe-h200-20261019.md`
(row dumps `runs/probe_rows_h200-*.jsonl`). Headlines (current artifacts,
one box, base NLL 1.3945 every run):

| quant | dPPL | wKL |
|---|---|---|
| champion | +3.20% | **0.1087** |
| v1rebuild (champion recipe, current script) | +3.36% | 0.1094 |
| **attnbf16** (cascade-v1, BF16 attn projections) | **+1.93%** | **0.0733** |
| qwenfinals@4096 | +2.99% | 0.1128 |
| if3x@4096 | +1.07% | 0.1106 |
| 4bself@4096 | +0.48% | 0.1125 |

- **The SGLang/Marlin-era script changes cost nothing measurable**:
  v1rebuild (champion calib on the current script, receipt
  runs/quant-4b.20261019T191105Z.37151) = +3.36 / 0.1094 vs champion
  +3.20 / 0.1087 — 0.16pt dPPL, inside the 0.8–1.7pt rebuild-drift
  band; wKL flat. The current-script nemotron-calib rebuild is the
  probe-equivalent, SGLang-safe successor of the champion artifact.

- **Projection bitrate carries ~40% of dPPL / 33% of wKL** (attnbf16 vs
  champion, uniform across all seven probe domains). Cost: +1.9GB
  (+51%) — attention projections are ~32% of params in this hybrid.
- **Rebuild drift measured** (if3x/4bself, same box/calib/recipe):
  dPPL moves 0.8–1.7pt, wKL ±0.002 → dPPL margins <2pt between single
  builds are not recipe evidence; four calib variants wKL-equivalent.
- **Precision-era caveat**: the shipped champion is a PRE-d281deb build
  (in_proj_a/b W4-packed; vLLM-servable, NOT SGLang/Marlin-safe); all
  post-10-18 builds keep a/b BF16 (3.9M params, 0.1%). The rebuild arms'
  layout is IDENTICAL to the 10-18 gdnfix arms whose battery read
  −6pt ifeval vs champion (see GDN div-64 section above): the −6pt was
  attributed to the a/b exclusion with calib confounded — `v1rebuild`
  (v1 calib + a/b BF16, same day) is the clean third arm that split
  runs. **RETRACTED (2026-10-20):** the −6pt attribution was a
  cross-protocol artifact, not exclusion cost — matched deployment-mode
  arms exonerated the exclusion (see 10-20 verdict below and
  `results/battery-protocol-audit-20261020.md`); **Probe optimism is NOT
  battery evidence** — but note the 10-18 "divergence" citation (qf:
  best qNLL, worst ifeval) was an era-1 completion cell and is UNTESTED
  in deployment mode (selfgen battery never ran); the doctrine stands
  on principle, not on that datapoint.

## 2026-10-20 — VERDICT: rebuild "damage" was a cross-protocol artifact; exclusion retained

Closed with three targeted arms (H200, n=541 paired, chat-template
serving = deployment mode; receipts runs/verdict-20261019/):

- Champion, chat+think 16k: **−7.6** [−10.5,−3.5] (fixedR; chat16k −7.0).
- v1rebuild AS BUILT (stock template, SERVING_PARAMS genconfig):
  **−6.5** [−10.2,−3.0] — champion-class parity in deployment mode.
- v1rebuild + champion template/genconfig (metadata swap, same weights):
  **−5.9** [−9.6,−2.4] — metadata worth 0.5pt; template/genconfig exonerated.
- No-think: v1rebuild-release **−3.7** [−6.7,−0.7] vs champion −1.7
  [−4.3,+0.9] (CIs overlap; ≤2pt possibly real — consistent with probe
  dPPL 3.36 vs 3.20).

Why the scare happened: the −7.6-vs−17.7 headline compared CHAT-MODE
champion against RAW-COMPLETION v1rebuild. Matched completion-16k puts
champion itself at **−11.8** [−15.7,−7.8] (comp16k) — completion regime
hammers everyone (base strict 51.8, 58% trunc; v1rebuild −17.7 there is
~6pt worse, CIs brushing, in a regime nothing is served in). The A100
gdnfix "cliff" (−13.7/−14.6) was the same protocol class; the
attribution in the previous section ("−6pt = a/b exclusion") is
retracted — it was protocol, not exclusion.

Decisions:
- **GDN in_proj_a/b BF16 exclusion RETAINED** (kernel-compat value at
  zero measured deployment-mode cost).
- **v1rebuild is the probe-equivalent, SGLang-safe successor** of the
  champion artifact; ship-decision pending only the standing per-model
  regen doctrine.
- Tooling: capability_battery.sh stamps protocol identity
  (chat_template/think_args/max_gen_toks/max_model_len) into events;
  battery_report.py hard-fails cross-protocol pairing (legacy runs
  without events skip the guard with a warning). arm2 (4bself full
  battery) was killed mid-run — its question (calib lottery vs
  systematic exclusion cost) is moot under this verdict.

## 2026-10-20 — Calib-authorship single-test gate: PARKED (teacher-authored finals do not carry to deployment mode)

NAMING (fixed 2026-10-20b): gdnfix-qwenfinals' calib finals were
authored by qwen/qwen3.8-flash via OpenRouter (a different, larger
Qwen-family TEACHER — see regen_calib_finals.py --model default) —
NOT self-generated by the model being quantized. "Selfgen" properly
refers to the 4bself arm (Qwen3.5-4B base regenerates its own finals,
regen_calib_self.sh), which remains unmeasured in deployment mode.
This entry's original "selfgen gate" framing was wrong; numbers
unchanged, only the label corrected.

One cheap decisive arm instead of a battery (pre-registered rule:
authorship effect real only if qf ≥5pt better than v1rebuild with CI
separation).
Fresh RunPod 4090 (gordhp95nvpnzt, US-NC-1, secure) — third host class
after GB10/A100/H200; one box, one fresh base, both quants as-built
(qf stock template, v1rebuild froggeric+champion genconfig — ≤0.5pt
known metadata delta noted). Protocol: ifeval full 541, chat16k-think
(chat_template=1, think on, max_model_len 20480, gen 16384, stamped in
BATTERY_CONFIG events; guard-clean pairing).

| arm | ifeval Δ (n=541) | 95% CI | paired-complete | trunc@cap |
|---|---|---|---|---|
| gdnfix-qwenfinals (qwen3.8-flash-authored finals) | **−8.7** | [−12.6, −4.8] | −0.9 [−4.5, +1.8] | 70% |
| v1rebuild (anchor) | **−7.4** | [−11.3, −3.7] | +0.0 [−2.6, +2.6] | 69% |

(base 51.9; both legs same base run — no cross-run pairing.)

**Gate decision: PARK.** qf vs v1rebuild is a NULL: paired McNemar
exact p=0.57 (53 vs 60 discordant docs of 541, same-base pairing) —
statistically indistinguishable. Teacher-authored calib finals did
NOT carry their probe advantage into deployment-mode ifeval
(probe dPPL 0.98 vs 3.33 champion-era was real; deployment flat).
Calib-authorship swaps are a probe-only screen — dPPL optimization
without deployment transfer, never a shipping argument. No wider
battery warranted on this evidence.

Bonus results:
- **v1rebuild anchor holds on a third host class**: −7.4 (4090) vs
  −6.5 (H200) vs champion −7.6 (H200) — CIs overlap throughout;
  instrument cross-host stability confirmed yet again.
- **No probe↔battery inversion on this pair**: qf probe ≈ champion
  (2.99 vs 3.20) and battery ≈ champion-class too. The 10-18
  "inversion" claim (audit item #6) stays untested-in-deployment-mode
  for the arms that motivated it; 4bself (probe +0.48, the strongest
  author-swap signal and the ONLY true selfgen arm) remains UNMEASURED
  in deployment mode — its artifact is stranded on the stopped H200 pod
  disk.
- pc deltas ≈ 0 for both arms, damage concentrated in cap-adjacent
  rows — same instrument signature as the fixed battery.

Receipts: runs/battery-selfgen-v-{base,qf,v1r}.* (events + raws,
protocol-stamped), runs/selfgen-gate-20261020/ (paired reports).
