# Fixed battery — champion 4B pair (H200, 2026-10-19)

The think-truncation confound retraction
(`results/battery-think-cap-confound-20261019.md`) adopted a fixed
protocol; this is its first full run. Pair: `Qwen/Qwen3.5-4B` vs
`nchapman/Qwen3.5-4B-W4A16-GPTQ` (sha d5874365…), greedy, same H200.

Run dirs (pod-synced): `runs/battery-fixedR.20261019T155447Z.10387`
(reasoning leg), `runs/battery-fixedD.20261019T154427Z.8858` (compliance
leg), `runs/battery-fixedC.20261019T164651Z.13540` (code leg).

## Protocol

| leg | tasks | protocol |
|---|---|---|
| R | ifeval, gpqa_diamond_cot_zeroshot, minerva_math500 | chat template, `enable_thinking=True`, 16k gen cap |
| D | ifeval | chat template, `enable_thinking=False`, 16k gen cap |
| C | humaneval_plus, mbpp_plus | completion, 4096 cap (unchanged — 0% think, clean) |

Template mechanics (verified by rendering locally with the 4B
tokenizer): `enable_thinking=True` prefills an OPEN `<think>\n` into the
assistant turn — thinking is forced, the model must close it;
`enable_thinking=False` prefills an empty `<think>\n\n</think>\n\n` —
direct answer forced. The battery's think-strip patch removes closed
reasoning from stored samples, so "no markers in resps" means *closed
and stripped*, not *never thought*: **0 unclosed think in every leg,
both sides** (541+541, 396+396, 500+500) — no truncation confound, no
mode ambiguity. Mode is template-forced identically on both sides, so
mode-flip damage is structurally impossible here.

## Results

| task | leg | base | quant | Δ | 95% CI | verdict |
|---|---|---|---|---|---|---|
| ifeval (compliance) | D nothink | 83.0 | 81.3 | **−1.7** | [−4.3,+0.9] | clean |
| ifeval (reasoning) | R think | 51.6 | 44.0 | **−7.6** | [−11.3,−3.9] | cliff |
| gpqa_diamond | R think | 71.2 | 60.1 | **−11.1** | [−17.2,−5.1] | **worst** |
| math500 | R think | 81.4 | 83.0 | **+1.6** | [−2.6,+5.6] | clean |
| humaneval+ | C completion | 53.0 | 50.6 | **−2.4** | [−8.5,+3.7] | clean |
| mbpp+ | C completion | 43.7 | 47.1 | **+3.4** | [−2.1,+9.0] | clean |

ifeval-think replicates the trial's T2 (−7.0 [−10.5,−3.5]); nothink
replicates T3 (−1.7) bit-for-bit. Both think-mode ifeval numbers sit at
the same magnitude as the confounded −7.0 — but the attribution is now
clean: pure within-thinking-mode damage (trial decomposition: zero
flips).

## vs the confounded campaign numbers (same pair, 4096 completion)

| task | confounded | fixed | change |
|---|---|---|---|
| ifeval | −7.0 ⛔ | −7.6 ⛔ | same magnitude, attribution fixed |
| gpqa | −4.5 ns | **−11.1 ⛔** | **truncation masked the champion's worst cliff** |
| math500 | +2.6 | +1.6 | same |
| humaneval+ | −2.4 | −2.4 | same |
| mbpp+ | +1.3 | +3.4 | same |

## Findings

1. **Damage signature: knowledge-heavy reasoning.** gpqa (−11.1) and
   ifeval-with-reasoning (−7.6) are the only cliffs; math500
   (procedural reasoning, +1.6), code (−2.4/+3.4), and compliance
   (−1.7) are clean. W4A16 hurts the model where fragile factual
   associations must survive a long reasoning chain, not where
   procedure or instruction-following carries it.
2. **Truncation flattered gpqa most**: −4.5 ns under the 4096 cap →
   −11.1 at 16k. Same mechanism the trial showed on ifeval (−7.0 →
   −11.8 completion-mode): capping cuts the chains where quant error
   compounds.
3. Thinking costs the BASE 31–32pt of ifeval strict acc (51.6 think /
   83.0 nothink) — deployment-relevant base-side fact for the 4B.
4. Instrument receipts: all deltas paired per-doc greedy, n = full task
   (541/198/500/164/378); CIs are paired bootstrap.

## Status

- These are the champion's **fixed-battery reference numbers** — the bar
  for arm re-adjudication.
- Arms (long8k / if3x / 4bself) re-quant from frozen calib + this
  battery: queued. Their campaign rejections leaned on ifeval/gpqa
  legs measured under the confounded protocol.
- 27b qwenfinals: humaneval −35.4 was the decisive clean leg —
  rejection likely stands; re-run only if 4B rankings shift materially.
