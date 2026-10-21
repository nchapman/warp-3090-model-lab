# Fixed battery — champion 4B pair (H200, 2026-10-19)

> **Postscript (2026-10-20):** numbers here are mode-consistent within
> runs and stand — but never pair them against numbers from other
> runs/eras without checking run mode (the 10-19/20 verdict found a
> chat-vs-completion cross-run mix that manufactured a false 10pt
> cliff; runs now stamp protocol and the report guard hard-fails
> mixes). Master audit: `battery-protocol-audit-20261020.md`.

The think-truncation confound retraction
(`battery-think-cap-confound-20261019.md`) adopted a fixed
protocol; this is its first full run. Pair: `Qwen/Qwen3.5-4B` vs
`nchapman/Qwen3.5-4B-W4A16-GPTQ` (sha d5874365…), greedy, same H200.

Run dirs (pod-synced): `runs/battery-fixedR.20261019T155447Z.10387`
(reasoning leg), `runs/battery-fixedD.20261019T154427Z.8858` (compliance
leg), `runs/battery-fixedC.20261019T164651Z.13540` (code leg).

## Paired-complete re-analysis of the v1 samples (2026-10-19, same 16k runs)

Scoring the SAME samples on rows where both sides closed under the cap
(`battery_report.py --gen_cap 16384`; truncation counted, never graded):

- **gpqa −11.1 → pc +0.0 [−5.2,+5.2] (n=115 of 198).** The entire gpqa
  cliff was truncation asymmetry (base 22% vs quant 36% cut), not
  reasoning damage. On prompts both sides finish, quant = base exactly
  (84.3 / 84.3).
- **ifeval-think −7.6 → pc −3.4 [−6.9,−0.9] (n=116 of 541).** Real but
  much smaller; heavy selection (only prompts where BASE finishes under
  16k — easiest tail), so treat as biased-low estimate pending v2.
- math500 +1.6 → pc +1.5 (truncation 5%/4% symmetric — unaffected).
- nothink −1.7 → pc −1.7 (n=521; unchanged).

**Reading:** the v1 "knowledge-heavy reasoning damage" signature was
mostly cap asymmetry. What remains provable at 16k: a small ifeval-think
delta plus a completion-rate cost (the quant thinks past the cap more
often — itself a deployment-relevant damage mode, now measured as a
length metric instead of silently graded). The 32k v2 rerun (champion +
arms, shared instrument) decides with far less selection pressure.

## Instrument caveat (added post-run, 2026-10-19 — token-length audit)

**Think-leg deltas still carry truncation asymmetry.** Token-count
audit of every stored response vs the cap: ifeval-think 311/541 base
(57%) vs 391/541 quant (72%) at 16k; gpqa 86/396 (22%) vs 144/396
(36%); math500 23/19 (≈4%, near-symmetric); nothink 9/17 (≈1–3%);
code 3+3/5+2 (≈1%). The quant's longer reasoning hits the cap more
often and those rows grade as zeros — so ifeval-think −7.6 and gpqa
−11.1 mix reasoning damage with cap asymmetry; nothink −1.7, math
+1.6, and the code legs are near-symmetric. Part of the gpqa
"widening" (−4.5→−11.1) may be cap asymmetry, not unmasked damage.
The earlier "0 unclosed think" marker check is blind for chat legs:
the template prefills the opening tag, so a capped generation carries
NO markers — token length vs cap is the only direct evidence.

**Consequence for re-adjudication:** champion and arms are measured
under the SAME instrument, so arm-vs-champion comparisons remain
valid; absolute think-leg deltas are "damage under a 16k cap", not
cap-free damage. Length (bigger cap / length-normalized scoring) is a
deliberately deferred revisit — see findings below.

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
and stripped*, not *never thought*: ~~**0 unclosed think in every leg,
both sides** (541+541, 396+396, 500+500) — no truncation confound~~
**[WRONG (2026-10-19, token-length audit): this marker check is blind
for chat legs — the template prefills the opening `<think>`, so a
capped generation carries NO markers and counts as "closed". Actual
at-cap rates: ifeval-think 57%/72%, gpqa 22%/36%, math 5%/4%. See
"Instrument caveat" below and the re-analysis above. The zero-flip /
mode-forcing statements stand.]** No mode ambiguity. Mode is
template-forced identically on both sides, so
mode-flip damage is structurally impossible here.

## Results

| task | leg | base | quant | Δ | 95% CI | verdict |
|---|---|---|---|---|---|---|
| ifeval (compliance) | D nothink | 83.0 | 81.3 | **−1.7** | [−4.3,+0.9] | clean |
| ifeval (reasoning) | R think | 51.6 | 44.0 | **−7.6** | [−11.3,−3.9] | ~~cliff~~ [pc −3.4] |
| gpqa_diamond | R think | 71.2 | 60.1 | **−11.1** | [−17.2,−5.1] | ~~**worst**~~ [pc **+0.0** — artifact] |
| math500 | R think | 81.4 | 83.0 | **+1.6** | [−2.6,+5.6] | clean |
| humaneval+ | C completion | 53.0 | 50.6 | **−2.4** | [−8.5,+3.7] | clean |
| mbpp+ | C completion | 43.7 | 47.1 | **+3.4** | [−2.1,+9.0] | clean |

ifeval-think replicates the trial's T2 (−7.0 [−10.5,−3.5]); nothink
replicates T3 (−1.7) bit-for-bit. Both think-mode ifeval numbers sit at
the same magnitude as the confounded −7.0 — ~~but the attribution is now
clean: pure within-thinking-mode damage (trial decomposition: zero
flips)~~ **[INCORRECTLY ASSUMED: the attribution was NOT clean — both
think-mode numbers carry cap-truncation asymmetry (57%/72% at 16k).
Paired-complete: −3.4 [−6.9,−0.9].]**

## vs the confounded campaign numbers (same pair, 4096 completion)

| task | confounded | fixed | change |
|---|---|---|---|
| ifeval | −7.0 ⛔ | −7.6 ⛔ | same magnitude, attribution fixed |
| gpqa | −4.5 ns | **−11.1 ⛔** | ~~**truncation masked the champion's worst cliff**~~ [wrong — the 16k run re-created the same artifact one cap up; pc **+0.0**] |
| math500 | +2.6 | +1.6 | same |
| humaneval+ | −2.4 | −2.4 | same |
| mbpp+ | +1.3 | +3.4 | same |

## Findings

1. ~~**Damage signature: knowledge-heavy reasoning.** gpqa (−11.1) and
   ifeval-with-reasoning (−7.6) are the only cliffs; math500
   (procedural reasoning, +1.6), code (−2.4/+3.4), and compliance
   (−1.7) are clean. W4A16 hurts the model where fragile factual
   associations must survive a long reasoning chain, not where
   procedure or instruction-following carries it.~~
   **[REFUTED by the paired-complete re-analysis: gpqa's cliff is 100%
   truncation asymmetry (pc +0.0, base=quant exactly on complete
   pairs). Corrected signature: a small ifeval-think delta (pc −3.4,
   selected n=116) plus a completion-rate cost — the quant thinks past
   the cap more often (length damage, measured separately). Math/code/
   compliance clean stands.]**
2. ~~**Truncation flattered gpqa most**: −4.5 ns under the 4096 cap →
   −11.1 at 16k. Same mechanism the trial showed on ifeval (−7.0 →
   −11.8 completion-mode): capping cuts the chains where quant error
   compounds.~~ **[WRONG DIRECTION: raising the cap to 16k did not
   unmask gpqa damage — it moved the truncation asymmetry (22%→36%)
   and that asymmetry IS the −11.1. gpqa damage on complete pairs:
   zero.]**
3. Thinking costs the BASE 31–32pt of ifeval strict acc (51.6 think /
   83.0 nothink) — deployment-relevant base-side fact for the 4B.
4. Instrument receipts: all deltas paired per-doc greedy, n = full task
   (541/198/500/164/378); CIs are paired bootstrap.

## Status

- ~~These are the champion's **fixed-battery reference numbers** — the bar
  for arm re-adjudication.~~ **[SUPERSEDED: think-leg rows above carry
  cap asymmetry; the pc rows (gpqa +0.0, ifeval-think −3.4, math +1.5,
  nothink −1.7) are the current reference-at-16k, and the v2 32k
  rerun (champion + arms, paired-complete scoring) is the actual bar.]**
- Arms (qwenfinals / if3x / 4bself) quants rebuilt from frozen calib
  (RUN_DONE, sha-pinned); v2 battery running.
- 27b qwenfinals: humaneval −35.4 was the decisive clean leg —
  rejection likely stands; re-run only if 4B rankings shift materially.
