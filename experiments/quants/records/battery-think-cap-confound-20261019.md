# Capability battery think-cap confound — discovered 2026-10-19

> **Superseded in part (2026-10-20):** this doc covers defect classes 1–2
> (mode flips, cap truncation). A THIRD defect — cross-run mode mixing —
> was found later the same week and retracts additional conclusions
> (exclusion cost, IF-damage narrative). Authoritative claim-by-claim
> standing: `battery-protocol-audit-20261020.md`.

**Status: re-adjudication in progress.** This document records a major
defect in `quant/capability_battery.sh` as run for every published
capability number to date, and retracts the campaign doctrine that was
built on it.

## The defect

The battery runs all five tasks (ifeval, gpqa, humaneval+, mbpp+,
math500) in **completion mode** (no chat template) with
`MAX_GEN_TOKS=4096` and a think-strip patch (everything before
`</think>` is dropped; unclosed `<think>` keeps the whole tail).

Qwen3.5 is a hybrid reasoning model: even in completion mode it
spontaneously emits `<think>` reasoning before answering. Two failure
modes follow:

1. **Mode flips.** The quant flips response mode on individual prompts
   (think↔direct) relative to base. A flip into truncated-think is an
   automatic zero regardless of instruction-following ability.
2. **Cap truncation.** Unclosed-think responses are ~99% at the 4096
   generation cap (not model behavior — WE cut them off mid-reasoning).

## Measured receipts (champion 4B pair, GB10, full samples, n=541)

- ifeval Δ −7.95pt decomposes as **mode-flips −7.39 + within-mode −0.55**
  (strata base→quant: think→think 283, think→direct 56, direct→direct
  115, direct→think 87). Base per-mode acc: direct 85.1% / think 24.2%.
- **Both-direct stratum (mode held fixed, n=115): −4.3 CI [−9.6, +0.9]**
  — not individually significant.
- Unclosed think = 99% at-cap (base 334/339, quant 367/370 at ≥4090 tok).
- **Determinism control:** same quant rerun on another host → 0/541 mode
  flips, +0.18pt score noise. The flips are a real quant effect, not
  sampling noise.

All-task sweep (same pair):

| task | unclosed-think base→quant | Δ blended | both-direct stratum | clean? |
|---|---|---|---|---|
| ifeval | 63% → 68% | −7.95 | −4.3 (n=115, ns) | NO |
| gpqa | 41% → 60%, 100% at-cap | −5.56 | **−7.9 (n=63)** | NO — blended masks damage |
| math500 | 22% → 15%, mostly self-stopped | −1.2 | −2.6 (n=343) | partial |
| humaneval+ | 0% think | −3.7 | — | YES |
| mbpp+ | ~0% think | +0.0 | — | YES |

## What is invalidated

- Every published **ifeval / gpqa battery absolute** and the
  mode-flip-as-mechanism story told from the 4096-cap runs.
- Paired ifeval/gpqa **deltas as deployment claims** — they remain real
  cliff signals but measure "damage under completion-mode-with-4096-cap",
  a protocol no deployment uses.
- The 2026-10-19 campaign doctrine: **"probe-v4 cannot arbitrate calib
  recipes; battery is the only decider" — RETRACTED.** The probe's
  apparent "inversions" (4bself, qwenfinals@27B, if3x-IF-domain) were
  scored against a confounded battery. Teacher-forced KL is a sound
  instrument (no generation involved); whether it can arbitrate calib
  recipes is **re-opened**, to be decided against the fixed battery.

## What stays clean

- **humaneval+ / mbpp+**: 0% think — completion mode is the correct mode
  for pass@1 code tasks and the numbers are unconfounded. (Note: the
  chat-template fix CANNOT be applied to code tasks — it breaks
  lm-eval's execution grader; code stays completion.)
- **probe-v4** (teacher-forced KL/dPPL): untouched by this defect.
- The campaign grid's *measured numbers* stand as measurements; the
  verdict framing built on ifeval cliffs is pending re-adjudication.

## Fixed-battery trial — COMPLETE (H200, 2026-10-19, champion 4B pair, ifeval-only, n=541)

| leg | protocol | base | quant | Δ | 95% CI | decomposition |
|---|---|---|---|---|---|---|
| T1 comp16k | completion, 16k cap, think on | 54.2 | 42.3 | **−11.8** | [−15.7,−7.8] | flips −10.2 + within −1.7; both-direct (n=140) −7.1 [−12.9,−2.1] |
| T2 chat16k | chat template, 16k cap, think on | 50.8 | 43.8 | **−7.0** | [−10.5,−3.5] | ~~0/541 unclosed think BOTH sides~~ [blind check — T2 actually cut 57%/72% at cap] — all within-mode, zero flips |
| T3 chat16k-nothink | chat, 16k cap, `enable_thinking=False` | 83.0 | 81.3 | **−1.7** | [−4.3,+0.9] ns | **0/541 raw `<think>` tokens both sides — plumbing verified** |

Findings:

1. **The champion's ifeval damage is think-mode damage.** Compliance
   measured clean of reasoning (T3): −1.7, CI crosses zero. ~~Reasoning-on
   under a sound instrument (T2: untruncated, zero mode flips): −7.0.
   The quant degrades the reasoning chains, not instruction-following.~~
   **[PARTLY REFUTED (2026-10-19): T2 was NOT untruncated — 57%/72% of
   rows cut at the 16k cap (marker check was blind for chat legs).
   Paired-complete: −3.4 [−6.9,−0.9]. Think-mode-only direction stands;
   magnitude and "degrades the chains" vs "thinks longer past the cap"
   is exactly what the 32k v2 run separates.]**
2. **Truncation FLATTERED the champion**: 4096-cap completion −7.0
   (H200) / −7.95 (GB10) → 16k-cap completion −11.8. At 16k the
   "direct" class absorbs closed-think chains and quant error compounds
   across the longer reasoning (both-direct: −4.3 ns → −7.1 sig).
3. **Completion mode is itself an amplifier**: same 16k cap, chat
   template −7.0 vs completion −11.8 — spontaneous mode flips
   (125+54/541) are a completion-mode phenomenon. [Later note: the −7.0
   here also carries cap asymmetry — pc −3.4; the amplifier comparison
   is between two truncated instruments, direction only.]
4. Base-side measurement worth knowing: thinking costs the 4B BASE
   32pt of ifeval strict acc (83.0 nothink vs 50.8 think-on).

## Adopted fixed battery — RUN COMPLETE (2026-10-19)

Results: `battery-fixed-20261019.md`. Champion fixed reference:
~~**gpqa −11.1 (was −4.5 ns confounded — truncation masked the worst
cliff), ifeval-think −7.6, everything else clean** (math +1.6, humaneval
−2.4, mbpp +3.4, ifeval-compliance −1.7 ns). Damage signature =
knowledge-heavy reasoning. These are the bar for arm re-adjudication.~~
**[REFUTED the same day by the truncation audit: the 16k think legs cut
57%/72% (ifeval) and 22%/36% (gpqa); paired-complete gpqa is +0.0 and
ifeval-think −3.4. Current bar = the v2 rerun (32k cap, paired-complete
scoring).]**

- **ifeval runs BOTH ways**: chat+nothink@16k (compliance instrument,
  T3 protocol) AND chat+think (reasoning-retention instrument, T2).
- **gpqa / math500**: chat+think ~~@16k (truncation removed; these tasks
  need reasoning)~~ **[16k did NOT remove truncation — v2 uses 32k]**;
  these tasks need reasoning.
- **humaneval+ / mbpp+**: stay completion@4096 (0% think — clean; and
  chat template breaks lm-eval's execution grader).
- Arm re-adjudication under this battery queued (recipes frozen,
  ~20–35 min re-quant each on H200).

## Campaign re-adjudication status

- Arm verdicts (ARM1 long8k / ARM2 if3x / ARM3 4bself / 27b qwenfinals,
  all REJECT) rest on ifeval cliffs measured under the confounded
  protocol. Same-protocol pairing means the *rankings* may survive, but
  no ifeval cliff is quotable until re-measured under the fixed battery.
- 27b trio: flagged for identical re-treatment.
- **Data-loss note:** the H200 pod's container disk did not survive its
  stop — arm battery samples and probe events are gone. Aggregate
  results (27b results jsons, campaign grid) were synced and committed
  first. Arms are re-quantizable from the frozen sha-pinned recipes.
  Going forward: samples sync BEFORE any pod stop.
