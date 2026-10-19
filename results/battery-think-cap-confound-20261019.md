# Capability battery think-cap confound — discovered 2026-10-19

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

## Fixes under trial (H200, 2026-10-19, champion 4B pair, ifeval-only)

- T1 `comp16k`: completion + 16384 cap. **RESULT: −11.8 CI [−15.7, −7.8]**
  — WORSE than 4096-cap (−7.0 H200 / −7.95 GB10). Raising the cap does
  not dissolve the cliff; letting think finish deepens it. The champion
  ifeval cliff is real, not a truncation artifact.
- T2 `chat16k`: chat template + 16k cap, thinking on — running.
- T3 `chat16k-nothink`: chat template + 16k cap, `enable_thinking=False`
  (lm-eval 0.4.13 passes it into `apply_chat_template`) — queued.

Winning condition becomes the battery default for generative tasks;
code tasks stay completion. Full fixed rerun (champion pair, all tasks,
both sides) follows the trial.

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
