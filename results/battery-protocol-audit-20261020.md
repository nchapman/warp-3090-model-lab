# Battery protocol audit — master standing of every capability claim (2026-10-20)

Written after the third and worst instrument defect (cross-run mode
mixing) was found and proven. This file is the authoritative index of
WHAT STANDS for every battery-derived conclusion; RESULTS.md sections
point here. Doctrine: a battery number without its mode tag
(comp / chat-think / chat-nothink) is unusable; modes never mix in one
comparison; completion-mode numbers are regime measurements, not
deployment claims.

## The three defect classes (chronological)

1. **Era-1 defect — completion-4096 mode flips + cap** (all numbers
   2026-10-17/18): battery ran completion mode (no chat template),
   4096-token cap. Qwen3.5 emits think-spontaneously; mode flips and
   unclosed-think truncation dominated deltas (champion ifeval −7.9 =
   −7.4 mode-flip + −0.55 within-mode). Decomposed and retracted
   2026-10-19: `battery-think-cap-confound-20261019.md`.
2. **Era-2 defect — truncation asymmetry at 16k** (2026-10-19 fixed
   reruns): chat-template legs truncate asymmetrically (ifeval-think
   base 57% vs champion 72% at cap); raw deltas mix damage with cap
   asymmetry. Paired-complete scoring + trunc-as-length-metric adopted:
   `battery-fixed-20261019.md`.
3. **Era-3 defect — CROSS-RUN MODE MIXING** (found 2026-10-19/20,
   owned by the agent): numbers from different runs/eras were compared
   without checking run mode — champion chat-think −7.6 was set against
   rebuild COMPLETION −17.7 and read as a 10pt damage cliff. Proven
   false by matched-mode arms (RESULTS.md 10-20 verdict): v1rebuild is
   champion-class (−6.5 vs −7.6, n=541, same base). This defect also
   manufactured the "~6pt exclusion cost" story. Fixed structurally:
   runs now stamp protocol identity in events; battery_report.py
   hard-fails cross-protocol pairing (commit b6eb4c7).

## Claim-by-claim standing

| # | claim (as stated in RESULTS.md) | instrument / mode | standing |
|---|---|---|---|
| 1 | Champion ifeval −7.9 cliff; IF-only damage; math/code clean | comp-4096 | **REGIME-ONLY.** Real measurement of the completion-4096 regime; −7.4 of it is mode flips. Deployment-mode champion: −7.6 chat-think / −1.7 nothink / pc −3.4 |
| 2 | "Damage concentrates in instruction-adherence, not capabilities" | comp-4096 | **RETRACTED as a general claim** (era-1 artifact). Regime version stands; deployment-mode profile unmeasured beyond ifeval |
| 3 | traj-v1 agentic/IF corroboration of the cliff | 32-token greedy rollouts (own instrument) | **STANDS** — unaffected by all three defects |
| 4 | gdnfix arms: "~6pt ifeval cost of exclusion" | comp-4096, cross-era comparison | **RETRACTED** (era-1 + era-3). Matched deployment-mode: exclusion ≈ 0 cost vs champion. **Decision reversed: exclusion RETAINED** (10-20 verdict) |
| 5 | "Keep in_proj IN the quantization; spend bits instead" (next-ship implication) | derived from #4 | **RETRACTED** — superseded by #4's reversal |
| 6 | "Probe wins do not carry to capability" (qf worst ifeval cell) | comp-4096 | **WEAKENED to untested.** The qf/4bself ifeval cells were era-1; deployment-mode selfgen battery NEVER ran (arm2 killed as moot). Doctrine "trust neither instrument alone" stands; the specific ordering does not |
| 7 | champion-xhost control: ≤0.1pt cross-host reproducibility | comp-4096, same-mode pairs | **STANDS** (instrument control was mode-consistent) |
| 8 | gpqa cliff = truncation asymmetry (pc +0.0, n=115) | chat-16k paired-complete | **STANDS** |
| 9 | ifeval-think pc −3.4 [−6.9,−0.9], selection-biased low | chat-16k paired-complete | **STANDS with stated bias**; full-n raw chat-16k is −7.6 (trunc 57/72%) — both true, different estimands |
| 10 | math500 battery unusable (35–57pt base-vs-base swing); gpqa flexible ±9.6pt; ifeval strict 0.2pt base-vs-base | same-mode base-vs-base pairs | **STAND** (mode-internal controls) |
| 11 | champion comp16k −11.8 [−15.7,−7.8]; v1rebuild −17.7 raw / −34.9 pc, trunc 58→67% | comp-16k matched | **STAND as regime measurements.** The ~6pt completion-regime gap is real but CIs brush; completion is not a deployment regime |
| 12 | champion chat-think −7.6 / −7.0; nothink −1.7 [−4.3,+0.9] | chat-16k | **STAND** — deployment-mode anchors |
| 13 | v1rebuild −6.5 as-built / −5.9 +champion-metadata / nothink −3.7; metadata 0.5pt; exclusion retained | chat-16k verdict arms, n=541 | **STANDS** — current truth (10-20 verdict) |
| 14 | Probe: calib authorship halves dPPL; 4bself +0.48 best-ever; drift band 0.8–1.7pt | teacher-forced probe (own instrument) | **STANDS as probe findings** — never battery-confirmed in deployment mode |
| 15 | GGUF showdown (imatrix tie, +2% W4A16 floor, qf instrument inversion) | llama.cpp (own instrument) | **STANDS**; qf inversion is a live warning for selfgen claims |

## Not ruined (explicitly)

Same-mode paired comparisons, mode-internal controls, and every other
instrument (probe, traj-v1, GGUF, dPPL/wKL) were never exposed to the
defects. The champion's original ship decision used era-1 numbers as a
same-mode pair — internally valid then, and its deployment-mode profile
is now measured (#12). What the defects destroyed: cross-mode
comparisons, the exclusion-cost story, and the general IF-damage
narrative — all retracted above.

## Rules going forward

- Quote mode with every number: `comp-4096`, `comp-16k`, `chat16k-think`,
  `chat16k-nothink`. Unstamped historical numbers: treat as comp and
  check the run's events/logs before using.
- Deployment decisions only on chat-template legs; completion legs are
  regime diagnostics.
- Runs before 2026-10-20 have no protocol stamp in events — verify mode
  from logs (`arguments["gen_args_0"]`, serve flags) before pairing.
- The guard (b6eb4c7) enforces this for future runs automatically.
