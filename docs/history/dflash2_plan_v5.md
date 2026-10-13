# v5 plan: beat full MTP on our ladder — blend + selector + tooling

Standing goal: tau@16 (20-prompt held-out ladder, temp 1.0, b16-native,
interval-weighted acceptance) > full native MTP measured on the SAME ladder
with the SAME convention. Not the blog's 3.90 (their model, their harness,
their convention) — we re-anchor first.

Current: v4b = 3.25. Oracle@16 = 7.59. Selector acc = 0.754.
Gap decomposition (documented): realized-vs-oracle 4.3 (selector), blend
domain split (+/-1.3 vs rasyosef), quant cost (unmeasured), convergence
(unknown until per-epoch curve).

## Phase 0 — re-anchor + free signal (local GB10, ~1 day, no pod)

0.1 MEASURE THE TARGET: serve cascade quant + model_mtp.safetensors (full
    native MTP, k=3), fire the 20-prompt ladder, interval-weighted number
    recorded. Also run it on the 9-subset benchmark for the external
    profile. This number gates everything below.
0.2 PER-EPOCH CURVE: ladder-eval all 7 pulled checkpoints (armA 0/1/2,
    armB 0/1/3/5) -> acceptance-vs-epoch per recipe. Rising at ep6 =>
    convergence arm in Phase 3; flat => structure/selector arms.
0.3 QUANT-COST DECOMPOSITION: serve v4b drafter with the bf16 base
    verifier -> tau@bf16 vs tau@quant isolates quant-alignment cost
    (long-queued diagnostic; decides how much blend can buy).
0.4 SELECTOR READ: from 0.2's logs, per-position conditional acceptance —
    if the dip pattern (pos-1 dip, mid-block recovery) narrows with epochs,
    selector training is converging; if static, selector needs help.

## Phase 1 — tooling hardening (local, parallel with Phase 0)

1.1 eval_one.sh v2: interval-weighted acceptance computed and printed as
    THE number, auto-appended to results/results.csv; one-command wrapper
    for the 9-subset benchmark per drafter.
1.2 extract v2: fp8 cast on GPU + writer thread pool (target 2x throughput,
    parity-check on 1k rows first); tap-shape pre-validator (states vs
    draft aux dim) run before any bulk job; train wrapper asserts
    checkpoint-freq=1 (save-gate bug class).
1.3 regression: 100-row end-to-end (regen -> extract -> 10 train steps ->
    eval) on GB10 in <15 min; run before every pod round.

## Phase 2 — Round 1 on 2xB200 (~1.5 days): blend + selector at fixed scale

2.1 Regenerate ~80M tok targeted conversations (math/code/CoT slices from
    the cascade corpus, quant-aligned as always). Mix: 180M existing chat
    + 80M targeted = 260M total (scale held fixed — blend is the only
    variable).
2.2 Arm C — fresh init, v4b recipe (6e-4 cosine, 6ep), freq=1.
2.3 Arm D — warm-start from v4b on the same mix, 2ep gentle: cheap test
    of whether blend moves an already-good drafter.
2.4 Arm E — selector push on v4b's data (no regen): selector loss weight
    x2-x4, or selector-head-only extra epochs. Oracle says 4.3 acceptance
    sits behind selection; this is the cheapest probe of it.
Eval: ladder + 9-subset on all arms.
Success: tau >= 3.45-3.55. Decision: best arm feeds Phase 3.

## Phase 3 — Round 2 on 2xB200 (~1.5 days), chosen by evidence

- Convergence arm (if 0.2 shows rising curve): best mix x 10ep.
- Selector arm (if 2.4 moved): selector rank/candidates scale-up.
- Structure arm (else): 7-aux taps (re-extract ~3h + retrain ~8h).
Success: tau >= 3.7. If still < MTP - 0.1, the 27B rung (2.6B tok) is the
documented fallback with ~+0.3 expected — that decision gets its own doc.

## Standing rules

- No GPU launch without user sign-off. Early checks at 90s and 4min.
- Interval-weighted acceptance everywhere; last-line reads banned.
- All numbers land in results.csv + dflash2_research.md same day.
- Thinking-mode eval arm (their Table 3 convention) once tau >= 3.7,
  for the direct external comparison.
