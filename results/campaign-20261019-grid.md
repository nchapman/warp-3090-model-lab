# Campaign grid 2026-10-19 — qwen winner + length / mix / author levers

Machine: H200 (RunPod yav18d8q81klrr, 143GB, $5.29/hr). Cross-machine
doctrine: every number below is measured on this machine, including the
champion re-anchor. Controls (GB10, 4096): gdnfix-qwenfinals +0.98% dPPL /
ifeval −14.6; gdnfix-head +2.33% / −13.7; champion +3.33% / −7.8.

## CHAMPION re-anchor (H200)

battery pair vs H200 base (tasks n=198/164/541/378/500), run dirs
battery-base-reanchor.20261019T055723Z.22037 + battery-champion-reanchor.20261019T060924Z.30897:

| Δ vs base | champion (H200) | champion (GB10) |
|---|---|---|
| ifeval | **-7.0** CI [-10.9,-3.1] ⛔ | -7.8 |
| gpqa | -4.5 [-12.1,+3.0] | — |
| humaneval+ | -2.4 [-7.9,+3.0] | — |
| mbpp+ | +1.3 [-4.5,+7.1] | — |
| math500 | +2.6 [-2.0,+7.0] | — |

Machines agree (-7.0 vs -7.8 on ifeval): H200 numbers are directly
comparable to GB10 controls; cross-machine caveat retired for this grid.

## ARM1 long8k — qwenfinals calib @ 8192 vs gdnfix-qwenfinals @ 4096

- quant: runs/quant-4b.20261019T044905Z.11153 (H200, 36 min, RUN_DONE ok,
  PREFLIGHT_DIV64_OK + assert_packed_div64 OK, 3.6G safetensors, serving
  params stamped)
- probe (H200, probe-v4, 378 rows): dPPL **+2.93%** (qNLL 1.4353),
  weightedKL 0.114, p99.9 KL 19.78. Domains: swe +13.82% (outlier,
  mean_kl 0.99), conversational_agent KL 1.62, math +1.19%, IF +1.33%.
  GB10 control gdnfix-qwenfinals: +0.98% / qNLL 1.4157 — probe says 8192
  LOSES to 4096 by ~2pt on its own instrument (cross-machine caveat;
  battery decides).
- battery (H200, same base side as champion): ifeval **-15.2** CI
  [-18.9,-11.6] ⛔ (double the champion's -7.0), mbpp **-7.1** [-13.2,-1.1]
  ⛔ (champion +1.3); gpqa/humaneval/math within noise of champion.
- **VERDICT: REJECT 8192 — 4096 calib windows stay.** Probe (+2.93 vs
  control +0.98 on GB10) called the direction; battery seals it with two
  cliffs. Also settles regen doctrine: calib window 4096 for ARM3.

## ARM2 if3x — IF-appended dose (14.1%) @ 4096 vs gdnfix-head @ 4096

- calib: data/calib-cascade-if3x.jsonl sha 5ee08273 (999 v1 rows byte-identical + 104 fresh IF)
- quant: runs/quant-4b.20261019T052907Z.19176 (H200, 18 min, RUN_DONE ok,
  PREFLIGHT_DIV64_OK + assert_packed_div64 OK, 3.6G safetensors)
- probe (H200, probe-v4, 378 rows): dPPL **+2.73%** (qNLL 1.4326),
  weightedKL 0.111. IF domain is the arm's BEST: +0.62% dPPL / KL 0.0426
  (dose didn't hurt IF fidelity — direction consistent with intent).
  Same H200 pairing vs long8k: if3x 1.4326 < long8k 1.4353 (clean,
  same-machine). GB10 control gdnfix-head: +2.33% / qNLL 1.4347
  (cross-machine caveat; ifeval battery decides the IF question —
  control domain tables were never persisted, aggregates only).
- battery (H200): ifeval **-14.2** CI [-17.9,-10.5] ⛔ — the IF dose
  DOUBLED ifeval damage vs champion (-7.0). But mbpp **+10.3** [+4.8,+15.9]
  ⬆ (largest gain on the board) and math500 **+5.4** [+0.6,+10.0] ⬆.
  Note: probe had IF as the arm's best domain (+0.62% dPPL) — second
  consecutive campaign where teacher-forced KL does NOT predict ifeval.
- **VERDICT: REJECT for the IF purpose** (dose hurts its target). The
  mbpp/math trade is real damage-shifting, not damage-fixing — logged as
  a lead (cascade+IF-append recipe shifts quant damage from code/math
  into IF), not a shippable recipe.

## 27B qwenfinals — built, held; canonical decision OPEN

- quant: RUN_DONE ok 06:00:22Z (5h35m, A100 80GB, run
  `quant-27b.20261019T002606Z.37464`), PREFLIGHT_DIV64_OK,
  assert_packed_div64 OK, MTP carried, serving params stamped. 19G
  artifact: model.safetensors sha256
  `9d1420e9c1b56bb866e49d63fd24f129cab048bcba2baed8dbb8f62ee4e01538`,
  model_mtp `1d8268aa…da9fe`. Receipts: `artifacts/Qwen3.8-27B-W4A16-GPTQ-qwenfinals/`
  (SHA256SUMS, quant.log 48MB, events.jsonl, cmd.txt, card).
- **Repo collision discovered**: `nchapman/Qwen3.8-27B-W4A16-GPTQ` (public,
  2026-10-10, 95 downloads) is the CASCADE-V1-calibrated build, measured
  +1.00% dPPL, froggeric v22.5 chat template (28.2KB). Today's qwenfinals
  build ≠ that artifact (main weights sha differs; MTP identical because
  copied verbatim from base).
- **Published to PRIVATE holding repo** `nchapman/Qwen3.8-27B-W4A16-GPTQ-qwenfinals`
  (Hub LFS shas verified byte-exact; 81s upload). Card states lab-artifact
  status, recipe, run provenance. Card corrected 07:00Z: calib teacher is
  **Qwen3.8-flash** (OpenRouter, thinking on) — NOT 27B-self-authored as
  first written (provenance.json is the truth: teacher
  qwen/qwen3.8-flash@openrouter, t0.7/p0.95/k20, max_tokens 16384,
  999 rows).
- paired probe (H200, probe-v4, 378 rows): dPPL **+0.28%** (qNLL 2.1532
  vs base 2.1471), weightedKL 0.1206, p99.9 24.49. Domains: several
  NEGATIVE (IF -1.88%, math -0.95%, chat -0.19% — quant ≈ base at
  noise level); swe +16.27% outlier + conv_agent KL 2.14 (the usual
  low-entropy-syntax signature, seen at every size).
  vs incumbent card (+1.00% aggregate, per-domain KL chat 0.033 / IF
  0.053 / math 0.098 / science 0.097 / terminal 0.486): today's build
  is 3.6× better on aggregate dPPL and KL-equivalent domain-by-domain
  (chat 0.032 / IF 0.052 / math 0.100 / science 0.088 / terminal 0.479).
  Same probe design (378 conv, 7 domains + 9 langs); incumbent's host
  machine unknown — treat dPPL as same-instrument, mild caution.
- battery (H200, same base side): incumbent cascade-v1 = near-perfect
  retention (ifeval -0.9, gpqa -3.0, humaneval -1.8, mbpp -4.2,
  math +4.0, NO cliffs); qwenfinals = CATASTROPHE on code/IF
  (humaneval **-35.4**, ifeval **-20.7**, gpqa -13.1) despite winning
  the probe. KNOWN CONFOUND: incumbent ships froggeric v22.5 chat
  template (28.2KB), qwenfinals build carries a different 8.9KB one;
  each battery side inherits its own template. One-lever arbitration
  RUNNING: qwenfinals weights + incumbent template (hybrid dir
  /root/qf-froggeric, symlinked weights + copied chat_template.jinja;
  tokenizer_config embeds no template on either side).
- hybrid side (qwenfinals weights + froggeric v22.5 template,
  /root/qf-froggeric, run battery-27b-qf-froggeric): humaneval **-34.1**
  [-42.1,-26.2], ifeval **-24.2** [-28.7,-19.8], gpqa **-18.7** — template
  ruled out; the WEIGHTS are damaged. Not a regression vs the original
  template side (humaneval -35.4→-34.1, ifeval -20.7→-24.2 — noise).
- **VERDICT: REJECT qwenfinals @27B — incumbent cascade-v1 stays
  canonical.** Incumbent battery (first ever measured): ifeval -0.9 /
  gpqa -3.0 / humaneval -1.8 / mbpp -4.2 / math +4.0, zero cliffs.
  Replacement required "better"; it is strictly worse on capability.
  No user decision needed — the data is unambiguous.
- **Doctrine: probe-v4 cannot arbitrate 27B caligs** — it ranked
  qwenfinals 3.6× better (dPPL +0.28 vs +1.00, KL-equivalent per domain)
  while the battery shows a code/IF catastrophe. Consistent with 4B:
  flash-authored finals hurt ifeval at both scales (4B: -14.6 vs -7.8;
  27B: -20.7 vs -0.9). Battery is the decider — starkest example yet.
- A100 pod 8v9gjqwhj1a5pf STOPPED 06:15Z (EXITED, disk retained) after
  Hub verify + receipts sync — nothing GPU-bound remained.

## ARM3 4bself — self-authored finals vs gdnfix-qwenfinals (flash-authored)

- regen: DONE-pending on H200 (GB10 first pass too slow: saturated but
  bandwidth-starved, 650 tok/s). H200 pass launched 08:19:45Z
  (run regen_calib_4bself.20261019T081945Z.95918, server pid 95943,
  GPU_MEM_UTIL=0.9, CONC=128, same resumable script, calib sha
  1c7964a0-verified on pod). GB10 pass left running as backup.
- quant: (pending — @4096 windows per ARM1 verdict)
- probe: (pending)
- battery: (pending)
- VERDICT: (pending)
