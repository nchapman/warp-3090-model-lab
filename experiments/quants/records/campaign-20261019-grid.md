# Campaign grid 2026-10-19 — qwen winner + length / mix / author levers

Machine: H200 (RunPod yav18d8q81klrr, 143GB, $5.29/hr). Cross-machine
doctrine: every number below is measured on this machine, including the
champion re-anchor. Controls (GB10, 4096): gdnfix-qwenfinals +0.98% dPPL /
ifeval −14.6; gdnfix-head +2.33% / −13.7; champion +3.33% / −7.8.

NOTE (2026-10-19): this grid describes the CAMPAIGN-ERA artifacts. The
arms were since rebuilt (different bytes; same-recipe rebuild drift is
0.8–1.7pt dPPL) and re-probed same-box — current-artifact probe numbers
live in `probe-h200-20261019.md`. Battery verdicts below were
rendered under the 16k cap instrument and carry the think-truncation
caveat (see battery-think-cap-confound doc).

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

> **RETRACTION (2026-10-19, later same day):** the doctrine recorded
> below — "probe-v4 cannot arbitrate calib recipes; battery is the only
> decider" — is WITHDRAWN. The battery's ifeval/gpqa legs ran in
> completion mode with a 4096-token generation cap: spontaneous
> `<think>` gets truncated mid-reasoning (99% of unclosed think is
> at-cap, i.e. WE cut it) and quant-vs-base response-mode flips dominate
> the ifeval delta (champion −7.95pt = −7.39 mode-flips − 0.55
> within-mode). Full record + receipts:
> `results/battery-think-cap-confound-20261019.md`. Measured numbers in
> this grid stand as measurements; every ifeval/gpqa verdict is pending
> re-adjudication under the fixed protocol (trial COMPLETE: the
> champion's ifeval cliff is think-mode damage — ~~−7.0 with reasoning on
> under a sound instrument, −1.7 ns with reasoning off~~ [the "sound
> instrument" assumption failed the same day: the 16k chat legs cut
> 57%/72% of ifeval rows; paired-complete −3.4, and gpqa's −11.1-style
> cliffs are truncation artifacts (pc +0.0) — re-adjudication now runs
> under the v2 instrument: 32k cap + paired-complete scoring, see
> `results/battery-fixed-20261019.md`], −1.7 ns with reasoning off — see
> the confound doc). Code legs (humaneval+/mbpp+) are unconfounded (0% think).

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
  Note: probe had IF as the arm's best domain (+0.62% dPPL).
  [Retracted: the "teacher-forced KL does NOT predict ifeval" framing
  was scored against the confounded ifeval leg — see the banner.]
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
- **Doctrine (RETRACTED — see banner): probe-v4 cannot arbitrate 27B
  caligs** — it ranked
  qwenfinals 3.6× better (dPPL +0.28 vs +1.00, KL-equivalent per domain)
  while the battery shows a code/IF catastrophe. Consistent with 4B:
  flash-authored finals hurt ifeval at both scales (4B: -14.6 vs -7.8;
  27B: -20.7 vs -0.9). ["Battery is the decider" — withdrawn same day;
  the battery's ifeval legs were confounded.]
- A100 pod 8v9gjqwhj1a5pf STOPPED 06:15Z (EXITED, disk retained) after
  Hub verify + receipts sync — nothing GPU-bound remained.

## ARM3 4bself — self-authored finals vs gdnfix-qwenfinals (flash-authored)

- regen: DONE 08:41:34Z on H200 (999 rows in 20 min, run
  regen_calib_4bself.20261019T081945Z.95918; GB10 backup killed by
  recorded pids 3558755/3558780 after TERM+KILL). Provenance:
  teacher Qwen/Qwen3.5-4B@local-vllm, t0.7/p0.95/k20 thinking on,
  max_tokens 16384, source=v1 (1c7964a0 verified). Calib sha
  3c2aa215… synced to GB10 repo, MANIFEST-pinned (commit aa57760).
- quant @4096: RUN_DONE 09:24:40Z (runs/quant-4b.20261019T090040Z.97716),
  PREFLIGHT_DIV64_OK + assert_packed_div64 OK, model.safetensors sha
  26013e963adb825f0044f8ba9e2c489dfdf7bcf47e1765abbed3946e198ea72d.
- probe (H200): dPPL **+1.32%** (qNLL 1.4130), weightedKL 0.1137,
  p99.9 19.51 — best 4B probe of the day (champion GB10 +3.33%, long8k
  +2.93%, if3x +2.73%). swe +15.23% usual outlier.
- battery (H200, same base side): ifeval **-12.6** CI [-16.5,-8.9] ⛔
  (champion -7.0); gpqa -6.6 / humaneval -3.0 / mbpp -2.4 / math +1.2 —
  all within CI noise. Exactly one cliff.
- **VERDICT: REJECT 4bself — self-authored finals do not rescue IF.**
  With flash@4B (-14.6 GB10), flash@27B (-20.7 H200), self@4B (-12.6):
  EVERY regenerated-finals calib doubles ifeval damage vs cascade-v1
  (-7.0 H200 / -7.8 GB10). The champion's Nemotron hand-authored finals
  are load-bearing for IF. (Probe ranked this arm best — the
  "probe/battery inversion" framing is retracted; see the banner.)

## Verdict table

| arm | recipe delta | probe dPPL | ifeval Δ (battery) | verdict |
|---|---|---|---|---|
| ARM1 long8k | calib windows 8192 | +2.93% | **-15.2** (mbpp -7.1 too) | **REJECT — 4096 stays** |
| ARM2 if3x | IF dose 14.1% @4096 | +2.73% | **-14.2** (mbpp +10.3, math +5.4) | **REJECT for purpose** (damage-shift lead) |
| ARM3 4bself | finals self-authored | **+1.32%** | **-12.6** | **REJECT — regenerated finals hurt IF at any authorship** |
| 27b qwenfinals | finals flash-authored @27B | +0.28% | **-20.7** (humaneval -35.4) | **REJECT — incumbent cascade-v1 stays canonical** |
| champion (ref) | cascade-v1 | +3.33% (GB10) | -7.0 (H200) / -7.8 (GB10) | ships |

**Campaign conclusions:**
1. Cascade-v1 @4096 wins every comparison — calib composition is
   already optimal among tested deltas; all four levers (window, IF
   dose, author flash, author self) made things worse on ifeval.
2. [RETRACTED 2026-10-19] Probe-v4 ranked the two worst battery
   artifacts as the two best probes — the doctrine this produced
   ("cannot arbitrate calib recipes; battery is the only decider") is
   withdrawn: the battery's ifeval/gpqa legs were confounded by
   think-cap truncation and mode flips
   (`results/battery-think-cap-confound-20261019.md`). Whether probe-v4
   can arbitrate calib recipes is RE-OPENED, to be decided against the
   fixed battery.
3. ifeval is the universal casualty of calib drift (measured under the
   confounded 4096-cap protocol — pending re-adjudication); code/math
   move around (if3x bought mbpp +10.3) but IF only degrades.

All REJECT verdicts above rest on ifeval cliffs measured under the
confounded protocol; same-protocol pairing means rankings may survive,
but no cliff is quotable until re-measured under the fixed battery.
