# Campaign grid 2026-10-19 — qwen winner + length / mix / author levers

Machine: H200 (RunPod yav18d8q81klrr, 143GB, $5.29/hr). Cross-machine
doctrine: every number below is measured on this machine, including the
champion re-anchor. Controls (GB10, 4096): gdnfix-qwenfinals +0.98% dPPL /
ifeval −14.6; gdnfix-head +2.33% / −13.7; champion +3.33% / −7.8.

## CHAMPION re-anchor (H200)

(pending)

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
- battery: RUNNING (t-mveu31n7, side queued after base + champion)
- VERDICT: (pending battery)

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
- battery: RUNNING (t-mveu7sro→t-mveue9rs; first attempt died at 4min — GPQA is a
  gated dataset, pod had no HF credential; token shipped + gpqa cached,
  refired)
- VERDICT: (pending)

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
  status, recipe, run provenance.
- paired probe: CANNOT run on A100 (73GB resident + int8 decompress
  transients > 80GB — hard OOM at 25% decompress, t-mveumpnh). Queued to
  H200 tail after batteries (~09:10Z).
- Canonical decision (user): keep incumbent / second public repo /
  replace-after-probe. Incumbent's +1.00% is a high bar; replacement also
  swaps chat template for 95 downloaders.
- A100 pod 8v9gjqwhj1a5pf STOPPED 06:15Z (EXITED, disk retained) after
  Hub verify + receipts sync — nothing GPU-bound remained.

## ARM3 4bself — self-authored finals vs gdnfix-qwenfinals (flash-authored)

- regen: BLOCKED until ARM1 length verdict (finals max_tokens tuned to winner)
- quant: (pending)
- probe: (pending)
- battery: (pending)
- VERDICT: (pending)
