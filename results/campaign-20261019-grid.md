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
- battery: RUNNING (t-mveu7sro)
- VERDICT: (pending)

## ARM3 4bself — self-authored finals vs gdnfix-qwenfinals (flash-authored)

- regen: BLOCKED until ARM1 length verdict (finals max_tokens tuned to winner)
- quant: (pending)
- probe: (pending)
- battery: (pending)
- VERDICT: (pending)

## 27b qwenfinals (A100 ssh-6, dflash handoff artifact)

- quant: runs/quant-27b.20261019T002606Z.37464 (pid 37462)
- probe vs base: (pending)
- synced artifact: artifacts/Qwen3.8-27B-W4A16-GPTQ-qwenfinals/ (pending, sha receipt required)
