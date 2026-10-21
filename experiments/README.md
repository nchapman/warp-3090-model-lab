# experiments/ — closed campaigns, frozen together

Code + docs + records for campaigns that have **ended**. Nothing here is a
living workflow: scripts were the campaign's drivers at their original
paths (renamed wholesale during the 2026-10-20 reorg), docs are the plans
as executed, records are the evidence. When a campaign closes, its pieces
move here in one `git mv`; when a new one starts, it starts in `quants/`,
`drafters/`, or `finetune/`.

## Contents

| dir | campaign | what's inside |
|---|---|---|
| `quants/` | calib-v2 arm (self-regen calib exploration) | `regen_calib_self.sh` · `regen_calib_finals.py` · `build_if3x_append.py` · `ifeval_modes.py` · `mini_moe_smoke.sh` (MoE W4A16 vLLM path validation) · `calib_v2_plan.md` · `records/` (battery audits + protocol notes, 2026-10-19/20 campaign grid, H200 probe, the Qwen3.8-27B qwenfinals bundle with SHA256SUMS + card) |
| `drafters/` | draft-OPD + dflash2 v5 research + ctrl closeout | `opd_collect/assemble/build_replay/probe` · `split_holdout.py` · `launch_ctrl_sft.sh` · `overnight_driver.sh` · `overnight_ladder_watch.sh` · `campaign_check.sh` · `draft_opd.md` · `dflash2_plan_v5.md` · `dflash2_research.md` · `records/` (bench csvs) |

## Rules

- **Frozen receipts cite pre-reorg paths** (`drafter/`, `quant/`, the old
  root `quants/` artifact dir). They are evidence, not living docs — don't
  rewrite them, don't "fix" their path references.
- Records grow only by **explicit copy** from a finished campaign. Pulled
  pod evidence under `pod_snapshot/` is gitignored by default; graduating
  receipts into `records/` is a deliberate act, never `git add -A`.
- Campaign drivers here (two levels deep, `experiments/<side>/`) `cd
  ../..` to the repo root and remain sourceable, but they are not maintained
  — copy forward what a new campaign needs rather than editing in place.
