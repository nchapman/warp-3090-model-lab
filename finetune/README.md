# finetune/ — managed & external fine-tuning workflows

**Charter (2026-10-20):** workflows where fine-tuning happens on someone
else's hardware — managed FT APIs (Together, Fireworks, Crusoe, Nebius),
OpenRouter-hosted teachers, or rented-pod SFT of stock models. Recipes here
ship data + hyperparameters out and pull checkpoints back; they never run
the local speculators stack.

**Why this exists:** `docs/compute-platforms.md` surveyed the managed-FT
landscape and concluded DFlash2 drafters can't run there — the drafter is a
different architecture initialized from tapped target layers, not a
fine-tune (see "Why managed fine-tuning can't host the pipeline"). But
managed FT *is* viable for off-policy SFT baselines on stock models, and
the lab had no home for those workflows.

**Scope split:**
- Local training (speculators, DFlash2, draft-OPD) → `drafters/`
- Managed/external FT of stock models, teacher-generation campaigns, and
  the upload/download plumbing they need → here.

**Seeds:** `docs/compute-platforms.md` (platform survey) is the
current-state doc; `experiments/drafters/launch_ctrl_sft.sh` is the
campaign-driver pattern to copy when the first recipe lands
(runlib-instrumented, PID-disciplined, results to records/).

**Status:** scaffold — no code yet. The first recipe here follows the repo
contract: `lib/runlib.sh` events for anything long-running, frozen inputs
under `data/`, provenance receipts in the run log.
