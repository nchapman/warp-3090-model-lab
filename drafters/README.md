# drafters/ — DFlash2 speculative drafter pipeline

Train quant-aligned block-diffusion drafters (speculators `dflash2`) that
draft for the **deployed quant** — exact-pairing acceptance is the whole
point, so regen always samples the `*-W4A16-GPTQ` artifact, never bf16.

**Venv:** `.venv-dflash` (torch 2.13.0 / speculators 0.8.1 / vllm 0.31.0),
plus the patches from `patches/` (fp8 load-cast required; draft-OPD optional
and inert without `opd_` columns — `experiments/drafters/draft_opd.md`).

**Training is documented in `train.md`** — arms, per-scale commands,
attention impl per machine (`sdpa` on Spark), checkpoint cadence. This file
is the map, not the manual.

## Pipeline (each stage's entry point)

| stage | tool |
|---|---|
| slice conversations → regen units | `slice_convs.py` (presets; outputs pinned by `data/conv-ids/`) |
| regenerate finals on-policy | `regen.py` (slices) · `regen_pool.sh <model-key>` (full corpus, resume-safe — the standing per-model process) |
| assemble + cap | `prep_dataset.py` (`--loss-mode all\|last`, `--max-tokens`) · `reshape_tokens.py` (prefix/merged/first/short shaping) |
| extract training states | `extract_states.py` — **transformers forward on the quant, never vLLM** (vLLM states are corrupt on W4A16); taps are N+1 (aux + final), filled from `models.py` |
| init from tapped layers | `bootstrap_draft.py` |
| train | commands in `train.md` (drafters venv; `launch_ctrl_sft.sh` pattern for pods lives in `experiments/drafters/`) |
| select checkpoints | `sweep_epochs.sh` — full serving-temp campaign, ranked table with pooled n |
| measure τ | `eval_ladder.sh` (`TEMP` env: 0 / 0.7 / 1.0 regimes) · `eval_mtp.sh` (native-MTP baseline) · `bench_subsets.sh` (per-domain breakdowns) |
| account tokens | `data_audit.py` — unique supervised tokens, conversations, dup rates; **never guess these** |
| budget rungs | `data_ladder.sh` + `slice_rungs.py` |

## Doctrine that is not optional

- Selection happens at **t=0.7** (serving point), n≥4 per contender; greedy
  is a stability instrument, t=1.0 is heavy-tailed. State temp and n with
  every τ (`AGENTS.md` "Measuring τ").
- Eval prompts: `data/ladder-20.jsonl` is **general-chat only** — every τ
  ever quoted is general-chat τ. Don't claim deployment relevance without a
  code/agentic/long eval.
- Every run goes through `lib/runlib.sh` — no exceptions.
