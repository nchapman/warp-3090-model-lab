# drafters/ — DFlash2 speculative drafter pipeline

Train quant-aligned block-diffusion drafters (speculators `dflash2`) that
draft for the **deployed quant** — exact-pairing acceptance is the whole
point, so regen always samples the `*-W4A16-GPTQ` artifact, never bf16.

**Training is documented in `train.md`** — arms, per-scale commands,
attention impl per machine (`sdpa` on Spark), checkpoint cadence. This file
is the map, not the manual. Deep docs also in-dir: `data_scaling.md` (the
ladder narrative + corpus truth), `external_references.md` (DFlash-family
literature, verified from primary sources — check its §7 before quoting
any external claim).

## Venv

```bash
# from the repo root
python3 -m venv .venv-dflash && .venv-dflash/bin/pip install -r drafters/requirements-drafter.txt
# fp8 states (default): patch speculators in that venv
SP=$(./.venv-dflash/bin/python -c 'import speculators,os;print(os.path.dirname(os.path.dirname(speculators.__file__)))') \
  && patch -p1 -d "$SP" < drafters/patches/speculators_fp8_load_cast.patch
# Draft-OPD post-training (optional, inert without opd_ columns):
#   patch -p1 -d "$SP" < drafters/patches/speculators_draft_opd.patch
```

torch 2.13.0 / speculators 0.8.1 / vllm 0.31.0. `vllm serve` needs the
venv's bin on PATH.

## Pipeline (each stage's entry point)

| stage | tool |
|---|---|
| slice conversations → regen units | `slice_convs.py` (presets; outputs pinned by `data/conv-ids/`) |
| regenerate finals on-policy | `regen.py` (slices) · `regen_pool.sh <model-key>` (full corpus, resume-safe — the standing per-model process) |
| assemble + cap | `prep_dataset.py` (`--loss-mode all\|last`, `--max-tokens`) · `reshape_tokens.py` (prefix/merged/first/short shaping) |
| extract training states | `extract_states.py` — **transformers forward on the quant, never vLLM** (vLLM states are corrupt on W4A16); taps are N+1 (aux + final), filled from `models.py` |
| init from tapped layers | `bootstrap_draft.py` |
| train | commands in `train.md` (pod pattern: `experiments/drafters/launch_ctrl_sft.sh`) |
| select checkpoints | `sweep_epochs.sh` — full serving-temp campaign, ranked table with pooled n |
| measure τ | `eval_ladder.sh` (`TEMP` env: 0 / 0.7 / 1.0 regimes) · `eval_mtp.sh` (native-MTP baseline) · `bench_subsets.sh` (per-domain breakdowns) |
| account tokens | `data_audit.py` — unique supervised tokens, conversations, dup rates; **never guess these** |
| budget rungs | `data_ladder.sh` + `slice_rungs.py` |

## Reproduce, small scale (any 24GB+ GPU)

```bash
python drafters/slice_convs.py --preset small --quant ./Qwen3.5-4B-W4A16-GPTQ
vllm serve ./Qwen3.5-4B-W4A16-GPTQ --port 8138 &     # the DEPLOYED quant
python drafters/regen.py slices/s_train.jsonl slices/s_regen.jsonl slices/s_merged.jsonl --model 4b
python drafters/prep_dataset.py --jsonl slices/s_merged.jsonl --out ds_small --quant ./Qwen3.5-4B-W4A16-GPTQ
python drafters/extract_states.py --dataset ds_small --verifier ./Qwen3.5-4B-W4A16-GPTQ \
    --out ds_small --model 4b --parity 20
python drafters/bootstrap_draft.py --model 4b --quant ./Qwen3.5-4B-W4A16-GPTQ --out draft_init
# train command: train.md (small-scale section; sdpa on Spark)
bash drafters/eval_ladder.sh draft_small/0 small 15 8151      # expect τ ≈ 2.18
```

Full-corpus regen for any registered model is one command —
`regen_pool.sh <model-key>` serves the deployed quant with `models.py`
SERVING_PARAMS (t 0.7 / top_p 0.95 / top_k 20 — the deployment point) and
regenerates every context's final, resume-safe; outputs land in `data/`.

## Champion scale (1×B200/B300 + ~4TB NVMe, ~$60 rental)

Same sequence with `--preset v4` (100k convs → 260M tokens; fp8 6-tap
states ≈ 3.2TB — disk is the gating resource), then both train arms from
`train.md` at checkpoint-freq 1. Expect τ ≈ 3.8 at the serving point
(temp 0.7) vs native MTP ≈ 2.9 — measure the MTP baseline on your own
ladder first; published MTP numbers are harness-specific. Checkpoint
selection: `sweep_epochs.sh <checkpoints_parent> <sweep_name>`.

## Published

The champion drafter lives at `nchapman/Qwen3.5-4B-DFlash2` (standard
speculators naming — model it drafts for, no quant marker: measured
drafting against both the quant verifier and the bf16 base; weights sha256
in the card). Earlier drafter generations (v3, v3.5) and the v4 per-epoch
snapshots stay as local research record under
`../llm-compressor/artifacts/` (read-only archive — never written to).

## Doctrine that is not optional

- Selection happens at **t=0.7** (serving point), n≥4 per contender; greedy
  is a stability instrument, t=1.0 is heavy-tailed. State temp and n with
  every τ (`AGENTS.md` "Measuring τ").
- Eval prompts: `data/ladder-20.jsonl` is **general-chat only** — every τ
  ever quoted is general-chat τ. Don't claim deployment relevance without a
  code/agentic/long eval.
- Every run goes through `lib/runlib.sh` — no exceptions.
