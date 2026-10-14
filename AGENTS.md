# warp-3090-quant-lab — agent rules

Project-specific doctrine for any agent (or human) working in this repo. The
global engineering guidelines still apply; this file adds what this lab
learned the hard way.

## Runs and processes

- Every eval/server/training run goes through `lib/runlib.sh` (bash) or
  `quant/lablog.py` (python) — no exceptions. If a script runs anything
  GPU-bound or longer than a minute without `run_init`/events, it's wrong.
- Inspect any run with `bash scripts/tail_run.sh [prefix]` — never go hunting
  for log paths manually.
- Servers die by the PID recorded in the SERVER_LAUNCH event (`stop_server`).
  Never `pkill` by pattern — it has killed our own shell twice.
- Never edit a script while an instance of it is running.
- vLLM serving with speculative decoding can wedge silently during CUDA-graph
  capture — pass `EXTRA_SERVE_ARGS="--enforce-eager"` if a spec-decode server
  hangs at startup.

## Measuring τ (acceptance)

- `TEMP` env on `drafter/eval_ladder.sh` / `eval_mtp.sh` selects sampling
  temperature (default 1.0). For checkpoint selection, use
  `drafter/sweep_epochs.sh` — one command, full campaign. Three regimes,
  three purposes:
  - **t=0 (greedy)** — the low-noise instrument (repeat spread ~0.05). Use to
    sanity-check stability, not to pick shipping checkpoints.
  - **t=0.7 (serving)** — the deployment point and the corpus generation
  config. **Selection happens here**, n≥4 per contender (spread ~0.1).
  - **t=1.0** — heavy-tailed (0.34 spread observed on identical inputs).
  Never rank checkpoints on small n at t≥0.7; never quote a single draw as a
  result. Report n and ranges with every number.
- Ranking is temperature-dependent. State the temperature with every τ. A
  checkpoint that wins at greedy may lose at serving — that's data, not error.
- Fresh GB10 numbers are the lab's anchors (`RESULTS.md`). Don't publish
  cross-machine τ comparisons; don't resurrect superseded numbers.

## Publishing

- One recipe per artifact. No internal run names ("v4b", "armB", epoch
  counts) in public Hub text — the card describes the recipe and the shipped
  checkpoint, nothing else.
- Hub ids: bare (`Qwen3.5-4B-W4A16-GPTQ`); `-cascade` suffix marks local-only
  calibration-generation variants; citable version = the frozen-data pin.
- Cards carry `model.safetensors` sha256 and full provenance
  (train_command, run.yaml, training_state, val_metrics).

## Environment

- Two venvs: `.venv-quant` (torch 2.14.1 / llmcompressor), `.venv-dflash`
  (torch 2.13.0 / speculators 0.8.1 / vllm 0.31.0). `vllm serve` needs the
  venv's bin on PATH.
- Everything caches via standard `~/.cache/huggingface`. No repo-local model
  litter; transient outputs are disposable and stay out of git.
- Frozen data in `data/` is load-bearing for every published number —
  `sha256sum -c` before any measurement campaign.
