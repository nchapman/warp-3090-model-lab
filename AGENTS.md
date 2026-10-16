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

## Corpus & capability facts (MEASURED — do not re-derive, do not assume)

**Data currency.** Always state amounts in **unique supervised tokens**
(loss_mask=1, once per conv), with conversations as the diversity co-unit.
Nominal row-tokens ≠ unique ≠ supervised: rung 1's "10M" was 7.5M unique
(25% prefix duplication) = **5.2M supervised** (1.93× exchange).
`drafter/data_audit.py` measures all of these — run it, never guess.

**Corpus** (`../llm-compressor/artifacts/v4_data/regen_all.jsonl`):
145,018 rows / ~100K convs / 260M nominal ≈ **115–120M supervised**.
Rows are conversation-turn PREFIXES (`v4-<i>-turnN`); turn-N contains
turns 1..N (early turns re-supervised, contexts re-extracted).
- Mix (measured, not labels): chat 65% / math 12.5% / science 8% /
  IF 5% / conv-agent 5% / terminal-agent 3.6%; **code ≈ 6%** (fence+
pattern heuristic, 3K-row sample — same as PerfectBlend's 7% code source;
the divergence is MATH 12.5 vs their 39, CHAT 65 vs their ~30).
- Length: mean 2,115 / median 1,676 / p90 4,588 / p99 6,207 tokens —
**uncapped**; DFlash paper spec caps at 3072. DSpark reference preps ≤1024.
- Shape tooling: `drafter/reshape_tokens.py` (prefix/merged/first/short),
`prep_dataset.py --loss-mode all|last --max-tokens N` (drop cap).

**Eval.** `data/ladder-20.jsonl` is GENERAL CHAT ONLY — every τ ever
quoted (incl. champion 3.79) is general-chat τ. No code/agentic/long
eval exists. Don't claim deployment relevance without one.

**Measured anchors (τ @ t=0.7 n=4 unless noted):** 4B 5.2M supervised →
**1.05 flat** (dead — coverage collapse, recall@16 0.30); 4B ~117M →
**3.79**; epoch-0 control 3.407. External: DSpark 100K prompts ≤1024 →
3.41; DFlash 100K-sample ablations → 4.9–5.6 greedy. Dead zone < ~5M
supervised (two sources); known-good 20–60M; rising to ~1B (DFlare).

**Trainer behavior.** Multipack sampler re-shuffles per epoch (seed+epoch)
→ epoch step-counts vary (rung 1: 1,310–1,850+); `step_ms` excludes
inter-step overhead — wall is ~1.7–2× step_ms. Plan in hours/epoch
(GB10: ~3.5h per 10M nominal), never in step counts.

**GB10 reality.** Bandwidth-starved (~17.6× off H200): extraction and
training both ~2k tok/s class. nvidia-smi memory reads [N/A]; 30–70W
at 96% util is normal load, not idle. A hung nvidia-smi can hold
/dev/nvidia0 — kill by exact PID (never pkill patterns).

**Models registry.** `models.py` keys: 4b (champion, taps_v4
[2,10,17,24,29,32]), 0.8b (testbed, taps [2,8,12,16,20,24], local quant
`Qwen3.5-0.8B-W4A16-GPTQ-cascade/`), 9b, 27b (taps UNVALIDATED).
Quant recipe: `quant/run_quant.sh <key>` (W4A16 GPTQ, cascade-v1 calib).

**Literature idiom (verified from primary sources).** On-policy regen at
eval temp; one conv per sample (first-turn common); short or capped
lengths; PerfectBlend = ecosystem default mix, NOBODY ablates mix;
`minimum-valid-tokens` filter (speculators) — we lack it.
