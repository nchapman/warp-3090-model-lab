# warp-3090-model-lab

Quants and DFlash2 speculative drafters for the **warp-3090** inference engine
(custom engine for 3090/Ampere). Part of the warp-3090 project — the 24GB
3090 budget drives every choice here: W4A16 GPTQ (Ampere-safe, no fp8/nvfp4
kernels), ~15GB weights leaving ~8GB for KV, and drafters trained against the
**deployed quant** for exact-pairing acceptance.

What this repo reproduces (numbers: `RESULTS.md`):

- **Quants**: W4A16 GPTQ for the Qwen 3.5 release family with the
    cascade-v1 calibration set — weightedKL 0.1087 (4B) / 0.0889 (9B) /
    0.1264 (27B).
- **Drafters**: quant-aligned DFlash2 block-diffusion speculative drafters —
    the 4B drafter hits τ 3.79 at the serving point (temp 0.7) vs full
    native MTP 2.86 on the same ladder.

Plain scripts, no package. Each subproject is self-contained (code, venv
requirements, patches, docs); a small shared core (registry, run plumbing,
frozen data, pod ops) lives at the top level. Lab doctrine — measurement
protocol, process rules, corpus truth, per-machine footguns — is
`AGENTS.md`, the operating manual for agents and humans alike.

## Subprojects

| dir | what |
|---|---|
| **[quants/](quants/)** | W4A16 GPTQ workflow: calib build → quantize → verify/probe → capability battery → publish. Intro, venv, and the reproduce-a-quant sequence in its README. |
| **[drafters/](drafters/)** | DFlash2 drafter pipeline: slice → on-policy regen (against the deployed quant) → prep → state extraction → train → select → measure τ. `train.md` is the training manual; `data_scaling.md` and `external_references.md` are its deep docs. |
| **[finetune/](finetune/)** | Managed/external fine-tuning of stock models (scaffold; charter + platform survey `compute-platforms.md` in-dir). Never the local speculators stack. |
| **[experiments/](experiments/)** | Closed campaigns — code + docs + records frozen together. Historical receipts, read not run. |

## Shared core

| path | role |
|---|---|
| `models.py` | family registry (dims, taps, hub ids, SERVING_PARAMS) — single source of truth for `--model` keys |
| **[lib/](lib/)** | run plumbing every script must use: `runlib.sh` (bash) + `lablog.py` (python) — run lifecycle, watchdogs, events |
| **[data/](data/)** | frozen rulers (calib/probe/guard/traj/ladder-20) + pinned regen pool; the reproducible half of every number |
| **[scripts/](scripts/)** | pod & repo ops: `pod_prep.sh`, `podsh.py`, `pod_pull_pass.sh`, `tail_run.sh`, `upload_frozen_data.py` |
| `RESULTS.md` | every measured number, per model |
| `AGENTS.md` | lab doctrine (τ measurement, process rules, machine footguns, corpus truth) |
| `regression.sh` · `pod.sh` | end-to-end pre-flight (<15 min, one GPU) · push/run/pull for rented pods |
| `hf/` · `issues/` | Hub mirror cards · open engineering issues |

## Quickstart

```bash
python3 -m venv .venv-quant   && .venv-quant/bin/pip  install -r quants/requirements-quant.txt
python3 -m venv .venv-dflash  && .venv-dflash/bin/pip install -r drafters/requirements-drafter.txt
# drafter venv needs the fp8 patch (see drafters/README.md#venv), then:
bash regression.sh                       # end-to-end pre-flight, Spark <15 min
(cd data && sha256sum -c MANIFEST.sha256 --quiet && echo "data OK — complete copy")   # gate: fails if any pinned file is absent/corrupt; fresh clone: see data/README.md
```

## Hardware: train on Blackwell, deploy on Ampere

| machine | role |
|---|---|
| **DGX Spark (GB10)** | local dev — everything verified here first |
| **RunPod B200/B300** | big rungs — same scripts, faster (`bash pod.sh push && pod.sh run`) |
| **3090 / Ampere** | deployment target — the quant format is the contract; acceptance is portable, throughput is not |

Acceptance is a distribution property, not a kernel property; W4A16 GPTQ
weights are deterministic given calibration data and serve identically on
sm_86. Never quote throughput for warp-3090 claims from Blackwell hardware,
and don't lean on cross-machine τ comparisons (single-draw noise swamps
hardware deltas; `RESULTS.md` anchors are GB10). Attention impl per machine:
`simple_flex_attention` on B200, `sdpa` on Spark — `drafters/train.md`.

## Model matrix

| key | model | gen | taps (aux+final) | quant | drafter |
|---|---|---|---|---|---|
| `4b` | Qwen/Qwen3.5-4B | 3.5 | [2,10,17,24,29]+32 | ✅ 0.1087 | ✅ τ 3.79 |
| `0.8b` | Qwen/Qwen3.5-0.8B | 3.5 | [2,8,12,16,20]+24 | ✅ local | ⬜ pilot |
| `9b` | Qwen/Qwen3.5-9B | 3.5 | same depth | ✅ 0.0889 | ⬜ untested |
| `27b` | Qwen/Qwen3.8-27B | 3.8 | [2,17,32,47,61]+64 | ✅ 0.1264 | ⬜ NO-GO |

Absolute KL/τ are **not comparable across release generations** — keep
results per-model; untested rows stay untested in `RESULTS.md` until
someone runs them.

## Conventions

Untracked (gitignored): `runs/` `results/` `slices/` `states/` `ds_*/`
`draft_*/` `eval_*/` `data-raw/` `smoke/` + local quant artifacts as bare
root dirs `*-W4A16-GPTQ*`. Hub pulls land in the standard
`~/.cache/huggingface`; the repo keeps no local model copies.
Names are what things are (`Qwen3.5-<size>-W4A16-GPTQ`), never the
experiment that produced them — details in each subproject README.

Historical receipts (`experiments/*/records/`, pod snapshots) may cite
pre-reorg paths (`drafter/`, `quant/`, old root `quants/` artifact dir) —
frozen evidence, not living docs; don't rewrite them.

Provenance: extracted from the `quant-lab` branch of our llm-compressor fork
(no library modifications — llm-compressor is consumed as a pinned dependency).
