# warp-3090-quant-lab

Quants and DFlash2 speculative drafters for the **warp-3090** inference engine
(custom engine for 3090/Ampere). Part of the warp-3090 project — the 24GB
3090 budget drives every choice here: W4A16 GPTQ (Ampere-safe, no fp8/nvfp4
kernels), ~15GB weights leaving ~8GB for KV, and drafters trained against the
**deployed quant** for exact-pairing acceptance.

What this repo reproduces (see `RESULTS.md` for all numbers):

- **Quants**: W4A16 GPTQ for the Qwen 3.5 release family (3.5-native, 3.6
  upgrades, 3.8-only — all one architecture class) with the cascade-v1
  calibration set (999 Nemotron-Cascade-2 conversations): weightedKL
  0.1087 (4B) / 0.0889 (9B) / 0.1264 (27B).
- **Drafters**: quant-aligned DFlash2 block-diffusion speculative drafters —
  the 4B drafter hits τ 3.79 at the serving point (interval-weighted
  acceptance, block 16, temp 0.7) / 4.05 greedy, vs full native MTP 2.86 /
  2.99 on the same ladder.

Plain scripts, no package. Two venvs (the speculators stack pins different
torch/transformers than the quant stack):

```bash
python3 -m venv .venv-quant   && .venv-quant/bin/pip  install -r requirements-quant.txt
python3 -m venv .venv-dflash  && .venv-dflash/bin/pip install -r requirements-drafter.txt
# + fp8 states (default): after the drafter venv installs, patch speculators in it:
#   SP=$(./.venv-dflash/bin/python -c 'import speculators,os;print(os.path.dirname(os.path.dirname(speculators.__file__)))') \
#     && patch -p1 -d "$SP" < patches/speculators_fp8_load_cast.patch
# + Draft-OPD post-training (optional, inert without opd_ columns; see docs/draft_opd.md):
#     patch -p1 -d "$SP" < patches/speculators_draft_opd.patch
```

## Hardware: train on Blackwell, deploy on Ampere

| machine | role |
|---|---|
| **DGX Spark (GB10)** | local dev — everything verified here first (`bash regression.sh`, <15 min, one GPU) |
| **RunPod B200/B300** | big rungs — same scripts, faster (`bash pod.sh push && pod.sh run`) |
| **3090 / Ampere** | deployment target — the quant format is the contract; acceptance is portable, throughput is not |

Why this split is safe (see `docs/history` + `RESULTS.md`): acceptance is a
distribution property, not a kernel property; W4A16 GPTQ weights are
deterministic given calibration data and serve identically on sm_86. Never
quote throughput for warp-3090 claims from Blackwell hardware — measure that
on a 3090. (Don't lean on cross-machine τ comparisons either — single-draw
τ noise swamps hardware deltas; the anchor numbers in `RESULTS.md` are all
GB10.) Attention impl per machine: `simple_flex_attention` on B200,
`sdpa` on Spark (flex Triton kernels exceed sm_121 shared memory) — see
`drafter/train.md`.

Workflow: **test locally, then push to a big box.**
`bash regression.sh` locally → `bash pod.sh push` → `bash pod.sh run '...'` →
`bash pod.sh pull results/ ...` → verify (`(cd data && sha256sum -c
MANIFEST.sha256 --quiet)` locally; hash pulled artifacts yourself) → teardown.

## Model matrix

`models.py` is the single source of truth — `--model 4b|9b|27b` everywhere:

| key | model | gen | layers/hidden | taps (aux+final) | quant | drafter |
|---|---|---|---|---|---|---|
| `4b` | Qwen/Qwen3.5-4B | 3.5 | 32 / 2560 | [2,10,17,24,29]+32 | ✅ 0.1087 | ✅ τ 3.79 |
| `0.8b` | Qwen/Qwen3.5-0.8B | 3.5 | 24 / 1024 | [2,8,12,16,20]+24 | ✅ local | ⬜ pilot |
| `9b` | Qwen/Qwen3.5-9B | 3.5 | 32 / 4096 | same depth | ✅ 0.0889 | ⬜ untested |
| `27b` | Qwen/Qwen3.8-27B | 3.8 | 64 / 5120 | [2,17,32,47,61]+64 | ✅ 0.1264 | ⬜ NO-GO |

Absolute KL/τ are **not comparable across release generations** (3.8 ≠ 3.5) —
keep results per-model. The champion recipe is validated on 4B; 9B/27B rows
stay "untested" in `RESULTS.md` until someone runs them.

## Sequence 1 — reproduce a quant (quant venv; 4B ≈ 24GB, ~2h)

```bash
python quant/quantize.py --model 4b          # gptq + cascade-v1 calib, 999x4096 (defaults)
python quant/probe.py --model 4b --quant ./Qwen3.5-4B-W4A16-GPTQ \
    --eval_set data/probe-v4.jsonl --tag repro
# expect weightedKL ≈ 0.1087; also probe data/probe-guard.jsonl (OOD check)
```

For 9B/27B: same commands with `--model 9b` / `--model 27b` (27B quant wants
~80GB or device offload; 9B probe holds base+quant resident, ~40GB).
Build new calib/probe sets with `quant/build_calib.py` (raw pool goes in
`data-raw/`, not in git).

## Sequence 2 — drafter, small scale (drafter venv; Spark or any 24GB+ GPU)

```bash
python drafter/slice_convs.py --preset small --quant ./Qwen3.5-4B-W4A16-GPTQ
vllm serve ./Qwen3.5-4B-W4A16-GPTQ --port 8138 &     # the DEPLOYED quant
python drafter/regen.py slices/s_train.jsonl slices/s_regen.jsonl slices/s_merged.jsonl --model 4b
python drafter/prep_dataset.py --jsonl slices/s_merged.jsonl --out ds_small --quant ./Qwen3.5-4B-W4A16-GPTQ
python drafter/extract_states.py --dataset ds_small --verifier ./Qwen3.5-4B-W4A16-GPTQ \
    --out ds_small --model 4b --parity 20
python drafter/bootstrap_draft.py --model 4b --quant ./Qwen3.5-4B-W4A16-GPTQ --out draft_init
# train command: drafter/train.md (small-scale section; sdpa on Spark)
bash drafter/eval_ladder.sh draft_small/0 small 15 8151      # expect τ ≈ 2.18
```

Full-corpus regen (any registered model) is one command —
`drafter/regen_pool.sh <model-key>` serves the deployed quant with
`models.py SERVING_PARAMS` and regenerates every context's final,
resume-safe; outputs land in `data/`.

The two rules that are not optional: states come from a **transformers**
forward on the quant (vLLM extraction is corrupt on W4A16 — any hardware),
and states must be **N+1 taps** (aux + final; `--model` fills this for you).

## Sequence 3 — champion scale (1×B200/B300 + ~4TB NVMe, ~$60 rental)

Same sequence with `--preset v4` (100k convs → 260M tokens; fp8 6-tap states
≈ 3.2TB — disk is the gating resource), then both train arms from
`drafter/train.md` at checkpoint-freq 1. Expect τ ≈ 3.8 at the serving point
(temp 0.7) vs native MTP ≈ 2.9 — measure the MTP baseline on your own ladder
first (`drafter/train.md`); published MTP numbers are harness-specific.
Checkpoint selection: `bash drafter/sweep_epochs.sh <checkpoints_parent>
<sweep_name>` — runs the serving-temp ladder (`TEMP=0.7`) across every epoch
and extra repeats on the top contenders, emitting a ranked table with
pooled n. τ rises monotonically to end-of-training at 0.7, but the ranking
is temperature-dependent (greedy peaks mid-run); never rank on single draws
(see the measurement protocol in `AGENTS.md`).

## Data (frozen, in git under `data/`)

Frozen **deliberately**: every published number is measured on these exact
files (weightedKL on probe-v4, τ on ladder-20), so drifting them silently
breaks cross-run comparability. Verify your checkout:

```bash
(cd data && sha256sum -c MANIFEST.sha256 --quiet && echo data OK)
```

| file | what | provenance |
|---|---|---|
| `calib-cascade-v1.jsonl` | champion calibration set (999 rows) | 100% nvidia/Nemotron-Cascade-2-SFT-Data, dataset-native proportions (verified 2026-10-16) |
| `probe-v4.jsonl` | frozen 378-row fidelity ruler (the weightedKL metric) | cascade pool per V4_TARGETS + Belebele 10×9 langs |
| `probe-guard.jsonl` | OOD guard (no_robots 40 + Belebele 4×9 langs) | no_robots **train** split (verified vs cached upstream arrows) + facebook/belebele |
| `ladder-20.jsonl` | the 20 held-out eval prompts (the τ metric) | rows 50–69 of the v1-era 100-row cascade dev set (`dflash_data_v1`), frozen since the first DFlash2 run; hash-banned from drafter training, disjoint from the probes. Prompt texts recur inside calib rows (both cascade-derived) — affects no quoted number |
| `conv-ids/` | conversation-id manifests for the v3/v4/v5 training slices | exact `slice_convs.py` preset outputs — deterministically reproducible (seeds, per-subset rounding, val sampling all verified against the manifests) |
| `MANIFEST.sha256` | hash pin for everything above | `find . -type f ! -name MANIFEST.sha256 \| sort \| sed 's|^\./||' \| xargs sha256sum` (prefix-free so `sha256sum -c` and the uploader's set check both round-trip) |

**Where each set is used** — three processes, three roles, deliberately
walled off (all frozen sets pairwise hash-disjoint; builders enforce it):

| process | uses | notes |
|---|---|---|
| quantization (`quant/quantize.py`) | `calib-cascade-v1` only | activations decide GPTQ grouping; no_robots/Belebele never enter — keeps the guard honest |
| drafter training (`drafter/slice_convs.py` → `regen.py` → …) | cascade pool prompts via `conv-ids` presets | slices hash-banned from every frozen set; `regen.py` rewrites assistant turns with the deployed quant — training data is quant-written completions + quant states, never cascade's original assistant text, never bf16 |
| evaluation (`quant/probe.py`, `drafter/eval_ladder.sh`) | `probe-v4` → headline weightedKL; `probe-guard` → OOD anti-gaming check; `ladder-20` → headline τ + per-epoch checkpoint selection | ladder-20 is hash-banned from training; probe-guard is consumed by nothing — pure held-out referee |

One documented exception: ladder-20 prompt texts recur inside calib rows (both
cascade-derived) — τ compares drafters against the same target model, so the
overlap can't game it. `regression.sh` regenerates its own smoke dataset and
never reads `data/`.

**Reproducibility contract.** `quant/build_calib.py` builds *new* sets of each
calib/probe/guard kind — always hash-disjoint from the frozen ones (freshness
by design, not regeneration). Every build writes a `<out>.sources.json` sidecar
pinning each upstream file it read (dataset revision + SHA256), so silent
upstream drift (HF reshards, split-order flips) is detectable by diffing
sidecars. The no_robots split is explicit (`--no-robots-split`, default `train`
= the frozen set's actual source split, verified against the cached upstream
arrows; the converted-parquet API has served split lists in unstable order). The frozen sets pre-date the sidecar mechanism and
are pinned by `MANIFEST.sha256` + the HF mirror below.

**Licenses** (check before redistributing): cascade-derived sets → NVIDIA Open
Model License; probe-guard no_robots rows → CC-BY-NC-4.0 (non-commercial);
Belebele rows → CC-BY-SA-4.0. Regen outputs are Qwen model outputs.

**HF mirror:** the frozen sets are dual-homed at
`nchapman/warp-3090-quant-lab-frozen-data` (**public**, CC-BY-NC-4.0 top-level
per the card — non-commercial research use; see license note above). Pinned at
commit `8e38aa4`, verify against
`data/MANIFEST.sha256` after pulling). Good reason to freeze *on the Hub*
rather than git alone: revision-hash citability, pull-by-pods without a repo
clone, and a license/provenance surface git can't provide.

**Published models (one recipe each, no version zoo):** the champion quants
live at the `quant_repo` ids in `models.py` (`nchapman/Qwen3.5-4B-W4A16-GPTQ`,
`…-9B…`, `nchapman/Qwen3.8-27B-W4A16-GPTQ` — each card pins the calibration
generation as `calib-cascade-v1`). The champion drafter lives at
`nchapman/Qwen3.5-4B-DFlash2` (standard speculators naming — model it drafts
for, no quant marker: measured drafting against both the quant verifier and
the bf16 base; weights sha256 in the card). Earlier drafter generations
(v3, v3.5) and the v4 per-epoch snapshots stay as local research record
under `../llm-compressor/artifacts/` (read-only archive — never written to).
The regen context pool is IN this repo (`data/context_pool.jsonl`,
sha-pinned) and per-model regen outputs land in `data/` (gitignored,
pinned when frozen).

**Naming rule:** quants are named for what they are —
`Qwen3.5-<size>-W4A16-GPTQ` — never for the calibration experiment that
produced them. (A `-cascade` dir suffix existed historically to mark the
calib generation; retired 2026-10-16 — historical dirs keep their names,
nothing new gets one. The citable calib version is the frozen-data pin.)
Dataset files are named for their purpose (`calib-*`, context pool, regen
outputs) with sources documented in the table above.

**Models & cache:** Hub ids are the canonical model references (`models.py`
resolves them); anything pulled from the Hub lands in the standard
`~/.cache/huggingface` — the repo sets no custom cache dirs and keeps no
local model copies. `quantize.py` outputs and eval working copies
(`eval_*/`, `Qwen3*/`) are transient, gitignored, and disposable: upload what
you keep, delete the rest.

## Run logging standard (no more guessing)

Every script and tool in this repo logs the same way — the process failures
this project has hit (silent input mismatch, wedged servers, results in
unexpected places) are failure modes of unmanaged runs, not of the models.

- **Every run** — evals, regression, quantize, probe, calib builds — gets
  `runs/<name>.<UTC ts>/` with `cmd.txt` (exact invocation) and
  `events.jsonl` (structured lifecycle: `RUN_START`, `PREFLIGHT_OK/FAIL`,
  `SERVER_LAUNCH/HEALTHY/STUCK/FAILED`, stage markers, `RUN_DONE/RUN_FAILED`).
- **Preflight or die**: inputs are validated by name before anything launches
  (`require_file`).
- **Watchdogged servers**: health waits fail fast — a server whose log goes
  idle *and* whose GPU is 0% for two consecutive probes is `SERVER_STUCK`,
  killed by its recorded PID, with a process snapshot in the log. No
  15-minute silent hangs. (`EXTRA_SERVE_ARGS="--enforce-eager"` dodges
  cudagraph-capture wedges when throughput doesn't matter.)
- **Servers die by PID** — never `pkill` scatter.
- Inspect any run: `bash scripts/tail_run.sh [name-prefix]`.

`lib/runlib.sh` is the bash implementation (sourced by `regression.sh`,
`drafter/eval_ladder.sh`, `drafter/eval_mtp.sh`); `quant/lablog.py` is the
python mirror (`probe.py`, `quantize.py`, `build_calib.py`). New scripts
must source/import one of them — no exceptions.

## Layout

```
models.py            family registry (dims, taps, hub ids) — single source of truth
quant/               build_calib.py · quantize.py · probe.py
drafter/             slice_convs.py · regen.py · regen_pool.sh (standing per-model regen) ·
                     prep_dataset.py · extract_states.py · reshape_tokens.py · slice_rungs.py ·
                     data_ladder.sh · sweep_epochs.sh · bootstrap_draft.py · eval_ladder.sh ·
                     train.md · draft_config_*.json
data/                frozen calib/probe/ladder/conv-ids + context_pool.jsonl (pinned) +
                     regen outputs (gitignored, pinned when frozen) — the reproducible half of every number
patches/             speculators fp8 load-cast patch
docs/history/        verbatim research notebooks (the full v1→v5 story + postmortems)
regression.sh        100-row end-to-end pre-flight (<15 min, one GPU)
pod.sh               push/run/pull helper for rented Blackwell pods
RESULTS.md           every measured number, per model
```

Provenance: extracted from the `quant-lab` branch of our llm-compressor fork
(no library modifications — llm-compressor is consumed as a pinned dependency).
