# warp-3090-quant-lab

Quants and DFlash2 speculative drafters for the **warp-3090** inference engine
(custom engine for 3090/Ampere). Part of the warp-3090 project — the 24GB
3090 budget drives every choice here: W4A16 GPTQ (Ampere-safe, no fp8/nvfp4
kernels), ~15GB weights leaving ~8GB for KV, and drafters trained against the
**deployed quant** for exact-pairing acceptance.

What this repo reproduces (see `RESULTS.md` for all numbers):

- **Quants**: W4A16 GPTQ for the Qwen 3.5 release family (3.5-native, 3.6
  upgrades, 3.8-only — all one architecture class) with the cascade
  calibration blend: weightedKL 0.1087 (4B) / 0.0889 (9B) / 0.1264 (27B).
- **Drafters**: quant-aligned DFlash2 block-diffusion speculative drafters —
  the 4B champion hits τ 3.443 (interval-weighted acceptance, block 16) vs
  full native MTP 2.71 on the same ladder.

Plain scripts, no package. Two venvs (the speculators stack pins different
torch/transformers than the quant stack):

```bash
python3 -m venv .venv-quant   && .venv-quant/bin/pip  install -r requirements-quant.txt
python3 -m venv .venv-dflash  && .venv-dflash/bin/pip install -r requirements-drafter.txt
# + fp8 states (default): patch speculators with patches/speculators_fp8_load_cast.patch
```

## Hardware: train on Blackwell, deploy on Ampere

| machine | role |
|---|---|
| **DGX Spark (GB10)** | local dev — everything verified here first (`bash regression.sh`, <15 min, one GPU) |
| **RunPod B200/B300** | big rungs — same scripts, faster (`bash pod.sh push && pod.sh run`) |
| **3090 / Ampere** | deployment target — the quant format is the contract; acceptance is portable, throughput is not |

Why this split is safe (measured, see `docs/history`): the same checkpoint
scored τ 3.250 trained/served on B200 vs 3.242 on GB10 (σ≈0.01 — acceptance
is a distribution property, not a kernel property); W4A16 GPTQ weights are
deterministic given calibration data and serve identically on sm_86. Never
quote throughput for warp-3090 claims from Blackwell hardware — measure that
on a 3090. Attention impl per machine: `simple_flex_attention` on B200,
`sdpa` on Spark (flex Triton kernels exceed sm_121 shared memory) — see
`drafter/train.md`.

Workflow: **test locally, then push to a big box.**
`bash regression.sh` locally → `bash pod.sh push` → `bash pod.sh run '...'` →
`bash pod.sh pull results/ ...` → verify SHA256s → teardown.

## Model matrix

`models.py` is the single source of truth — `--model 4b|9b|27b` everywhere:

| key | model | gen | layers/hidden | taps (aux+final) | quant | drafter |
|---|---|---|---|---|---|---|
| `4b` | Qwen/Qwen3.5-4B | 3.5 | 32 / 2560 | [2,10,17,24,29]+32 | ✅ 0.1087 | ✅ τ 3.443 |
| `9b` | Qwen/Qwen3.5-9B | 3.5 | 32 / 4096 | same depth | ✅ 0.0889 | ⬜ untested |
| `27b` | Qwen/Qwen3.8-27B | 3.8 | 64 / 5120 | [2,17,32,47,61]+64 | ✅ 0.1264 | ⬜ NO-GO |

Absolute KL/τ are **not comparable across release generations** (3.8 ≠ 3.5) —
keep results per-model. The champion recipe is validated on 4B; 9B/27B rows
stay "untested" in `RESULTS.md` until someone runs them.

## Sequence 1 — reproduce a quant (quant venv; 4B ≈ 24GB, ~2h)

```bash
python quant/quantize.py --model 4b          # gptq + cascade calib, 999x4096 (defaults)
python quant/probe.py --model 4b --quant ./Qwen3.5-4B-W4A16-GPTQ-cascade \
    --eval_set data/probe-v4.jsonl --tag repro
# expect weightedKL ≈ 0.1087; also probe data/probe-guard.jsonl (OOD check)
```

For 9B/27B: same commands with `--model 9b` / `--model 27b` (27B quant wants
~80GB or device offload; 9B probe holds base+quant resident, ~40GB).
Build new calib/probe sets with `quant/build_calib.py` (raw pool goes in
`data-raw/`, not in git).

## Sequence 2 — drafter, small scale (drafter venv; Spark or any 24GB+ GPU)

```bash
python drafter/slice_convs.py --preset small --quant ./Qwen3.5-4B-W4A16-GPTQ-cascade
vllm serve ./Qwen3.5-4B-W4A16-GPTQ-cascade --port 8138 &     # the DEPLOYED quant
python drafter/regen.py slices/s_train.jsonl slices/s_regen.jsonl slices/s_merged.jsonl --model 4b
python drafter/prep_dataset.py --jsonl slices/s_merged.jsonl --out ds_small --quant ./Qwen3.5-4B-W4A16-GPTQ-cascade
python drafter/extract_states.py --dataset ds_small --verifier ./Qwen3.5-4B-W4A16-GPTQ-cascade \
    --out ds_small --model 4b --parity 20
python drafter/bootstrap_draft.py --model 4b --quant ./Qwen3.5-4B-W4A16-GPTQ-cascade --out draft_init
# train command: drafter/train.md (small-scale section; sdpa on Spark)
bash drafter/eval_ladder.sh draft_small/0 small 15 8151      # expect τ ≈ 2.18
```

The two rules that are not optional: states come from a **transformers**
forward on the quant (vLLM extraction is corrupt on W4A16 — any hardware),
and states must be **N+1 taps** (aux + final; `--model` fills this for you).

## Sequence 3 — champion scale (1×B200/B300 + ~4TB NVMe, ~$60 rental)

Same sequence with `--preset v4` (100k convs → 260M tokens; fp8 6-tap states
≈ 3.2TB — disk is the gating resource), then both train arms from
`drafter/train.md` at checkpoint-freq 1, **ladder-eval every epoch**, keep the
best (champion = armB epoch 2). Expected: τ 3.443 b16 / 3.131 b8 vs native
MTP 2.71. Measure the MTP baseline on your own ladder first
(`drafter/train.md`) — published MTP numbers are harness-specific.

## Data (frozen, in git under `data/`)

| file | what |
|---|---|
| `calib-cascade-v1.jsonl` | champion calibration set (999 rows) |
| `probe-v4.jsonl` | frozen 378-row fidelity ruler (the weightedKL metric) |
| `probe-guard.jsonl` | OOD guard (no_robots + Belebele, 9 langs) |
| `ladder-20.jsonl` | the 20 held-out eval prompts (the τ metric) |
| `conv-ids/` | conversation-id manifests for the v3/v4/v5 training slices |

Derived from nvidia/Nemotron-Cascade-2-SFT-Data (+ no_robots / Belebele for
guards); regen outputs are Qwen model outputs — check source licenses before
redistributing. Big artifacts (v4 regen jsonls, champion checkpoint) go to the
HF Hub under warp-3090 naming, not git.

## Layout

```
models.py            family registry (dims, taps, hub ids) — single source of truth
quant/               build_calib.py · quantize.py · probe.py
drafter/             slice_convs.py · regen.py · prep_dataset.py · extract_states.py
                     bootstrap_draft.py · eval_ladder.sh · train.md · draft_config_*.json
data/                frozen calib/probe/ladder/conv-ids (the reproducible half of every number)
patches/             speculators fp8 load-cast patch
docs/history/        verbatim research notebooks (the full v1→v5 story + postmortems)
regression.sh        100-row end-to-end pre-flight (<15 min, one GPU)
pod.sh               push/run/pull helper for rented Blackwell pods
RESULTS.md           every measured number, per model
```

Provenance: extracted from the `quant-lab` branch of our llm-compressor fork
(no library modifications — llm-compressor is consumed as a pinned dependency).
