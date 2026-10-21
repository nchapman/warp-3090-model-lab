# quants/ — W4A16 GPTQ quantization workflow

Everything that turns a stock Qwen 3.5-family checkpoint into the deployed
`*-W4A16-GPTQ` artifact the rest of the lab builds against. Ampere-safe by
design (no fp8/nvfp4 kernels) — the quant format is the warp-3090 contract.

## Venv

```bash
# from the repo root
python3 -m venv .venv-quant && .venv-quant/bin/pip install -r quants/requirements-quant.txt
```

torch 2.14.1 / llmcompressor. (The drafter venv is separate — see
`drafters/README.md`; the speculators stack pins different torch.)

## Entry points (in pipeline order)

| tool | role |
|---|---|
| `build_calib.py` | build new calib/probe/guard sets from raw pools (`data-raw/`); `--mode v1-repro` rebuilds frozen `calib-cascade-v1` byte-exact and self-verifies |
| `run_quant.sh <key>` | logged quant run (`lib/runlib.sh` lifecycle) — the way to quantize, not raw `quantize.py` |
| `quantize.py --model 4b\|9b\|27b` | the quantizer itself (GPTQ, cascade-v1 calib defaults) |
| `probe.py` | fidelity ruler — weightedKL on frozen `data/probe-v4.jsonl` + OOD guard on `probe-guard.jsonl` |
| `traj_probe.py` | rollout agreement on frozen `data/traj-v1.jsonl` (greedy 32-token trajectory match — the compounding regime teacher-forced KL can't see) |
| `verify_quant.py` | artifact sanity: weights/config/tokenizer completeness |
| `capability_battery.sh` + `battery_report.py` | multi-capability eval battery over a set of quants |
| `publish_quant.py` + `cards/` | package + card for the Hub (one recipe per artifact; `model.safetensors` sha256 in the card) |

## Reproduce a quant (≈24GB VRAM, ~2h for 4B)

```bash
python quants/quantize.py --model 4b          # gptq + cascade-v1 calib, 999x4096 (defaults)
python quants/probe.py --model 4b --quant ./Qwen3.5-4B-W4A16-GPTQ \
    --eval_set data/probe-v4.jsonl --tag repro
# expect weightedKL ≈ 0.1087; also probe data/probe-guard.jsonl (OOD check)
```

For 9B/27B: same commands with `--model 9b` / `--model 27b` (27B quant
wants ~80GB or device offload; 9B probe holds base+quant resident, ~40GB).
Build new calib/probe sets with `build_calib.py` (raw pool goes in
`data-raw/`, not in git). Frozen-set provenance, the walls between
calib/probe/ladder, and the byte-exact v1-repro contract: `data/README.md`.
Numbers: `RESULTS.md`.

## Conventions

- Local artifacts are **bare root dirs** matching `*-W4A16-GPTQ*` at the
  repo root (gitignored, disposable). Quants are named for what they are —
  `Qwen3.5-<size>-W4A16-GPTQ` — never for the calib experiment that
  produced them; the citable calib version is the frozen-data pin
  (`data/README.md`), not a dir suffix.
- `models.py` (repo root) is the registry: `--model` keys, hub ids,
  SERVING_PARAMS (stamped into every quant's `generation_config.json` at
  quant time).
- Published quants live at the `quant_repo` ids in `models.py` — one
  recipe each, no version zoo.
