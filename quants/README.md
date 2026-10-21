# quants/ — W4A16 GPTQ quantization workflow

Everything that turns a stock Qwen 3.5-family checkpoint into the deployed
`*-W4A16-GPTQ` artifact the rest of the lab builds against. Ampere-safe by
design (no fp8/nvfp4 kernels) — the quant format is the warp-3090 contract.

**Venv:** `.venv-quant` (torch 2.14.1 / llmcompressor). Not `.venv-dflash`.

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

## Conventions

- Local artifacts are **bare root dirs** matching `*-W4A16-GPTQ*` at the repo
  root (gitignored, disposable). Quants are named for what they are, never
  for the calib experiment that produced them — the citable calib version is
  the frozen-data pin, not a dir suffix.
- `models.py` is the registry: `--model` keys, hub ids, SERVING_PARAMS
  (stamped into every quant's `generation_config.json` at quant time).
- Reproduce end-to-end: README **Sequence 1**; numbers: `RESULTS.md`;
  frozen-set provenance and the wall between calib/probe/ladder sets:
  top-level README "Data".
