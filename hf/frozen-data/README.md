---
license: cc-by-nc-4.0
language:
  - en
  - de
  - es
  - fr
  - it
  - ja
  - zh
  - ar
  - hi
  - ru
pretty_name: warp-3090 frozen evaluation and calibration sets
size_categories:
  - n<1K
---

# warp-3090 frozen data

The frozen calibration, probe, and ladder sets underpinning every published
number in the [warp-3090-quant-lab](https://github.com/nchapman/warp-3090-quant-lab)
repo (W4A16 GPTQ quants + DFlash2 speculative drafters for Qwen 3.5-family
models). These files are **deliberately frozen**: the headline metrics are
defined on these exact rows —

- **calib-cascade-v1.jsonl** (999 rows) — the champion GPTQ calibration set
  (weightedKL 0.1087 @ 4B, 0.0889 @ 9B, 0.1264 @ 27B trace to it)
- **probe-v4.jsonl** (378 rows) — the fidelity ruler; the weightedKL metric
- **probe-guard.jsonl** (76 rows) — OOD guard (no_robots + Belebele, 9 languages)
- **ladder-20.jsonl** (20 prompts) — the acceptance ladder; the τ metric
  (champion τ 3.443 b16, native MTP 2.71 re-anchored on it)
- **conv-ids/** — conversation-id manifests of the v3/v4/v5 drafter training
  slices (deterministic outputs of the repo's slice_convs.py presets)

## Provenance

| file | origin |
|---|---|
| calib-cascade-v1 | [nvidia/Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data), stratified at the card's own domain blend, seed 0 |
| probe-v4 | cascade pool (v4 domain allocation) + [facebook/belebele](https://huggingface.co/datasets/facebook/belebele) 10 rows × 9 languages |
| probe-guard | no_robots **train** split (40, verified against cached upstream arrows) + Belebele 4 × 9 languages |
| ladder-20 | rows 50–69 of the v1-era 100-row cascade dev set, held out from drafter training since the first DFlash2 run (2025-10-10); prompts-only (system+user) |
| conv-ids | deterministic outputs of the repo's slicing presets |

Integrity: `MANIFEST.sha256` (sha256sum format) pins every file; verify with
`sha256sum -c MANIFEST.sha256` after download.

## Licenses

This is a derived, research-only collection; per-component terms apply and are
carried here for transparency:

- **calib-cascade-v1, probe-v4 (cascade rows), ladder-20** — derived from
  Nemotron-Cascade-2-SFT-Data: [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
- **probe-guard (no_robots rows)** — derived from
  [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots):
  **CC-BY-NC-4.0** (non-commercial)
- **probe-v4 / probe-guard (Belebele rows)** — derived from
  [facebook/belebele](https://huggingface.co/datasets/facebook/belebele):
  CC-BY-SA-4.0

## Reproduction

These sets are *anchors, not regenerable outputs*: the repo's builders
construct **new** sets guaranteed hash-disjoint from these (freshness by
design) and write a `.sources.json` provenance sidecar (dataset revision +
file SHA256) for every build. See `quant/build_calib.py` and the repo README's
data section for the full reproducibility contract.
