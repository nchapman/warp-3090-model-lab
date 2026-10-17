---
base_model: ornith-ai/Ornith-1.5-35B-A3B
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- w4a16
- compressed-tensors
- vllm
- moe
- vision
---

# Ornith-1.5-35B-A3B-W4A16-GPTQ

INT4 weight-only GPTQ quantization of
[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
(35B MoE, ~3B active, vision-language) for single-GPU serving.

**Language-model weights only** — the vision tower, projector, and all
`mtp.*` speculative-decoding weights are carried over at BF16, untouched.

## Quantization recipe

- W4A16 (INT4 weights, BF16 activations), group 128, act-order,
  compressed-tensors (pack-quantized) — vLLM Marlin kernels
- Calibrated with [llm-compressor](https://github.com/vllm-project/llm-compressor) oneshot
- Calibration set: **`calib-cascade-v1`** — 999×4096-token windows at
  dataset-native proportions from
  [nvidia/Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data)
  The exact rows are hash-pinned in
  [warp-3090-quant-lab-frozen-data](https://huggingface.co/datasets/nchapman/warp-3090-quant-lab-frozen-data)
  — that pin is the citable calibration generation.

## Checksums

See [`SHA256SUMS`](SHA256SUMS) in this repo. model.safetensors shard checksums:

```
{{SHA256_BLOCK}}
```

Provenance: quantized on a single B200 via the
[warp-3090-quant-lab](https://github.com/nchapman/warp-3090-quant-lab)
champion recipe (unchanged from the 4B release); run log
`quant-35b-ornith.20261017T114909Z.13535` (2026-10-17).

## Usage (vLLM)

```python
from vllm import LLM
llm = LLM(model="nchapman/Ornith-1.5-35B-A3B-W4A16-GPTQ")
```
