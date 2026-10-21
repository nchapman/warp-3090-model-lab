---
base_model: Qwen/Qwen3.6-35B-A3B
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- w4a16
- compressed-tensors
- vllm
- moe
---

# Qwen3.6-35B-A3B-W4A16-GPTQ

INT4 weight-only GPTQ quantization of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
(35B MoE, ~3B active) for single-GPU serving.

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
- MTP (`mtp.*`), vision tower (`.*visual.*`), shared-expert and gated-delta
  linear-attention projections, and `lm_head` kept at BF16.

## Checksums

See [`SHA256SUMS`](SHA256SUMS) in this repo. model.safetensors shard checksums:

```
3fa42b79baa19dedc0b872462af27d7f600e964cd7ce47a39b6ea1ea58e779b0  model-00001-of-00002.safetensors
1631b73be95d7bac4e01772e1d53be42e37445dd710c63ea477280e45f95fff1  model-00002-of-00002.safetensors
68b2cabe2ff84bf3ba0fde0af082795e175a7ab427cb78e3b53df96e1d4dc929  model_mtp.safetensors (BF16 MTP, unquantized)
```

Provenance: quantized on a single B200 via the
[warp-3090-quant-lab](https://github.com/nchapman/warp-3090-quant-lab)
champion recipe (unchanged from the 4B release); run log
`quant-35b.20261017T065050Z.2036` (2026-10-17).

## Usage (vLLM)

```python
from vllm import LLM
llm = LLM(model="nchapman/Qwen3.6-35B-A3B-W4A16-GPTQ")
```
