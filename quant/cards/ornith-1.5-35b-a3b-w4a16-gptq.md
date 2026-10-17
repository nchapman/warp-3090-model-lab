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
```
182e77dd83bd8e9ca818b240b82e28f243762cd5dda32e6eef327df7b1cd107e  chat_template.jinja
fda8dd193e8dfe66232f01d7719ed679da765952cf8e7b9bbe45af6557a19de1  config.json
9eb15d950e7d08cf1b6ae62e40cde27abf60574f62c86b64a2d56aeeba2a6183  generation_config.json
1a7a211fcd89a7d48d07cf238ce824e94ef1b5884e8d85184c3a467492a46ac0  model-00001-of-00002.safetensors
efcafa741d4b8f6a8bf5ebfe2824e3cc20f3680912392d9413fe9486c7bb3a7f  model-00002-of-00002.safetensors
47abfc9963df5b1f18c1afc891bd8a294a9d40013daac03fbfed1cc54c67d3c6  model.safetensors.index.json
57e5159f86cc9e9d902f5f376532ccee1cdc96be3aef822601dc9009fc613a1f  model_mtp.safetensors
bfbc24af59a3e73a9cd0653b8d4ae758dfaec4e3e6c15dfb1c6c8ee8d5683c85  processor_config.json
4e536ddcd36d982387be242e8789c54c05a10d900fe056b84394155dd389c2bf  recipe.yaml
67d8113912fabc4af848be59c550922fb4945e3f9ad0853948d5def219fc4e46  tokenizer.json
7ff0211d20c398343cab1300beeb9e038ece910498ff279ecda9e3e5fd41026b  tokenizer_config.json
```
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
