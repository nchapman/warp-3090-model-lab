---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- w4a16
- compressed-tensors
- vllm
---

# Qwen3.8-27B-W4A16-GPTQ

INT4 weight-only GPTQ quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
for memory-constrained single-GPU serving (fits a 24GB card with room for KV cache).

## Quantization recipe

- W4A16 (INT4 weights, BF16 activations), group 128, act-order,
  compressed-tensors (pack-quantized) — vLLM Marlin kernels
- Calibrated with [llm-compressor](https://github.com/vllm-project/llm-compressor)
- Calibration set: **`calib-cascade-v1`** — 999 conversations sampled at
  dataset-native proportions from
  [nvidia/Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data)
  (~2.6M effective tokens: chat / math / science / instruction-following /
  conversational-agent / terminal-agent / SWE). The exact rows are hash-pinned
  in
  [warp-3090-quant-lab-frozen-data](https://huggingface.co/datasets/nchapman/warp-3090-quant-lab-frozen-data)
  — that pin is the citable calibration generation. (A `-cascade` suffix on
  local paths in the [lab repo](https://github.com/nchapman/warp-3090-quant-lab)
  refers to the same generation.)

## Fidelity vs BF16 base (378-conv held-out probe, 7 domains + 9 languages)

- Perplexity delta: **+1.00%** aggregate; **-0.52%** on the fully
  out-of-distribution guard set (human-written chat + multilingual reading
  comprehension)
- Per-domain KL(base‖quant): chat 0.033, IF 0.053, math 0.098, science 0.097,
  multilingual (9 langs) 0.109, terminal-agent 0.486
- Chat template: froggeric v22.5 (`qwen3.8-froggeric-v22.5`);
  `generation_config.json` carries the base model's recommended sampling
  (temperature 1.0, top_p 0.95, top_k 20, do_sample)

## Usage (vLLM)

```python
from vllm import LLM
llm = LLM(model="nchapman/Qwen3.8-27B-W4A16-GPTQ")
```

## Known trade-offs

- Structured tool-calling text (conversational-agent/SWE) shows the highest
  quantization divergence — typical for W4 on low-entropy syntax
- Long-tail KL is elevated (p99.9 ~25); occasional token-level wobble in
  rarely-seen contexts
