"""One-shot probe: which sampler path does dflash decode actually take?"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # hook must see the engine

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # opd_collect lives here
counts = defaultdict(int)


def instrument():
    from vllm.v1.worker import gpu_model_runner as gmr
    from vllm.v1.sample import rejection_sampler as rs

    orig_sample = gmr.GPUModelRunner._sample

    def sample(self, logits, spec_decode_metadata):
        counts["_sample"] += 1
        counts[f"spec_meta={spec_decode_metadata is not None}"] += 1
        return orig_sample(self, logits, spec_decode_metadata)

    gmr.GPUModelRunner._sample = sample

    orig_fwd = rs.RejectionSampler.forward

    def fwd(self, *a, **k):
        counts["rejection_fwd"] += 1
        return orig_fwd(self, *a, **k)

    rs.RejectionSampler.forward = fwd
    print("PROBE instrumented", flush=True)


instrument()

from opd_collect import install_hook  # noqa: E402

records = []
install_hook(records, [0])

from models import resolve  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(resolve("0.8b")[1]["quant_repo"])
prompts = [json.loads(l) for l in open("data/smoke2_corpus_0.8b.jsonl")][:1]
text = tok.apply_chat_template(prompts[0]["messages"], tokenize=False,
                               add_generation_prompt=True)

from vllm import LLM, SamplingParams  # noqa: E402

llm = LLM(model="smoke/draft_static/checkpoint_best", max_model_len=4096,
          gpu_memory_utilization=0.6, enforce_eager=True)
out = llm.generate([text], SamplingParams(temperature=0.7, top_p=0.95,
                                          top_k=20, max_tokens=64, seed=42))
print("PROBE counts:", dict(counts), flush=True)
print("PROBE hook records:", len(records), flush=True)
print("PROBE tokens out:", len(out[0].outputs[0].token_ids), flush=True)
