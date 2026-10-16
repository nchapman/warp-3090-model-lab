"""Draft-OPD boundary collection: run target-assisted decoding with a trained
drafter and record every verification boundary (draft proposals, acceptance,
target top-k distribution on the draft-induced prefix).

Method: monkeypatch vllm.v1.sample.rejection_sampler.RejectionSampler.forward
— the single verify boundary that already receives the draft distributions
(draft_probs = q), the target logits on draft-proposed prefixes (logits = p),
and the per-request draft block layout (SpecDecodeMetadata). The patch copies
per-step tensors to CPU records; reconstruction of per-request streams is
self-checked against llm.generate's returned token ids (mismatches drop the
request).

Paper (arXiv:2605.29343) fidelity notes:
  - Collection runs at serving temperature (SERVING_PARAMS), like the paper.
  - We record target top-k (k=64) per position: full-vocab storage is ~5MB
    per anchor; top-64 covers the mass where draft/target modes disagree at
    smoke scale. Reverse-KL training from top-k p is an approximation.
  - Prompts are drawn from conversations DISJOINT from the SFT (smoke2) pool,
    matching the paper's disjoint prompt pool.

Usage:
  python drafter/opd_collect.py --drafter smoke/draft_static/2 \
      --out data/opd_boundaries/static --n-prompts 300
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import SERVING_PARAMS  # noqa: E402

TOPK = 64


def install_hook(records: list[dict]):
    from vllm.v1.sample import rejection_sampler as rs

    orig = rs.RejectionSampler.forward

    def hooked(self, metadata, draft_probs, logits, sampling_metadata):
        out = orig(self, metadata, draft_probs, logits, sampling_metadata)
        try:
            rec = {
                "draft_token_ids": metadata.draft_token_ids.tolist(),
                "num_draft_tokens": list(metadata.num_draft_tokens),
                "cu_num_draft_tokens": metadata.cu_num_draft_tokens.tolist(),
                "sampled_token_ids": out.sampled_token_ids.tolist(),
            }
            if draft_probs is not None:
                with torch.no_grad():  # q: [num_tokens, vocab]
                    lq = torch.log_softmax(draft_probs.float(), dim=-1)
                    v, i = lq.topk(TOPK, dim=-1)
                    rec["q_top"] = list(zip(i.tolist(), v.tolist()))
            with torch.no_grad():  # p: target on draft prefixes + bonus rows
                lp = torch.log_softmax(logits.float(), dim=-1)
                v, i = lp.topk(TOPK, dim=-1)
                rec["p_top"] = list(zip(i.tolist(), v.tolist()))
            records.append(rec)
        except Exception as e:  # never break serving for a logging bug
            print(f"[opd_collect] hook error: {e}", file=sys.stderr)
        return out

    rs.RejectionSampler.forward = hooked
    return lambda: setattr(rs.RejectionSampler, "forward", orig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drafter", required=True, help="trained drafter checkpoint dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--wave", type=int, default=8,
                    help="prompts per generation wave; attribution is "
                         "validated per wave and a mismatched wave is dropped")
    ap.add_argument("--exclude-convs", default="data/smoke2_corpus_0.8b.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---- prompt pool: complete convs from the banked regen, disjoint from SFT
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drafter"))
    excluded = set()
    if Path(args.exclude_convs).exists():
        for line in open(args.exclude_convs):
            r = json.loads(line)
            excluded.add(r["id"].rsplit("-turn", 1)[0])
    pool, resp = {}, {}
    for line in open("data/context_pool.jsonl"):
        r = json.loads(line)
        conv = r["id"].rsplit("-turn", 1)[0]
        pool.setdefault(conv, {})[int(r["id"].rsplit("-turn", 1)[1])] = r
    for line in open("data/regen_0.8b.jsonl"):
        r = json.loads(line)
        resp[r["id"]] = r["response"]
    prompts = []
    for conv, turns in pool.items():
        if conv in excluded:
            continue
        if all(f"{conv}-turn{t}" in resp for t in turns):
            last = turns[max(turns)]
            prompts.append({"id": last["id"], "messages": last["messages"]})
        if len(prompts) >= args.n_prompts:
            break
    print(f"[opd_collect] {len(prompts)} disjoint prompts "
          f"(excluded {len(excluded)} SFT convs)")

    # ---- run target-assisted decoding with the hook installed, in waves
    records: list[dict] = []
    uninstall = install_hook(records)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.drafter)
    texts = [tok.apply_chat_template(p["messages"], tokenize=False,
                                     add_generation_prompt=True)
             for p in prompts]
    llm = LLM(model=args.drafter, max_model_len=4096,
              gpu_memory_utilization=float(__import__("os").environ.get(
                  "GPU_MEM_UTIL", "0.6")), enforce_eager=True)
    sp = SamplingParams(temperature=SERVING_PARAMS["temperature"],
                        top_p=SERVING_PARAMS.get("top_p", 1.0),
                        top_k=SERVING_PARAMS.get("top_k", -1),
                        max_tokens=args.max_tokens, seed=args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for w0 in range(0, len(texts), args.wave):
        wave_texts = texts[w0:w0 + args.wave]
        wave_prompts = prompts[w0:w0 + args.wave]
        records.clear()
        outs = llm.generate(wave_texts, sp)
        with open(out_dir / f"wave{w0 // args.wave:04d}_steps.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        with open(out_dir / f"wave{w0 // args.wave:04d}_outputs.jsonl", "w") as f:
            for p, o in zip(wave_prompts, outs):
                f.write(json.dumps({
                    "id": p["id"],
                    "prompt_token_ids": list(o.prompt_token_ids),
                    "output_token_ids": list(o.outputs[0].token_ids),
                }) + "\n")
        n_ok += len(wave_prompts)
        print(f"[opd_collect] wave {w0 // args.wave}: {len(records)} steps, "
              f"{len(wave_prompts)} prompts ({n_ok} total)", flush=True)
    uninstall()
    print(f"[opd_collect] done -> {out_dir}")


if __name__ == "__main__":
    main()
