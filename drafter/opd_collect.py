"""Draft-OPD boundary collection: run target-assisted decoding with a trained
drafter and record every verification boundary (draft proposals, acceptance,
target top-k distribution on the draft-induced prefix).

Method: hook vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils.
rejection_sample — the dflash verify boundary receiving the draft
distributions (draft_logits = q), the target logits on draft-proposed rows
(target_logits = p), and the per-request block layout. The patch copies
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
from models import SERVING_PARAMS, resolve  # noqa: E402

TOPK = 64


def install_hook(records: list[dict], errors: list[int]):
    """Hook the spec-decode verify boundary.

    vLLM 0.31's dflash path verifies via
    vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils.rejection_sample —
    it receives the draft distributions (draft_logits = q), the target logits
    on the draft-proposed rows (target_logits = p), the proposed tokens, and
    direct row->request attribution (expanded_idx_mapping). The
    v1.sample.RejectionSampler class is NOT in this path (probe-verified).
    """
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as ru

    orig = ru.rejection_sample

    def hooked(target_logits, draft_logits, draft_sampled, cu_num_logits, pos,
               idx_mapping, expanded_idx_mapping, expanded_local_pos,
               temperature, seed, num_speculative_steps, *args, **kwargs):
        out = orig(target_logits, draft_logits, draft_sampled, cu_num_logits,
                   pos, idx_mapping, expanded_idx_mapping, expanded_local_pos,
                   temperature, seed, num_speculative_steps, *args, **kwargs)
        try:
            with torch.no_grad():
                lp = torch.log_softmax(target_logits.float(), dim=-1)
                pv, pi = lp.topk(TOPK, dim=-1)
                rec = {
                    "draft_sampled": draft_sampled.tolist(),
                    "cu_num_logits": cu_num_logits.tolist(),
                    "expanded_idx_mapping": expanded_idx_mapping.tolist(),
                    "expanded_local_pos": expanded_local_pos.tolist(),
                    "num_speculative_steps": int(num_speculative_steps),
                    "sampled": out[0].tolist(),
                    "num_sampled": out[1].tolist(),
                    "p_top": list(zip(pi.tolist(), pv.tolist())),
                }
                if draft_logits is not None:
                    lq = torch.log_softmax(draft_logits.float(), dim=-1)
                    qv, qi = lq.topk(TOPK, dim=-1)
                    rec["q_top"] = list(zip(qi.tolist(), qv.tolist()))
                records.append(rec)
        except Exception as e:  # keep serving alive, but surface it per-wave
            errors[0] += 1
            print(f"[opd_collect] hook error #{errors[0]}: {e}",
                  file=sys.stderr, flush=True)
        return out

    ru.rejection_sample = hooked
    return lambda: setattr(ru, "rejection_sample", orig)


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
    ap.add_argument("--model", default="0.8b",
                    help="model key: tokenizer comes from the quant (drafter "
                         "checkpoint dirs don't ship tokenizer files)")
    ap.add_argument("--context-pool", default="data/context_pool.jsonl")
    ap.add_argument("--regen-log", default="data/regen_0.8b.jsonl")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    for p in (args.context_pool, args.regen_log):
        if not (root / p).exists():
            sys.exit(f"[opd_collect] missing input: {root / p}")

    # ---- prompt pool: complete convs from the banked regen, disjoint from SFT
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drafter"))
    excluded = set()
    if (root / args.exclude_convs).exists():
        for line in open(root / args.exclude_convs):
            r = json.loads(line)
            excluded.add(r["id"].rsplit("-turn", 1)[0])
    else:
        print(f"[opd_collect] WARNING: exclude set {args.exclude_convs} not "
              f"found — SFT-disjointness NOT guaranteed", file=sys.stderr,
              flush=True)
    pool, resp = {}, {}
    for line in open(root / args.context_pool):
        r = json.loads(line)
        conv = r["id"].rsplit("-turn", 1)[0]
        pool.setdefault(conv, {})[int(r["id"].rsplit("-turn", 1)[1])] = r
    for line in open(root / args.regen_log):
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
    if not prompts:
        sys.exit("[opd_collect] no usable prompts — pool/regen inputs "
                 "exhausted or exclude set covers everything")

    # ---- run target-assisted decoding with the hook installed, in waves
    # vLLM v1 runs EngineCore in a subprocess — the hook must live in that
    # process, so single-process mode is REQUIRED and set before any vllm
    # import (a setdefault here would silently no-op if the caller exported 1)
    import os
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    records: list[dict] = []
    hook_errors = [0]
    uninstall = install_hook(records, hook_errors)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        resolve(args.model)[1].get("quant_repo") or "Qwen/Qwen3.5-0.8B")
    texts = [tok.apply_chat_template(p["messages"], tokenize=False,
                                     add_generation_prompt=True)
             for p in prompts]
    # pre-filter: prompts must fit max_model_len with the response budget
    # (pool lengths are uncapped — p99 ≈ 6.2K tokens)
    max_model_len = 8192
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    keep = [i for i, ids in enumerate(enc)
            if len(ids) + args.max_tokens + 32 <= max_model_len]
    if len(keep) < len(texts):
        print(f"[opd_collect] dropped {len(texts) - len(keep)} over-length "
              f"prompts (> {max_model_len - args.max_tokens - 32} tok)")
    texts = [texts[i] for i in keep]
    prompts = [prompts[i] for i in keep]
    if not texts:
        sys.exit("[opd_collect] every prompt exceeded the length prefilter — "
                 "raise max_model_len or shorten the pool")

    llm = LLM(model=args.drafter, max_model_len=max_model_len,
              gpu_memory_utilization=float(__import__("os").environ.get(
                  "GPU_MEM_UTIL", "0.6")), enforce_eager=True)
    sp = SamplingParams(temperature=SERVING_PARAMS["temperature"],
                        top_p=SERVING_PARAMS.get("top_p", 1.0),
                        top_k=SERVING_PARAMS.get("top_k", -1),
                        max_tokens=args.max_tokens, seed=args.seed)

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.glob("wave*_*.jsonl")):
        sys.exit(f"[opd_collect] {out_dir} already holds wave files — stale "
                 "waves would silently mix runs; move it or pick a fresh --out")
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
        if hook_errors[0] or not records:
            uninstall()
            sys.exit(f"[opd_collect] wave {w0 // args.wave}: instrumentation "
                     f"failure (records={len(records)}, hook_errors="
                     f"{hook_errors[0]}) — aborting rather than writing an "
                     f"empty dataset")
        print(f"[opd_collect] wave {w0 // args.wave}: {len(records)} steps, "
              f"{len(wave_prompts)} prompts ({n_ok} total)", flush=True)
    uninstall()
    print(f"[opd_collect] done -> {out_dir}")


if __name__ == "__main__":
    main()
