"""Rollout-agreement probe: 32-token greedy trajectories, base vs quant.

Teacher-forced KL (probe.py) scores the quant's predictions along the BASE's
continuation — the quant's own drift never feeds back into later tokens.
This probe measures the compounding regime: both models greedily decode the
SAME prompt (chat template + generation prompt, 4096-token cap) for up to
--new-tokens steps, and the metric is how far the quant's rollout tracks the
base's. Looping, tool-format drift and agentic breakdown live here (the
regime Unsloth's Divergence-300 measures; our conv_agent KL wall predicts it
concentrates there).

Per row (deterministic — greedy, identical batches both sides, frozen set):
  rollout      generated ids up to and including the first EOS (early stop
               counts as a token: stopping where the base stops is agreement)
  prefix_len   length of the common token prefix of the two rollouts
  exact        full-rollout match (both trajectories identical)
  agree@k      quant's first k tokens == base's first k tokens; for k past
               the base's EOS the row only counts if the quant stopped at
               the same step (exact). agree@1 is the top-1 analog, exact@N
               the Unsloth-style headline.

Headlines reported overall and per domain; --dump_rows writes per-row
records for paired candidate comparisons later.

Usage:
  python quant/traj_probe.py --model 4b --quant <dir-or-hub-id> --tag champ \
      [--eval_set data/traj-v1.jsonl] [--new-tokens 32] [--batch-size 8] \
      [--dump_rows rows.jsonl] [--limit 16]
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import model_class, resolve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAX_LEN = 4096
AGREE_KS = [1, 4, 8, 16, 32]


def load_model(path):
    return model_class(path).from_pretrained(path, dtype="auto", device_map={"": 0})


def prompt_ids(tok, msgs):
    """Tokenized generation prompt, capped at MAX_LEN without ever slicing
    the assistant-header suffix off the end (long rows lose body tail,
    keep the generation header — both models see identical prompts)."""
    msgs = list(msgs)
    if msgs and msgs[-1]["role"] == "assistant":
        msgs = msgs[:-1]  # generate THE final response, not a next one
    body = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
    full = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if not full.startswith(body):
        raise RuntimeError("chat template is not body-prefix shaped; "
                           "cannot cap length safely")
    suffix_ids = tok(full[len(body):], add_special_tokens=False)["input_ids"]
    body_ids = tok(body, add_special_tokens=False)["input_ids"]
    return body_ids[: MAX_LEN - len(suffix_ids)] + suffix_ids


@torch.no_grad()
def rollouts(model, tok, prompts, new_tokens, bs):
    """Greedy rollouts for a list of prompt id-lists. Returns list of
    token-id lists (generated region only, EOS inclusive)."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else None
    eos_cfg = model.generation_config.eos_token_id
    eos_set = {eos_cfg} if isinstance(eos_cfg, int) else set(eos_cfg or [])
    if not eos_set and tok.eos_token_id is not None:
        print(f"warning: generation_config has no eos_token_id; "
              f"falling back to tokenizer eos {tok.eos_token_id}", flush=True)
        eos_set = {tok.eos_token_id}
    pad_id = pad_id if pad_id is not None else next(iter(eos_set), None)

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    for s in range(0, len(order), bs):
        chunk = order[s: s + bs]
        ids = torch.full((len(chunk), max(len(prompts[i]) for i in chunk)),
                         pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for r, i in enumerate(chunk):
            n = len(prompts[i])
            ids[r, -n:] = torch.tensor(prompts[i])  # left pad
            mask[r, -n:] = 1
        ids, mask = ids.to(model.device), mask.to(model.device)
        seq = model.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=new_tokens,
            do_sample=False, num_beams=1, pad_token_id=pad_id,
        )[:, ids.shape[1]:]
        for r, i in enumerate(chunk):
            gen = seq[r].tolist()
            stop = next((j for j, t in enumerate(gen) if t in eos_set), None)
            out[i] = gen[: stop + 1] if stop is not None else gen
    return out


def row_metrics(rb, rq, new_tokens):
    prefix = 0
    for a, b in zip(rb, rq):
        if a != b:
            break
        prefix += 1
    exact = rb == rq
    agree = {k: (prefix >= min(k, len(rb)) and (k <= len(rb) or exact))
             for k in range(1, new_tokens + 1)}
    return prefix, exact, agree


def aggregate(rows, new_tokens):
    n = len(rows)
    if not n:
        return {}
    ks = [k for k in (1, 8, 16) if k <= new_tokens]
    prefs = sorted(r["prefix"] for r in rows)
    p10 = prefs[max(0, math.floor(0.10 * (n - 1)))]
    p50 = prefs[len(prefs) // 2]
    out = {"rows": n}
    for k in ks:
        out[f"agree@{k}"] = round(sum(r["agree"][k] for r in rows) / n, 4)
    out.update({
        "exact@N": round(sum(r["exact"] for r in rows) / n, 4),
        "prefix_mean": round(sum(prefs) / n, 2),
        "prefix_p50": p50,
        "prefix_p10": p10,
    })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="4b",
                        help="model key from models.py — resolves the BF16 base")
    parser.add_argument("--base", default=None,
                        help="override: explicit base model path/hub id")
    parser.add_argument("--quant", required=True)
    parser.add_argument("--eval_set", default="data/traj-v1.jsonl")
    parser.add_argument("--new_tokens", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dump_rows", default=None,
                        help="write per-row records (prefix, exact, texts)")
    parser.add_argument("--limit", type=int, default=None,
                        help="first N rows only (smoke tests)")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    from lablog import log_event
    log_event("RUN_START", model=args.model, quant=args.quant,
              eval_set=args.eval_set, new_tokens=args.new_tokens)

    hub_id, _ = resolve(args.model)
    base_path = args.base or hub_id
    eval_set = Path(args.eval_set)
    if not eval_set.is_absolute():
        eval_set = ROOT / eval_set

    prompts, domains = [], []
    with open(eval_set) as f:
        for line in f:
            row = json.loads(line)
            msgs = row.get("messages") or row.get("conversations")
            prompts.append(msgs)
            domains.append(row.get("domain", "all"))
            if args.limit is not None and len(prompts) >= args.limit:
                break
    if not prompts:
        sys.exit(f"eval set {eval_set} yielded no rows")

    from transformers import AutoProcessor
    tok = AutoProcessor.from_pretrained(base_path).tokenizer
    ids = [prompt_ids(tok, m) for m in prompts]
    print(f"prompts: {len(ids)} rows, "
          f"{min(map(len, ids))}-{max(map(len, ids))} tokens", flush=True)

    base = load_model(base_path)
    quant = load_model(args.quant)

    t0 = time.time()
    rb = rollouts(base, tok, ids, args.new_tokens, args.batch_size)
    t1 = time.time()
    rq = rollouts(quant, tok, ids, args.new_tokens, args.batch_size)
    t2 = time.time()
    print(f"rollouts: base {t1 - t0:.0f}s, quant {t2 - t1:.0f}s", flush=True)
    del base
    torch.cuda.empty_cache()

    recs = []
    for dom, b, q in zip(domains, rb, rq):
        prefix, exact, agree = row_metrics(b, q, args.new_tokens)
        recs.append({"domain": dom, "prefix": prefix, "exact": exact,
                     "agree": agree, "base": b, "quant": q})

    if args.dump_rows:
        with open(args.dump_rows, "w") as f:
            for r, b, q in zip(recs, rb, rq):
                f.write(json.dumps({
                    "domain": r["domain"], "prefix": r["prefix"],
                    "exact": r["exact"],
                    "agree@ks": {k: r["agree"][k]
                                for k in AGREE_KS if k <= args.new_tokens},
                    "base_text": tok.decode(b), "quant_text": tok.decode(q),
                }) + "\n")

    overall = aggregate(recs, args.new_tokens)
    by_dom = defaultdict(list)
    for r in recs:
        by_dom["ML" if r["domain"].startswith("belebele") else r["domain"]].append(r)
    domains_out = {d: aggregate(rs, args.new_tokens)
                   for d, rs in sorted(by_dom.items())}

    k_tab = min(16, args.new_tokens)
    print(f"{'domain':24s} {'n':>4s} {'agr@1':>6s} {f'agr@{k_tab}':>7s} "
          f"{'exact@N':>8s} {'pfx_mean':>8s} {'pfx_p10':>7s}")
    for d, v in [("OVERALL", overall)] + list(domains_out.items()):
        print(f"{d:24s} {v['rows']:4d} {v['agree@1']:6.3f} "
              f"{v[f'agree@{k_tab}']:7.3f} "
              f"{v['exact@N']:8.3f} {v['prefix_mean']:8.2f} {v['prefix_p10']:7d}")

    print(json.dumps({
        "base": base_path, "quant": args.quant, "tag": args.tag,
        "eval_set": str(eval_set), "new_tokens": args.new_tokens,
        **overall, "domains": domains_out,
    }))
    from lablog import run_done
    run_done(agree1=overall["agree@1"], exact=overall["exact@N"],
             prefix_mean=overall["prefix_mean"],
             prefix_p10=overall["prefix_p10"], rows=overall["rows"])


if __name__ == "__main__":
    main()
