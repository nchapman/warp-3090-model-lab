"""Fidelity probe: compare a quantized model against its BF16 base.

  - Perplexity (NLL) per model, delta % (aggregate + per domain)
  - Token-weighted mean KL(base || quant) and p99.9 KL (tail)
  - Blend-weighted aggregate (AGG_WEIGHTS) — the headline "weightedKL"

Single pass: base and quant both resident; one forward per model per batch.
Rows carry an optional "domain" tag (the frozen probe sets do).

Usage:
  python quants/probe.py --model 4b --quant ./Qwen3.5-4B-W4A16-GPTQ-cascade \
      --eval_set data/probe-v4.jsonl --tag repro
  python quants/probe.py --model 4b --quant ... --eval_set data/probe-guard.jsonl \
      --dump_rows rows.jsonl        # per-row records for paired bootstrap

Expected numbers: see RESULTS.md (weightedKL 0.1087 @ 4b on probe-v4).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import model_class, resolve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

from transformers import AutoProcessor  # noqa: E402

MAX_LEN = 4096
KL_ROWS_CAP = 400  # compute KL on all rows up to this many
CHUNK = 512  # low-memory: fp32 math on sequence slices only (248K-vocab logits)

# Analytic aggregate weights: Cascade-2 dataset blend (renormalized to 90%)
# + 10% multilingual as a deployment statement. Sampling counts in the probe
# are for statistical power ONLY; the aggregate's meaning lives here.
AGG_WEIGHTS = {
    "chat": 0.586 * 0.9, "math": 0.185 * 0.9, "science": 0.114 * 0.9,
    "instruction_following": 0.052 * 0.9, "conversational_agent": 0.035 * 0.9,
    "terminal_agent": 0.020 * 0.9, "swe": 0.006 * 0.9, "ML": 0.10,
}


def load_model(path):
    return model_class(path).from_pretrained(path, dtype="auto", device_map={"": 0})


def load_batches(processor, path, limit=None):
    batches = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            msgs = row.get("messages") or row.get("conversations")
            domain = row.get("domain", "all")
            text = processor.apply_chat_template(msgs, tokenize=False)
            enc = processor.tokenizer(
                text,
                max_length=MAX_LEN,
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            batches.append((domain, enc["input_ids"]))
            if limit and len(batches) >= limit:
                break
    return batches


@torch.no_grad()
def run_pair(base, quant, batches):
    nll = {"base": 0.0, "quant": 0.0}
    count = 0
    tok_kls = []
    row_dump = []
    dom = defaultdict(lambda: {"base": 0.0, "quant": 0.0, "count": 0, "kl": 0.0, "kl_n": 0})

    for i, (domain, ids) in enumerate(batches):
        ids = ids.to(base.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lb = base(input_ids=ids).logits[:, :-1]
            lq = quant(input_ids=ids).logits[:, :-1]
        targets = ids[:, 1:]
        d = dom[domain]
        row_nll = {"base": 0.0, "quant": 0.0}
        row_kl_sum, row_kl_n = 0.0, 0
        for s in range(0, lb.shape[1], CHUNK):
            sl = slice(s, min(s + CHUNK, lb.shape[1]))
            lbs = F.log_softmax(lb[:, sl].float(), dim=-1)
            lqs = F.log_softmax(lq[:, sl].float(), dim=-1)
            tgt_sl = targets[:, sl]
            for name, ls in (("base", lbs), ("quant", lqs)):
                loss = -ls.gather(-1, tgt_sl.unsqueeze(-1)).sum().item()
                nll[name] += loss
                d[name] += loss
                row_nll[name] += loss
            kl = F.kl_div(lqs, lbs, log_target=True, reduction="none").sum(-1).flatten()
            n_tok = kl.numel()
            count += n_tok
            d["count"] += n_tok
            tok_kls.append(kl.cpu())
            d["kl"] += kl.sum().item()
            d["kl_n"] += n_tok
            row_kl_sum += kl.sum().item()
            row_kl_n += n_tok
            del lbs, lqs, kl
        del lb, lq
        row_dump.append({"row": i, "domain": domain, "tokens": row_kl_n,
                         "mean_kl": row_kl_sum / max(row_kl_n, 1),
                         "base_nll": row_nll["base"], "quant_nll": row_nll["quant"]})

    return nll, count, torch.cat(tok_kls), dom, row_dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="4b",
                        help="model key from models.py — resolves the BF16 base")
    parser.add_argument("--base", default=None,
                        help="override: explicit base model path/hub id")
    parser.add_argument("--quant", required=True)
    parser.add_argument("--eval_set", default="data/probe-v4.jsonl")
    parser.add_argument("--dump_rows", default=None, help="write per-row KL/NLL records")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    from lib.lablog import log_event
    log_event("RUN_START", model=args.model, quant=args.quant, eval_set=args.eval_set)

    hub_id, _ = resolve(args.model)
    base_path = args.base or hub_id
    eval_set = Path(args.eval_set)
    if not eval_set.is_absolute():
        eval_set = ROOT / eval_set

    processor = AutoProcessor.from_pretrained(base_path)
    batches = load_batches(processor, eval_set)
    if len(batches) > KL_ROWS_CAP:
        batches = batches[:KL_ROWS_CAP]

    base = load_model(base_path)
    quant = load_model(args.quant)

    nll, count, tok_kls, dom, row_dump = run_pair(base, quant, batches)

    if args.dump_rows:
        with open(args.dump_rows, "w") as f:
            for r in row_dump:
                f.write(json.dumps(r) + "\n")

    base_ppl = nll["base"] / count
    quant_ppl = nll["quant"] / count
    ppl_delta = 100 * (quant_ppl - base_ppl) / base_ppl
    mean_kl = tok_kls.mean().item()
    p999_kl = torch.quantile(tok_kls, 0.999).item()

    print(f"[base{args.tag}] nll={base_ppl:.4f}")
    print(f"[quant{args.tag}] nll={quant_ppl:.4f} ({ppl_delta:+.2f}%)")
    print(f"[kl{args.tag}] mean KL(base||quant) = {mean_kl:.4f}  p99.9 = {p999_kl:.4f}")

    domains = {}
    for d, v in sorted(dom.items()):
        bn, qn = v["base"] / v["count"], v["quant"] / v["count"]
        domains[d] = {
            "tokens": v["count"],
            "dppl_pct": round(100 * (qn - bn) / bn, 2),
            "mean_kl": round(v["kl"] / max(v["kl_n"], 1), 4),
        }

    # analytic weighted aggregate (blend weights, not row shares)
    def dom_key(d):
        return "ML" if d.startswith("belebele") else d

    agg = defaultdict(lambda: [0.0, 0.0, 0])  # wsum, kl weighted, n
    for d, v in dom.items():
        k = dom_key(d)
        agg[k][0] += AGG_WEIGHTS.get(k, 0.0)
        agg[k][1] += AGG_WEIGHTS.get(k, 0.0) * (v["kl"] / max(v["kl_n"], 1))
        agg[k][2] += 1
    used = {k: v for k, v in agg.items() if v[0] > 0}
    weighted_mean_kl = sum(v[1] for v in used.values()) / sum(v[0] for v in used.values())

    print(
        json.dumps(
            {
                "base": base_path,
                "quant": args.quant,
                "eval_set": str(eval_set),
                "tag": args.tag,
                "rows": len(batches),
                "base_nll": round(base_ppl, 4),
                "quant_nll": round(quant_ppl, 4),
                "ppl_delta_pct": round(ppl_delta, 2),
                "mean_kl": round(mean_kl, 4),
                "weighted_mean_kl": round(weighted_mean_kl, 4),
                "p999_kl": round(p999_kl, 4),
                "domains": domains,
            }
        )
    )
    from lib.lablog import run_done
    run_done(mean_kl=round(mean_kl, 4), weighted_mean_kl=round(weighted_mean_kl, 4),
             p999_kl=round(p999_kl, 4), dppl_pct=round(ppl_delta, 2))


if __name__ == "__main__":
    main()
