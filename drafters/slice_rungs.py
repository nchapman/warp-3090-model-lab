"""Conversation-grouped nested rung slices for the data-scaling ladder.

Rungs are nested by construction: conversation groups are walked in sorted
conversation-id order and emitted until the token budget is reached, so a
smaller rung's conversations are a prefix-subset of a larger rung's (same
walk, bigger budget). Rows are never split across rungs — every row of a
conversation (id prefix v4-<i>) lands together, preserving the conv-disjoint
discipline the champion recipe was trained under.

Deterministic: same jsonl + same budget => identical slice (sorted groups,
same row filters as prep_dataset). Token counts use the quant's own chat
template — the same tokenizer path training sees.

Usage:
  python drafters/slice_rungs.py --jsonl regen_all.jsonl --tokens 10000000 \
      --out rung10m.jsonl --quant nchapman/Qwen3.5-4B-W4A16-GPTQ
"""

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_dataset import tokenize_with_masks  # noqa: E402


def conv_of(row_id):
    return row_id.rsplit("-turn", 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--tokens", type=int, required=True,
                    help="token budget; slice ends at the first conversation "
                         "boundary that reaches it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default="nchapman/Qwen3.5-4B-W4A16-GPTQ",
                    help="quant dir/hub id — tokenizer comes from the quant")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.quant)

    groups = {}
    with open(args.jsonl) as f:
        for line in f:
            r = json.loads(line)
            groups.setdefault(conv_of(r["id"]), []).append(r)

    total = kept_rows = kept_convs = 0
    with open(args.out, "w") as out:
        for conv in sorted(groups):
            if total >= args.tokens:
                break
            rows = sorted(groups[conv], key=lambda r: r["id"])
            blob, conv_tok = [], 0
            for r in rows:
                ids, mask = tokenize_with_masks(tok, r["messages"])
                if len(ids) >= 16 and sum(mask) > 0:  # prep_dataset's filters
                    blob.append(r)
                    conv_tok += len(ids)
            if not blob:
                continue
            kept_convs += 1
            kept_rows += len(blob)
            total += conv_tok
            for r in blob:
                out.write(json.dumps(r) + "\n")

    print(f"slice_rungs: {kept_convs} convs, {kept_rows} rows, "
          f"{total / 1e6:.2f}M tokens -> {args.out}")


if __name__ == "__main__":
    main()
