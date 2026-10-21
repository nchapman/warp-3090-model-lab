#!/usr/bin/env python
"""Split the merged regen corpus into SFT train + disjoint OPD holdout.

Splits by WHOLE CONVERSATION (rows sharing a v4-<n> conv id never straddle
the split) so the OPD pool is context-disjoint from SFT training data.

Usage:
  python drafter/split_holdout.py --in data/regen_4b_final.jsonl \
      --holdout-convs 10000 --seed 42 \
      --out-sft data/regen_4b_sft.jsonl --out-holdout data/regen_4b_opd.jsonl

Writes a manifest line (counts + sha256 of both outputs) to stdout for
RESULTS.md / run-log provenance.
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict

CONV_RE = re.compile(r"^(v4-\d+)-turn\d+$")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--holdout-convs", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-sft", required=True)
    ap.add_argument("--out-holdout", required=True)
    args = ap.parse_args()

    by_conv = defaultdict(list)
    n_rows = 0
    with open(args.inp) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            m = CONV_RE.match(r.get("id", ""))
            conv = m.group(1) if m else f"row{i}"  # unlabeled rows: 1:1 "convs"
            by_conv[conv].append(line)
            n_rows += 1

    convs = sorted(by_conv)  # sorted before shuffle: deterministic across runs
    rng = random.Random(args.seed)
    rng.shuffle(convs)
    holdout = set(convs[: args.holdout_convs])

    n_sft = n_ho = 0
    with open(args.out_sft, "w") as sft, open(args.out_holdout, "w") as ho:
        for conv, lines in by_conv.items():
            out = ho if conv in holdout else sft
            for line in lines:
                out.write(line)
                n_ho += conv in holdout
                n_sft += conv not in holdout

    manifest = {
        "in": args.inp,
        "rows_total": n_rows,
        "convs_total": len(convs),
        "holdout_convs": len(holdout),
        "sft": {"path": args.out_sft, "rows": n_sft,
                "sha256": sha256_file(args.out_sft)},
        "holdout": {"path": args.out_holdout, "rows": n_ho,
                    "sha256": sha256_file(args.out_holdout)},
        "seed": args.seed,
    }
    print(json.dumps(manifest))


if __name__ == "__main__":
    sys.exit(main())
