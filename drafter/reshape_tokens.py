"""Token-shape reshaper for the data-scaling testbed.

Our corpus rows are conversation-turn prefixes (id `v4-<conv>-turn<N>`): a
turn-4 row contains and re-supervises turns 1-3. This tool re-shapes a regen
jsonl into non-duplicated forms so data-budget arms compare like-for-like in
UNIQUE SUPERVISED TOKENS, not nominal row tokens.

Modes (--shape):
  prefix   passthrough (current champion shape) — every row as-is
  merged   one row per conversation: keep only the max-turn row per conv
           (every assistant token supervised exactly once, conditioning =
           full conversation prefix)
  short    merged + drop conversations whose tokenized length exceeds
           --max-len (reference-repo style: short samples only)

Row selection only — tokenization/loss masking stays in prep_dataset.py
(--loss-mode there controls which assistant turns are trainable).

Usage:
  python drafter/reshape_tokens.py --jsonl regen_all.jsonl \
      --out reshaped_merged.jsonl --shape merged
"""

import argparse
import json
import re
from collections import defaultdict

ID_RE = re.compile(r"^(v4-\d+)-turn(\d+)$")


def conv_key(row_id: str) -> tuple[str, int] | None:
    m = ID_RE.match(row_id)
    return (m.group(1), int(m.group(2))) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shape", choices=["prefix", "merged", "short"],
                    default="merged")
    ap.add_argument("--max-len", type=int, default=1024,
                    help="short mode: drop convs longer than this many "
                         "CHARACTERS x ~3.5 chars/token estimate is wrong on "
                         "purpose — you must pass the tokenized cap from the "
                         "caller that owns a tokenizer")
    args = ap.parse_args()

    best: dict[str, tuple[int, dict]] = {}
    n_rows = n_unmatched = 0
    with open(args.jsonl) as f:
        for line in f:
            r = json.loads(line)
            n_rows += 1
            key = conv_key(r.get("id", ""))
            if key is None:
                n_unmatched += 1
                key = (r.get("id", f"row{n_rows}"), 0)
            conv, turn = key
            if conv not in best or turn > best[conv][0]:
                best[conv] = (turn, r)

    kept = [r for _, r in best.values()]
    if args.shape == "short":
        # char-length prefilter only: prep_dataset applies the true token cap
        # (it owns the tokenizer); we just shed obvious long-tail convs early.
        kept = [r for r in kept
                if sum(len(m.get("content") or "") for m in r["messages"])
                <= args.max_len * 4]

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    print(f"shape={args.shape}: {n_rows} rows -> {len(kept)} rows "
          f"({len(best)} conversations, {n_unmatched} unmatched ids)")
    if args.shape == "prefix":
        print("NOTE: prefix mode does not dedup — budget accounting must "
              "count unique supervised tokens upstream of slicing.")


if __name__ == "__main__":
    main()
