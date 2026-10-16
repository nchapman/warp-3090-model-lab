"""Data-units audit: make "amount of data" precise for ladder corpora.

We have sloppily used four currencies — rows, conversations, nominal row
tokens, and literature "samples". This tool computes ALL of them plus the
two the trainer actually cares about (unique supervised tokens, anchor
block supply) so every rung comparison can be stated in one canonical
unit: UNIQUE SUPERVISED TOKENS (loss_mask=1 tokens counted once per
conversation), with conversations as the diversity co-unit.

Because rows are conversation-turn PREFIXES (turn-N row contains turns
1..N), per conversation:
  unique tokens        = max-row length (the longest prefix IS the conv)
  nominal tokens       = sum of all row lengths (prefix duplication here)
  supervised (all)     = assistant tokens in each row, summed over rows
                         (early turns re-supervised — the duplication bug)
  supervised (unique)  = assistant tokens in the max row, counted once

Anchor supply (per --loss-mode): 16-token blocks whose FIRST position is
supervised are anchor-eligible under the DFlash construction (anchor =
first clean position of a block). We report eligible positions as a
proxy for per-epoch anchor supply.

Usage (arrow ds preferred — ids zipped from the source jsonl by order):
  python drafter/data_audit.py --ds ../llm-compressor/ladder/ds_r10 \
      --jsonl ../llm-compressor/ladder/rung_r10.jsonl
  python drafter/data_audit.py --jsonl regen_all.jsonl --sample 20000
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ID_RE = re.compile(r"^(v4-\d+)-turn(\d+)$")


def rows_from_ds(ds_path, jsonl_path):
    ids = [json.loads(l).get("id", f"row{i}")
           for i, l in enumerate(open(jsonl_path))]
    try:
        from datasets import load_from_disk
        ds = load_from_disk(ds_path)
        return ids, list(ds["input_ids"]), list(ds["loss_mask"])
    except Exception:
        return ids, None, None


def rows_from_jsonl(jsonl_path, tok):
    ids, idsl, masks = [], [], []
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prep_dataset import tokenize_with_masks
    for i, line in enumerate(open(jsonl_path)):
        r = json.loads(line)
        ids.append(r.get("id", f"row{i}"))
        a, m = tokenize_with_masks(tok, r["messages"])
        idsl.append(a); masks.append(m)
    return ids, idsl, masks


def audit(ids, input_ids, loss_masks, label, block=16):
    convs = defaultdict(list)          # conv -> [(turn, len, sup, ids)]
    for rid, ids_r, mask_r in zip(ids, input_ids, loss_masks):
        m = ID_RE.match(rid)
        conv, turn = (m.group(1), int(m.group(2))) if m else (rid, 0)
        convs[conv].append((turn, len(ids_r), sum(mask_r)))

    n_rows = len(ids)
    nominal = sum(t[1] for v in convs.values() for t in v)
    unique = sum(max(t[1] for t in v) for v in convs.values())
    sup_all = sum(t[2] for v in convs.values() for t in v)
    sup_uniq = sum(max(t[2] for t in v) for v in convs.values())
    # anchor-eligible positions: supervised positions (first-clean proxy)
    anchors_uniq = sup_uniq  # 1 eligible anchor per unique supervised token

    print(f"== {label} ==")
    print(f"  rows                    {n_rows:>12,}")
    print(f"  conversations           {len(convs):>12,}")
    print(f"  nominal tokens          {nominal:>12,}   (slice budget counted this)")
    print(f"  unique tokens           {unique:>12,}   ({unique/nominal:.1%} of nominal — prefix duplication)")
    print(f"  supervised (as-trained) {sup_all:>12,}   (loss_mask=1, with re-supervision)")
    print(f"  supervised (unique)     {sup_uniq:>12,}   ({sup_uniq/unique:.1%} of unique — assistant share)")
    print(f"  anchor-eligible (proxy) {anchors_uniq:>12,}   (~{anchors_uniq//block:,} block-{block} blocks)")
    print(f"  nominal/supervised-unique exchange: {nominal/sup_uniq:.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ds", help="arrow dataset dir (exact, preferred)")
    ap.add_argument("--jsonl", required=True, help="source jsonl (ids; fallback tokenization)")
    ap.add_argument("--quant", default="nchapman/Qwen3.5-4B-W4A16-GPTQ")
    ap.add_argument("--sample", type=int, default=0, help="jsonl-mode: sample N rows (estimate)")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    if args.ds and Path(args.ds).exists():
        ids, input_ids, masks = rows_from_ds(args.ds, args.jsonl)
        assert input_ids is not None, "ds load failed"
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.quant)
        lines = open(args.jsonl).readlines()
        if args.sample and len(lines) > args.sample:
            import random
            lines = random.Random(7).sample(lines, args.sample)
        with open("/tmp/_audit_sample.jsonl", "w") as f:
            f.writelines(lines)
        ids, input_ids, masks = rows_from_jsonl("/tmp/_audit_sample.jsonl", tok)
    audit(ids, input_ids, masks, args.label or (args.ds or args.jsonl))


if __name__ == "__main__":
    main()
