"""Build a torch-format arrow dataset from a conversations jsonl.

Same conventions as the champion (v4/v5) datasets: chat-template tokenization,
integer-coerced ids, loss_mask=1 on assistant-response tokens (all-zero would
zero the training loss entirely — caught in rehearsal, see docs/history).

Usage:
  python drafter/prep_dataset.py --jsonl merged.jsonl --out ds_v4 \
      --model 4b            # tokenizer from the published quant
  python drafter/prep_dataset.py --jsonl merged.jsonl --out ds --quant ./Qwen3.5-4B-W4A16-GPTQ-cascade
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import resolve  # noqa: E402


def tokenize_with_masks(tok, messages, loss_mode="all"):
    """Chat-template ids with loss_mask=1 on assistant-response tokens.

    Incremental prefix tokenization: ids added by message i are trainable
    iff message i is an assistant turn. loss_mode="last" keeps only the
    final assistant turn trainable (de-duplicated supervision; earlier
    turns become pure conditioning).
    """
    trainable_roles = {"assistant"}
    if loss_mode == "last":
        last_asst = max(
            (i for i, m in enumerate(messages) if m["role"] == "assistant"),
            default=None,
        )
        # loop below additionally forces non-final assistant turns untrainable
    prev_ids: list[int] = []
    ids: list[int] = []
    mask: list[int] = []
    for i, m in enumerate(messages):
        prev_text = tok.apply_chat_template(messages[:i], tokenize=False) \
            if i else ""
        full_text = tok.apply_chat_template(messages[: i + 1], tokenize=False)
        prev = tok(prev_text, add_special_tokens=False)["input_ids"] if i else []
        full = tok(full_text, add_special_tokens=False)["input_ids"]
        new = full[len(prev):] if len(full) > len(prev) else []
        if new and isinstance(new[0], list):  # batched guard
            new = new[0]
        if loss_mode == "last" and m["role"] == "assistant" and i != last_asst:
            trainable = False
        else:
            trainable = m["role"] in trainable_roles
        ids.extend(int(x) for x in new)
        mask.extend([1 if trainable else 0] * len(new))
    return ids, mask


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True,
                    help="dataset dir (arrow); hidden_states will be written "
                         "here by extract_states.py")
    ap.add_argument("--model", default="4b",
                    help="model key — tokenizer comes from the published quant")
    ap.add_argument("--quant", default=None,
                    help="override: explicit quant dir/hub id for the tokenizer")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--loss-mode", choices=["all", "last"], default="all",
                    help="all: every assistant token trainable (champion "
                         "shape). last: only the final assistant turn "
                         "trainable — earlier turns are pure conditioning "
                         "(de-duplicated supervision)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="drop rows longer than this many tokens (0 = no "
                         "cap). Paper spec: 3072 (4096 for Coder); DSpark "
                         "reference prep: 1024. Drop matches DSpark's "
                         "prompt filter; truncation would cut responses "
                         "mid-turn.")
    args = ap.parse_args()

    _, info = resolve(args.model)
    quant = args.quant or info.get("quant_repo") or "Qwen/Qwen3.5-4B"
    tok = AutoTokenizer.from_pretrained(quant)

    recs, n = [], 0
    with open(args.jsonl) as f:
        for line in f:
            r = json.loads(line)
            ids, mask = tokenize_with_masks(tok, r["messages"], args.loss_mode)
            if len(ids) >= 16 and sum(mask) > 0:
                recs.append({"input_ids": ids, "loss_mask": mask, "seq_len": len(ids)})
            n += 1
            if args.limit and n >= args.limit:
                break
    ds = Dataset.from_list(recs)
    ds.set_format(type="torch", columns=["input_ids", "loss_mask", "seq_len"])
    ds.save_to_disk(args.out)
    print(f"prep_dataset: {len(recs)} rows ({n} read) -> {args.out}")


if __name__ == "__main__":
    main()
