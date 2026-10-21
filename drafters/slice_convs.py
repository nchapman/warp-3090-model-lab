"""Slice DFlash2 training conversations from the Nemotron-Cascade-2 pool.

Presets (validated recipes; see RESULTS.md / experiments/drafters/):
  small         ~2k convs — the on-ramp (24GB Blackwell afternoon, τ≈2.2 at 2.7M tok)
  v3            11k convs — the 29M-token rung
  v4            100k convs — the 260M-token champion rung
  v5-targeted   math/swe/terminal/science tilt (measured NULL on the ladder —
                kept for reproduction of the v5 experiment only)

Common rules: <=3 assistant-turn slices per conversation, <=7k rendered tokens
(checked with the quant's own chat template via transformers 5.x), conversation-
disjoint val split, conv-id manifest. Hash-disjoint (SHA1 of first user
message) from the frozen sets in data/ plus anything in --exclude.

Raw pool is NOT in git: data-raw/cascade/{subset}_*.parquet (see
quants/build_calib.py docstring).

Usage:
  python drafters/slice_convs.py --preset small --quant ./Qwen3.5-4B-W4A16-GPTQ-cascade
  python drafters/slice_convs.py --preset v4 --slices-dir slices   # excludes v3 if present
"""

import argparse
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

# domain blend from the Cascade-2 card (matches quants/build_calib.py)
BLEND = {"chat": 0.60, "math": 0.19, "science": 0.12, "instruction_following": 0.05,
         "conversational_agent": 0.03, "terminal_agent": 0.02}

PRESETS = {
    "small": dict(seed=3, convs=2_000, val=100, tag="s", targets=None),
    "v3": dict(seed=7, convs=11_000, val=500, tag="v3", targets=None),
    "v4": dict(seed=11, convs=100_000, val=5_000, tag="v4", targets=None),
    # v5 targeted blend (build_dflash_v5.py): message-aligned prefix truncation.
    # Measured NULL on the ladder (see RESULTS.md) — reproduction only.
    "v5-targeted": dict(seed=20261012, convs=None, val=0, tag="v5",
                        targets={"math": 6000, "swe": 6331, "terminal_agent": 6000,
                                 "science": 5000}),
}

MAX_TURNS_PER_CONV = 3
MAX_TOK = 7000
CHAR_PREFILTER = 24000   # ~4 chars/token -> safely under 7k before exact check
V5_MAX_CHARS = 28000     # ~7k tokens; math allowed longer
V5_MATH_MAX_CHARS = 40000


def first_user(messages):
    for m in messages:
        if m.get("role") == "user":
            return m["content"]
    return None


def row_hash(msgs):
    fu = first_user(msgs)
    return hashlib.sha1((fu or "").encode()).hexdigest()


def banned_hashes(extra_excludes, slices_dir):
    """Frozen sets in data/ + prior slices in slices_dir/ + user excludes."""
    banned = set()
    patterns = [str(ROOT / "data" / "*.jsonl")]
    if slices_dir:
        patterns += [f"{slices_dir}/*train*.jsonl", f"{slices_dir}/*val*.jsonl"]
    patterns += list(extra_excludes)
    for pattern in patterns:
        for p in glob.glob(pattern):
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    msgs = r.get("messages") or r.get("conversations")
                    if msgs:
                        banned.add(row_hash(msgs))
    return banned


def slice_pool(pool_dir, subset, need, rng, banned, max_chars=120_000, min_chars=200):
    files = sorted(glob.glob(f"{pool_dir}/{subset}_*.parquet"))
    if not files:
        sys.exit(f"no parquet shards for {subset!r} under {pool_dir}")
    pfs = [pq.ParquetFile(f) for f in files]
    order = [(fi, rg) for fi, pf in enumerate(pfs)
             for rg in range(pf.metadata.num_row_groups)]
    rng.shuffle(order)
    got = 0
    for fi, rg in order:
        if got >= need:
            break
        t = pfs[fi].read_row_group(rg, columns=["messages"])
        for msgs in t["messages"].to_pylist():
            if got >= need:
                break
            msgs = [{"role": m["role"], "content": m.get("content") or ""} for m in msgs]
            roles = [m["role"] for m in msgs]
            if "user" not in roles or "assistant" not in roles:
                continue
            total = sum(len(m["content"]) for m in msgs)
            if not (min_chars <= total <= max_chars):
                continue
            h = row_hash(msgs)
            if h in banned:
                continue
            banned.add(h)
            yield msgs
            got += 1
    return got


def truncate_messages(msgs, max_chars):
    """Keep whole messages from the head until the char budget is spent
    (v5 targeted-slice convention: message-aligned prefix truncation)."""
    out, total = [], 0
    for m in msgs:
        c = len(str(m.get("content", "")))
        if total + c > max_chars and out:
            break
        out.append(m)
        total += c
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=list(PRESETS), required=True)
    ap.add_argument("--quant", required=True,
                    help="quant dir/hub id — provides the chat template + tokenizer")
    ap.add_argument("--pool", default="data-raw/cascade")
    ap.add_argument("--slices-dir", default="slices",
                    help="prior slice jsonls here are excluded (rung freshness)")
    ap.add_argument("--out-dir", default="slices")
    ap.add_argument("--exclude", nargs="*", default=[])
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    rng = random.Random(cfg["seed"])
    pool = Path(args.pool)
    pool_dir = str(pool if pool.is_absolute() else ROOT / pool)
    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.quant)

    def render_len_ok(ctx):
        s = tok.apply_chat_template(ctx, add_generation_prompt=True, tokenize=False)
        return len(tok(s, add_special_tokens=False)["input_ids"]) <= MAX_TOK

    banned = banned_hashes(args.exclude, args.slices_dir)
    print(f"exclusion set: {len(banned)} conversation hashes", flush=True)

    targets = cfg["targets"] or {k: int(cfg["convs"] * v) for k, v in BLEND.items()}
    convs = []
    for subset, need in targets.items():
        max_chars = V5_MATH_MAX_CHARS if (args.preset == "v5-targeted"
                                          and subset == "math") else \
            (V5_MAX_CHARS if args.preset == "v5-targeted" else 120_000)
        got = 0
        for msgs in slice_pool(pool_dir, subset, need, rng, banned, max_chars):
            if args.preset == "v5-targeted":
                msgs = truncate_messages(msgs, V5_MATH_MAX_CHARS if subset == "math"
                                         else V5_MAX_CHARS)
            convs.append((subset, msgs))
            got += 1
        print(f"{subset}: {got}/{need} conversations", flush=True)

    rng.shuffle(convs)
    val_set = set()
    if cfg["val"]:
        val_set = set(random.Random(cfg["seed"] + 1).sample(
            range(len(convs)), min(cfg["val"], len(convs))))

    n_rows = n_drop = 0
    train_p = out_dir / f"{cfg['tag']}_train.jsonl"
    val_p = out_dir / f"{cfg['tag']}_val.jsonl"
    ids_p = out_dir / f"conv_ids_{cfg['tag']}.json"
    train_ids, val_ids = [], []
    with open(train_p, "w") as ftr, open(val_p, "w") as fva:
        for i, (subset, msgs) in enumerate(convs):
            cuts = [k for k, m in enumerate(msgs)
                    if m["role"] == "assistant"][:MAX_TURNS_PER_CONV]
            for k in cuts:
                ctx = msgs[:k]
                if not ctx or sum(len(m["content"]) for m in ctx) > CHAR_PREFILTER:
                    n_drop += 1
                    continue
                if not render_len_ok(ctx):
                    n_drop += 1
                    continue
                row = {"id": f"{cfg['tag']}-{i}-turn{k}", "messages": ctx,
                       "subset": subset}
                if i in val_set:
                    fva.write(json.dumps(row) + "\n")
                    val_ids.append(i)
                else:
                    ftr.write(json.dumps(row) + "\n")
                    train_ids.append(i)
                n_rows += 1
    json.dump({"train": sorted(set(train_ids)), "val": sorted(set(val_ids))},
              open(ids_p, "w"))
    print(f"rows kept {n_rows} (train {len(set(train_ids))} convs, "
          f"val {len(set(val_ids))} convs) | dropped {n_drop}")
    print(f"-> {train_p}\n   {val_p}\n   {ids_p}")


if __name__ == "__main__":
    main()
