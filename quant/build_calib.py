"""Build calibration sets and evaluation probes from raw data pools.

Modes:
  calib  — stratified slice of nvidia/Nemotron-Cascade-2-SFT-Data at the
           dataset's own domain proportions (the champion cascade-v1 recipe;
           the frozen set itself ships in data/calib-cascade-v1.jsonl — you
           only need this mode to build NEW variants).
  probe  — a fresh multi-domain ruler in the frozen probe-v4 style, with
           explicit --targets (v4 allocation: chat 90, math 80, IF 40,
           science 30, conv_agent 18, terminal 14, swe 16, ML 10x9).
  guard  — external OOD guard (HuggingFaceH4/no_robots + facebook/belebele,
           9 languages), never calibrated on.

Every mode asserts hash-disjointness (SHA1 of first user message) against the
frozen sets in data/ and anything you pass via --exclude.

Raw pools are NOT in git. Download once:
  Nemotron-Cascade-2-SFT-Data parquet shards -> data-raw/cascade/{subset}_*.parquet
  (build_guard_probe fetches no_robots / belebele itself on first use)

Usage:
  python quant/build_calib.py --mode calib --out slices/calib-vX.jsonl
  python quant/build_calib.py --mode probe --out slices/probe-vX.jsonl \
      --targets-json '{"chat": 90, "math": 80, ...}'
  python quant/build_calib.py --mode guard --out slices/guard-vX.jsonl
"""

import argparse
import glob
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SEED = 0
MIN_CHARS, MAX_CHARS = 200, 600_000
CASCADE_SOURCE = "nvidia/Nemotron-Cascade-2-SFT-Data"
OVERSAMPLE = 2.5

# dataset's own blend (card statistics, safety excluded, renormalized)
CARD_TOTALS = {
    "chat": 9_320_000,
    "math": 2_940_000,
    "science": 1_810_000,
    "instruction_following": 820_000,
    "conversational_agent": 548_000,
    "terminal_agent": 324_000,
    "swe": 102_000,
}
TOT = sum(CARD_TOTALS.values())

BELEBELE_LANGS = ["deu_Latn", "spa_Latn", "fra_Latn", "ita_Latn", "jpn_Jpan",
                  "zho_Hans", "arb_Arab", "hin_Deva", "rus_Cyrl"]

# frozen probe-v4 allocation (the shipped ruler; default for --mode probe)
V4_TARGETS = {"chat": 90, "math": 80, "instruction_following": 40, "science": 30,
              "conversational_agent": 18, "terminal_agent": 14, "swe": 16}
ML_PER_LANG = 10


def row_hash(msgs):
    first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
    return hashlib.sha1(first_user.encode()).hexdigest()


def banned_hashes(extra_excludes):
    """Hashes of every frozen set in data/ + user-provided excludes."""
    banned = set()
    for frozen in ["calib-cascade-v1.jsonl", "probe-v4.jsonl", "probe-guard.jsonl"]:
        p = DATA / frozen
        if p.exists():
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    msgs = r.get("messages") or r.get("conversations")
                    banned.add(row_hash(msgs))
    for pattern in extra_excludes:
        for p in glob.glob(str(Path(pattern).absolute() if Path(pattern).is_absolute()
                               else ROOT / pattern)):
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    msgs = r.get("messages") or r.get("conversations")
                    banned.add(row_hash(msgs))
    return banned


def valid(msgs):
    roles = [m["role"] for m in msgs]
    if "user" not in roles or "assistant" not in roles:
        return False
    total = sum(len(m.get("content") or "") for m in msgs)
    return MIN_CHARS <= total <= MAX_CHARS


def pool_for(pool_dir, subset, need, rng, banned):
    """Randomized row-group scan across the subset's parquet shards."""
    files = sorted(glob.glob(f"{pool_dir}/{subset}_*.parquet"))
    if not files:
        sys.exit(f"no parquet shards for {subset!r} under {pool_dir} — download the "
                 f"pool first (see module docstring)")
    pool = []
    pfs = [(pq.ParquetFile(f), f) for f in files]
    order = list(range(sum(pf.metadata.num_row_groups for pf, _ in pfs)))
    rng.shuffle(order)
    idx = 0
    for pf, f in pfs:
        while idx < len(order) and len(pool) < need:
            rg = order[idx]
            idx += 1
            if rg >= pf.metadata.num_row_groups:
                continue
            t = pf.read_row_group(rg, columns=["messages", "generator"])
            for msgs, gen in zip(t["messages"].to_pylist(), t["generator"].to_pylist()):
                msgs = [{"role": m["role"], "content": m.get("content") or ""} for m in msgs]
                if not valid(msgs):
                    continue
                h = row_hash(msgs)
                if h in banned:
                    continue
                banned.add(h)
                pool.append((msgs, gen or "unknown"))
    return pool


def stratified(pool, n, rng):
    """Sample n rows preserving the pool's generator proportions."""
    by_gen = defaultdict(list)
    for row in pool:
        by_gen[row[1]].append(row)
    for rows in by_gen.values():
        rng.shuffle(rows)
    alloc = {}
    remaining = n
    gens = sorted(by_gen, key=lambda g: -len(by_gen[g]))
    for i, g in enumerate(gens):
        want = round(n * len(by_gen[g]) / len(pool)) if i < len(gens) - 1 else remaining
        want = min(want, len(by_gen[g]), remaining)
        alloc[g] = want
        remaining -= want
    out = []
    for g, k in alloc.items():
        out += by_gen[g][:k]
    return out


def build_calib(args, rng, banned):
    weights = {k: v / TOT for k, v in CARD_TOTALS.items()}
    rows = []
    print(f"{'subset':24s} {'share':>6s} {'rows':>5s} pool")
    for subset, share in weights.items():
        n = round(args.n * share)
        pool = pool_for(args.pool, subset, int(n * OVERSAMPLE), rng, banned)
        rows += stratified(pool, min(n, len(pool)), rng)
        print(f"{subset:24s} {share:6.1%} {n:5d} {len(pool)}", flush=True)
    rng.shuffle(rows)
    return [{"conversations": msgs, "source": CASCADE_SOURCE,
             "domain": None, "generator": gen} for msgs, gen in rows]


def belebele_rows(lang, need, rng, banned, raw_dir):
    """ML probe rows from facebook/belebele (fetched on first use)."""
    dest = raw_dir / f"belebele_{lang}.parquet"
    if not dest.exists():
        urls = hf_parquet_urls("facebook/belebele", lang)
        curl(urls[0], dest)
    t = pq.read_table(dest, columns=["flores_passage", "question", "mc_answer1",
                                     "mc_answer2", "mc_answer3", "mc_answer4",
                                     "correct_answer_num"])
    idx = list(range(len(t)))
    rng.shuffle(idx)
    out, got = [], 0
    for i in idx:
        if got >= need:
            break
        r = {c: t[c][i].as_py() for c in t.column_names}
        opts = [r["mc_answer1"], r["mc_answer2"], r["mc_answer3"], r["mc_answer4"]]
        correct = opts[int(r["correct_answer_num"]) - 1]
        user = ("Read the passage and answer the question.\n\n"
                f"Passage: {r['flores_passage']}\n\nQuestion: {r['question']}\n\n"
                "Options:\n- " + "\n- ".join(opts))
        norm = [{"role": "user", "content": user}, {"role": "assistant", "content": correct}]
        h = row_hash(norm)
        if h in banned:
            continue
        banned.add(h)
        out.append({"domain": f"belebele_{lang}", "source": "facebook/belebele",
                    "conversations": norm})
        got += 1
    return out


def build_probe(args, rng, banned):
    rows = []
    for subset, target in args.targets.items():
        if target <= 0:
            continue
        pool = pool_for(args.pool, subset, int(target * OVERSAMPLE), rng, banned)
        sel = stratified(pool, min(target, len(pool)), rng)
        rows += [{"domain": subset, "generator": gen, "source": CASCADE_SOURCE,
                  "conversations": msgs} for msgs, gen in sel]
        print(f"{subset}: {len(sel)}/{target}", flush=True)
    for lang in BELEBELE_LANGS:
        rows += belebele_rows(lang, args.ml_per_lang, rng, banned, args.raw_dir)
    rng.shuffle(rows)
    return rows


def hf_parquet_urls(dataset, config):
    out = subprocess.run(
        ["curl", "-sL", f"https://huggingface.co/api/datasets/{dataset}/parquet"],
        capture_output=True, text=True,
    ).stdout
    d = json.loads(out)
    splits = d.get(config) or next(iter(d.values()), {})
    return next(iter(splits.values()), []) if splits else []


def curl(url, dest):
    subprocess.run(["curl", "-sL", "--retry", "3", "-o", str(dest), url], check=True)


def build_guard(args, rng, banned):
    raw_dir = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # no_robots: human-written general chat/IF (Coding excluded)
    dest = raw_dir / "no_robots_0.parquet"
    if not dest.exists():
        curl(hf_parquet_urls("HuggingFaceH4/no_robots", "default")[0], dest)
    t = pq.read_table(dest, columns=["messages", "category"])
    pool = [(c, m) for c, m in zip(t["category"].to_pylist(), t["messages"].to_pylist())
            if c != "Coding"]
    rng.shuffle(pool)
    got = 0
    for cat, msgs in pool:
        if got >= args.n_robots:
            break
        norm = [{"role": m["role"], "content": m.get("content") or ""} for m in msgs]
        if not any(m["role"] == "assistant" for m in norm):
            continue
        if row_hash(norm) in banned:
            continue
        rows.append({"domain": "no_robots", "source": "HuggingFaceH4/no_robots",
                     "category": cat, "conversations": norm})
        got += 1

    for lang in BELEBELE_LANGS:
        rows += belebele_rows(lang, 4, rng, banned, raw_dir)
    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["calib", "probe", "guard"], required=True)
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--pool", default="data-raw/cascade",
                    help="dir with Nemotron-Cascade-2 parquet shards")
    ap.add_argument("--raw-dir", default="data-raw", help="cache for guard downloads")
    ap.add_argument("--n", type=int, default=1000, help="calib rows (mode=calib)")
    ap.add_argument("--targets-json", default=None,
                    help="mode=probe domain targets as a JSON dict "
                         "(default: the frozen probe-v4 allocation)")
    ap.add_argument("--ml-per-lang", type=int, default=ML_PER_LANG)
    ap.add_argument("--n-robots", type=int, default=40)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="jsonl glob(s) whose conversations must not be sampled")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if args.targets_json:
        args.targets = json.loads(args.targets_json)
    else:
        args.targets = V4_TARGETS

    pool = Path(args.pool)
    args.pool = str(pool if pool.is_absolute() else ROOT / pool)
    raw = Path(args.raw_dir)
    args.raw_dir = raw if raw.is_absolute() else ROOT / raw

    rng = random.Random(args.seed)
    banned = banned_hashes(args.exclude)

    if args.mode == "calib":
        rows = build_calib(args, rng, banned)
    elif args.mode == "probe":
        rows = build_probe(args, rng, banned)
    else:
        rows = build_guard(args, rng, banned)

    own = [row_hash(r["conversations"]) for r in rows]
    assert len(own) == len(set(own)), "duplicate rows in output"
    assert not (set(own) & banned), "output overlaps banned set"

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n{args.mode}: {len(rows)} rows -> {out}")
    print("domains:", dict(Counter(
        "ML" if (r.get("domain") or "").startswith("belebele") else r.get("domain")
        for r in rows)))


if __name__ == "__main__":
    main()
