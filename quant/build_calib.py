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
import os
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

# Provenance ledger: one record per upstream byte we read this run. Written
# next to the output as <out>.sources.json — transparency contract: any set
# this script builds can name exactly which dataset revisions + file hashes
# fed it. Rebuilds that silently drift (upstream reshards) are detectable.
SOURCES = []

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
    """Hashes of every frozen set in data/ + user-provided excludes.
    Every file opened here gates row selection, so each is recorded in the
    provenance ledger (the sidecar must enumerate ALL selection inputs,
    not just the pools sampled from)."""
    banned = set()
    for frozen in ["calib-cascade-v1.jsonl", "probe-v4.jsonl", "probe-guard.jsonl"]:
        p = DATA / frozen
        if p.exists():
            record_source("frozen-set (exclusion input)", p)
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    msgs = r.get("messages") or r.get("conversations")
                    banned.add(row_hash(msgs))
    for pattern in extra_excludes:
        for p in glob.glob(str(Path(pattern).absolute() if Path(pattern).is_absolute()
                               else ROOT / pattern)):
            record_source("--exclude (exclusion input)", p)
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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_dataset_revision(dataset):
    """Current HEAD commit sha of a HF dataset repo (None offline/stale cache)."""
    out = subprocess.run(
        ["curl", "-sL", "--connect-timeout", "10", "--max-time", "30",
         f"https://huggingface.co/api/datasets/{dataset}"],
        capture_output=True, text=True,
    ).stdout
    try:
        d = json.loads(out)
        return d.get("sha") if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def record_source(dataset, path, **extra):
    # repo-relative when the file lives in the checkout (sidecars are meant
    # to be diffed/shared — don't leak the builder's absolute paths);
    # external scratch dirs keep their absolute path as the honest record
    p = Path(path).resolve()
    try:
        loc = str(p.relative_to(ROOT))
    except ValueError:
        loc = str(p)
    rec = {"dataset": dataset, "file": loc, "sha256": sha256_file(p)}
    rec.update(extra)
    SOURCES.append(rec)
    return rec


def pool_for(pool_dir, subset, need, rng, banned):
    """Randomized row-group scan across the subset's parquet shards."""
    files = sorted(glob.glob(f"{pool_dir}/{subset}_*.parquet"))
    if not files:
        sys.exit(f"no parquet shards for {subset!r} under {pool_dir} — download the "
                 f"pool first (see module docstring)")
    recorded = set()  # shard -> provenance recorded on FIRST actual read
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
            if f not in recorded:  # local pool snapshot — no repo revision exists
                record_source(f"{CASCADE_SOURCE} (local pool)", f)
                recorded.add(f)
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
    fresh = not dest.exists()
    rev, split = None, "unknown (cached)"
    if fresh:
        rev = hf_dataset_revision("facebook/belebele")
        split, urls = hf_parquet_urls("facebook/belebele", lang)
        if not urls:
            sys.exit(f"no converted parquet for facebook/belebele[{lang}]")
        curl(urls[0], dest)
    record_source("facebook/belebele", dest, config=lang, split=split,
                  revision=rev or ("revision-lookup-failed" if fresh
                                   else "cached-before-this-run"))
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


def hf_parquet_urls(dataset, config, split=None):
    """Converted-parquet URLs for a dataset config. Returns (split, urls).
    An explicitly requested split MUST exist (hard exit otherwise — silent
    fallback is the exact hazard this records against). With split=None the
    API's first split is used and its name returned so callers record what
    they actually got (API split order is not guaranteed)."""
    out = subprocess.run(
        ["curl", "-sL", "--connect-timeout", "10", "--max-time", "30",
         f"https://huggingface.co/api/datasets/{dataset}/parquet"],
        capture_output=True, text=True,
    )
    try:
        d = json.loads(out.stdout)
    except json.JSONDecodeError:
        sys.exit(f"non-JSON response from parquet API for {dataset} "
                 f"(rc={out.returncode}, body: {out.stdout[:120]!r})")
    if config not in d or not d[config]:
        sys.exit(f"{dataset} has no parquet config {config!r} "
                 f"(available: {sorted(d)})")
    splits = d[config]
    if not splits:
        return split, []
    if split is not None:
        if split not in splits:
            sys.exit(f"{dataset}[{config}] has no split {split!r} "
                     f"(available: {sorted(splits)})")
        return split, splits[split]
    name = next(iter(splits))
    return name, splits[name]


def curl(url, dest):
    # --fail: an HTTP error must not "succeed" and poison the cache with an
    # error page (which record_source would then hash-pin as authoritative);
    # download to a temp file + atomic rename so partial downloads never cache
    tmp = f"{dest}.part"
    try:
        subprocess.run(["curl", "-sL", "--fail", "--retry", "3",
                        "--connect-timeout", "15", "--max-time", "600",
                        "-o", tmp, url], check=True)
        os.replace(tmp, dest)
    except subprocess.CalledProcessError as e:
        sys.exit(f"download failed ({e.returncode}) for {url} -> {dest}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def build_guard(args, rng, banned):
    raw_dir = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # no_robots: human-written general chat/IF (Coding excluded).
    # The frozen probe-guard's rows are from the TRAIN split (verified against
    # the cached upstream arrows) — the split is explicit here because the
    # converted-parquet API has served split lists in unstable order (test has
    # come back first on some calls), which would silently switch splits.
    dest = raw_dir / f"no_robots_{args.no_robots_split}.parquet"
    fresh = not dest.exists()
    rev, split = None, args.no_robots_split  # filename pins it when cached
    if fresh:
        rev = hf_dataset_revision("HuggingFaceH4/no_robots")
        split, urls = hf_parquet_urls("HuggingFaceH4/no_robots", "default",
                                      split=args.no_robots_split)
        if not urls:
            sys.exit("no converted parquet for HuggingFaceH4/no_robots")
        curl(urls[0], dest)
    record_source("HuggingFaceH4/no_robots", dest, split=split,
                  revision=rev or ("revision-lookup-failed" if fresh
                                   else "cached-before-this-run"))
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
    ap.add_argument("--no-robots-split", choices=["train", "test"], default="train",
                    help="no_robots split to sample (default train — the frozen "
                         "probe-guard's rows are train-split, verified against "
                         "the cached upstream arrows; the parquet API's split "
                         "ordering is unstable, so the split is explicit)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="jsonl glob(s) whose conversations must not be sampled")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    from lablog import log_event
    log_event("RUN_START", mode=args.mode, out=args.out)

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
    # `banned` is mutated during the build (selected rows are added so later
    # subsets can't re-pick them) — snapshot the frozen+excludes hashes now;
    # the output must be disjoint from THOSE, not from itself.
    pre_banned = set(banned)

    if args.mode == "calib":
        rows = build_calib(args, rng, banned)
    elif args.mode == "probe":
        rows = build_probe(args, rng, banned)
    else:
        rows = build_guard(args, rng, banned)

    own = [row_hash(r["conversations"]) for r in rows]
    # explicit raises, not asserts: the disjointness contract must survive
    # `python -O` (asserts are stripped under -O)
    if len(own) != len(set(own)):
        raise RuntimeError("duplicate rows in output")
    if set(own) & pre_banned:
        raise RuntimeError("output overlaps banned set")

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # provenance sidecar: every upstream file this build read, hash-pinned;
    # output path normalized the same way as source paths (no host leakage)
    sidecar = out.with_suffix("").with_name(out.name + ".sources.json")
    try:
        out_loc = str(out.resolve().relative_to(ROOT))
    except ValueError:
        out_loc = str(out.resolve())
    with open(sidecar, "w") as f:
        json.dump({"mode": args.mode, "seed": args.seed, "output": out_loc,
                   "sources": SOURCES}, f, indent=1)
    print(f"\n{args.mode}: {len(rows)} rows -> {out}")
    print(f"sources manifest: {sidecar}")
    from lablog import run_done
    run_done(mode=args.mode, rows=len(rows), sidecar=str(sidecar))
    print("domains:", dict(Counter(
        "ML" if (r.get("domain") or "").startswith("belebele") else r.get("domain")
        for r in rows)))


if __name__ == "__main__":
    main()
