#!/usr/bin/env python3
"""Decompose ifeval battery samples into think-mode strata.

For a base/quant pair of lm-eval ifeval sample files, classify each doc's
judged text (filtered_resps) as direct answer vs unclosed-think (raw
reasoning graded as answer), then split the paired strict-acc delta into
mode-flip vs within-stratum components. Stdlib only.

Usage: ifeval_modes.py BASE.jsonl QUANT.jsonl [--json out]
"""
import argparse, json, random, statistics, sys
from collections import Counter


def load(path):
    rows = {}
    for line in open(path):
        rec = json.loads(line)
        rows[rec["doc_id"]] = rec
    return rows


def mode(rec):
    t = rec["filtered_resps"][0]
    if "<think>" in t and "</think>" not in t:
        return "think"
    return "direct"


def val(rec, m="prompt_level_strict_acc"):
    x = rec[m]
    return x[0] if isinstance(x, list) else x


def boot(diffs, n=10000, seed=0):
    rng = random.Random(seed)
    if not diffs:
        return None, None
    L = len(diffs)
    means = sorted(sum(diffs[rng.randrange(L)] for _ in range(L)) / L for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"), ap.add_argument("quant")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    b, q = load(a.base), load(a.quant)
    common = sorted(set(b) & set(q))
    sc = Counter((mode(b[i]), mode(q[i])) for i in common)
    print(f"n paired: {len(common)}")
    print(f"strata base->quant: {dict(sc)}")
    for label, rows in (("base", b), ("quant", q)):
        for s in ("direct", "think"):
            ids = [i for i in common if mode(rows[i]) == s]
            if ids:
                ps = 100 * sum(val(rows[i]) for i in ids) / len(ids)
                print(f"{label:5s} {s:6s} n={len(ids):3d} strict={ps:5.1f}%")
    tot = flip = within = 0.0
    for i in common:
        d = (val(q[i]) - val(b[i])) * 100
        tot += d
        if mode(b[i]) != mode(q[i]):
            flip += d
        else:
            within += d
    n = len(common)
    print(f"delta total {tot / n:+.2f}pt = mode-flips {flip / n:+.2f} "
          f"+ within-stratum {within / n:+.2f}")
    both = [i for i in common if mode(b[i]) == "direct" and mode(q[i]) == "direct"]
    if both:
        db = 100 * sum(val(b[i]) for i in both) / len(both)
        dq = 100 * sum(val(q[i]) for i in both) / len(both)
        diffs = [(val(q[i]) - val(b[i])) * 100 for i in both]
        lo, hi = boot(diffs)
        print(f"both-direct: n={len(both)} base={db:.1f} quant={dq:.1f} "
              f"delta={dq - db:+.1f} CI[{lo:+.1f},{hi:+.1f}]")
    if a.json:
        json.dump({"n": n, "strata": {f"{k[0]}->{k[1]}": v for k, v in sc.items()},
                   "delta_total": tot / n, "delta_mode_flips": flip / n,
                   "delta_within": within / n,
                   "both_direct": {"n": len(both), "base": db, "quant": dq}
                   if both else None}, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
