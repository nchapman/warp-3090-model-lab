#!/usr/bin/env python3
"""calib-cascade-if3x — appended-dose IF variant of the frozen cascade-v1.

The dose IS the delta: every cascade-v1 row is kept byte-identical (999,
incl. its 52 instruction_following rows); fresh IF rows are APPENDED to
triple the IF count (52 -> 156, n=1103, IF share 14.1%). Non-IF gradient
mass is untouched — unlike a re-mix, this isolates "more IF rows" with no
displacement and no resample noise. Titration is trivial: the next dose
arm changes only the appended count (+52 = 2x, +208 = 5x).

Fresh IF rows come from the pinned raw pool (data-raw/cascade/) via the
same pool_for/stratified machinery as every other build, hash-disjoint
from all frozen sets (banned_hashes: calib-v1/probe-v4/guard/traj-v1).
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_calib as bc  # noqa: E402
from lablog import log_event, run_done, run_fail  # noqa: E402

IF = "instruction_following"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib", default="data/calib-cascade-v1.jsonl")
    ap.add_argument("--out", default="data/calib-cascade-if3x.jsonl")
    ap.add_argument("--pool", default="data-raw/cascade")
    ap.add_argument("--seed", type=int, default=bc.SEED)
    args = ap.parse_args()

    src = bc.ROOT / args.calib
    rows = [json.loads(l) for l in open(src)]
    src_sha = hashlib.sha256(open(src, "rb").read()).hexdigest()
    n_if = sum(1 for r in rows if r["domain"] == IF)
    append_n = 2 * n_if  # 3x total IF count: keep n_if, append 2*n_if

    rng = random.Random(args.seed)
    banned = bc.banned_hashes([])
    pre_banned = set(banned)  # pool_for mutates banned for in-scan dedup —
    # the disjointness contract is against the FROZEN sets, not the pool itself
    log_event("RUN_START", calib=str(src), out=args.out,
              n_v1=len(rows), n_if_v1=n_if, append_n=append_n,
              seed=args.seed)

    pool = bc.pool_for(args.pool, IF, int(append_n * bc.OVERSAMPLE),
                       rng, banned)
    drawn = bc.stratified(pool, append_n, rng)
    fresh = [{"conversations": msgs,
              "source": bc.CASCADE_SOURCE,
              "domain": IF,
              "generator": gen} for msgs, gen in drawn]

    out_rows = rows + fresh
    rng.shuffle(out_rows)

    # every v1 row byte-identical exactly once; fresh rows disjoint from
    # frozen sets and unique among themselves
    v1_hashes = [bc.row_hash(r["conversations"]) for r in rows]
    out_hashes = [bc.row_hash(r["conversations"]) for r in out_rows]
    assert all(out_hashes.count(h) == 1 for h in v1_hashes), "v1 row lost/duped"
    fresh_hashes = [bc.row_hash(r["conversations"]) for r in fresh]
    assert not (set(fresh_hashes) & pre_banned), "fresh row overlaps frozen set"
    assert len(fresh_hashes) == len(set(fresh_hashes)), "dup fresh rows"

    out = bc.ROOT / args.out
    with open(out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    out_sha = hashlib.sha256(open(out, "rb").read()).hexdigest()

    sidecar = {
        "recipe": "cascade-v1 + appended IF rows (dose variant)",
        "dose": {"v1_rows_kept": len(rows), "if_in_v1": n_if,
                 "if_appended": append_n, "if_total": n_if + append_n,
                 "n_total": len(out_rows),
                 "if_share": round((n_if + append_n) / len(out_rows), 4)},
        "source_calib": str(src), "source_sha256": src_sha,
        "pool": args.pool, "seed": args.seed,
        "out_sha256": out_sha,
    }
    sc_path = out.parent / (out.stem + ".dose.json")
    with open(sc_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"out: {len(out_rows)} rows (IF {n_if}->{n_if + append_n}, "
          f"share {sidecar['dose']['if_share']:.1%}) -> {out}")
    print(f"sha256: {out_sha}")
    print(f"sidecar: {sc_path}")
    run_done(rows=len(out_rows), if_total=n_if + append_n)


if __name__ == "__main__":
    main()
