#!/usr/bin/env python3
"""Paired base-vs-quant report for lm-eval capability battery outputs.

Reads the two output dirs produced by quant/capability_battery.sh
(lm-eval --output_path <dir> --log_samples), aligns per-doc scores by
doc_id across sides, and reports per-task deltas with paired-bootstrap
CIs.

Doctrine (README / RESULTS.md): capability evals are CLIFF DETECTORS —
report paired deltas with uncertainty, never rankings; only |delta| >=
CLIFF_PT (default 5) points is actionable at these n. Stdlib only.

Usage:
  battery_report.py --base runs/<...>/raw/base --quant runs/<...>/raw/quant \
      [--json out.json] [--cliff_pt 5]
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from pathlib import Path

# primary metric per battery task; first candidate found in the results
# dict wins. Keys arrive as "metric" or "metric,filter"; stderr variants
# are excluded explicitly.
TASK_METRICS = {
    "humaneval_plus_instruct": ["pass@1"],
    "humaneval_plus": ["pass@1"],
    "mbpp_plus_instruct": ["pass_at_1", "pass@1"],
    "mbpp_plus": ["pass_at_1", "pass@1"],
    "ifeval": ["prompt_level_strict_acc"],
    "minerva_math500": ["exact_match"],
    # gpqa exposes two exact_match filters; flexible-extract is the headline
    # metric (strict is near-floor for this family) — pin it so the paired
    # bootstrap runs on the filter we quote
    "gpqa_diamond_cot_zeroshot": ["exact_match,flexible-extract", "exact_match"],
}
DEFAULT_METRICS = ["pass@1", "pass_at_1", "prompt_level_strict_acc", "exact_match", "acc"]


def find_one(pattern, side_dir):
    hits = sorted(glob.glob(os.path.join(side_dir, "**", pattern), recursive=True))
    return hits[0] if len(hits) == 1 else hits


def load_results(side_dir):
    """Return (path, {task: {metric_key: value}}) from a side's results_*.json."""
    path = find_one("results_*.json", side_dir)
    if isinstance(path, list):
        sys.exit(f"{side_dir}: expected exactly one results_*.json, found {len(path)}: {path}")
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for task, metrics in raw.get("results", {}).items():
        clean = {k: v for k, v in metrics.items()
                 if not k.startswith("alias") and "stderr" not in k}
        out[task] = clean
    return path, out


def pick_metric(task, metrics):
    # pass 1: exact key (candidates may pin a filter, e.g.
    # "exact_match,flexible-extract"); pass 2: metric-part only
    for cand in TASK_METRICS.get(task, []) + DEFAULT_METRICS:
        for key in metrics:
            if key == cand:
                return key
    for cand in TASK_METRICS.get(task, []) + DEFAULT_METRICS:
        for key in metrics:
            if key.split(",")[0] == cand:
                return key
    return None


def load_samples(side_dir, task, key):
    """Per-doc scores {doc_id: float} from samples_{task}_*.jsonl.

    Metric fields arrive as "metric" or "metric,filter"; the resolved
    results key is matched first, then a unique prefix match (so a bare
    sample field still aligns with a suffixed aggregate key). Ambiguous
    records (e.g. strict-match AND flexible-extract present) are counted
    and reported, never silently picked.
    """
    pats = [f"samples_{task}_*.jsonl", f"samples_{task}.jsonl"]
    paths = []
    for pat in pats:
        for hit in sorted(glob.glob(os.path.join(side_dir, "**", pat), recursive=True)):
            # exact stem match only: samples_<task>_<timestamp>.jsonl — a bare
            # prefix glob would also ingest sibling tasks whose name extends
            # this one (humaneval_plus vs humaneval_plus_instruct)
            core = os.path.basename(hit)[: -len(".jsonl")]
            suffix = core[len(f"samples_{task}_"):]
            if core == f"samples_{task}" or re.fullmatch(r"\d{4}-\d{2}-\d{2}T[\d.\-]+Z?", suffix):
                paths.append(hit)
    rows = {}
    unresolved = 0
    dupes = 0
    bad = 0
    stem = key.split(",")[0]
    suffix = key.split(",", 1)[1] if "," in key else None
    for path in paths:
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "doc_id" not in rec:
                        continue
                    # when the aggregate key pins a filter (e.g.
                    # exact_match,flexible-extract), keep only the sample
                    # records logged for THAT filter — records for other
                    # filters carry the same bare metric field and must not
                    # overwrite the pinned one
                    if suffix is not None and rec.get("filter") not in (None, suffix):
                        continue
                    if key in rec:
                        val = float(rec[key])
                    else:
                        cands = [k for k in rec if k.split(",")[0] == stem and "stderr" not in k]
                        if len(cands) != 1:
                            unresolved += 1
                            continue
                        val = float(rec[cands[0]])
                except (json.JSONDecodeError, ValueError, TypeError):
                    bad += 1  # truncated tail etc. — degrade one doc, not the report
                    continue
                if rec["doc_id"] in rows:
                    dupes += 1  # duplicate doc for the same filter — ambiguous
                    continue
                rows[rec["doc_id"]] = val
    if bad:
        print(f"  ({task}: skipped {bad} malformed sample records)", file=sys.stderr)
    if dupes:
        print(f"  ({task}: ignored {dupes} duplicate doc_id records)", file=sys.stderr)
    if unresolved:
        print(f"  ({task}: {unresolved} sample records lack an unambiguous "
              f"'{key.split(',')[0]}' field)", file=sys.stderr)
    return rows


def paired_bootstrap(diffs, n_boot=10000, seed=0):
    """95% percentile CI on the mean of paired differences."""
    if not diffs:
        return None, None
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def load_resp_lengths(side_dir, task, key, cap, tok):
    """{doc_id: generated tokens} for a task's samples.

    Tokenizes the STORED response (post think-strip). Closed-think rows
    store only the answer (short); rows truncated at the generation cap
    never closed, so the strip never fired and they store ~cap tokens of
    raw reasoning — length>=cap-50 flags them. Same file/filter discipline
    as load_samples so doc alignment holds.
    """
    pats = [f"samples_{task}_*.jsonl", f"samples_{task}.jsonl"]
    paths = []
    for pat in pats:
        for hit in sorted(glob.glob(os.path.join(side_dir, "**", pat), recursive=True)):
            core = os.path.basename(hit)[: -len(".jsonl")]
            suffix = core[len(f"samples_{task}_"):]
            if core == f"samples_{task}" or re.fullmatch(r"\d{4}-\d{2}-\d{2}T[\d.\-]+Z?", suffix):
                paths.append(hit)
    suffix = key.split(",", 1)[1] if "," in key else None
    rows = {}
    texts = []
    for path in paths:
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "doc_id" not in rec or rec["doc_id"] in rows:
                    continue
                if suffix is not None and rec.get("filter") not in (None, suffix):
                    continue
                r = rec.get("resps") or []
                if not r or not r[0]:
                    continue
                rows[rec["doc_id"]] = None
                texts.append((rec["doc_id"], r[0][0] if isinstance(r[0], list) else str(r[0])))
    for doc_id, chunk in zip([d for d, _ in texts],
                             tok([t for _, t in texts], add_special_tokens=False)["input_ids"]):
        rows[doc_id] = len(chunk)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="lm-eval output dir (base side)")
    ap.add_argument("--quant", required=True, help="lm-eval output dir (quant side)")
    ap.add_argument("--json", default=None, help="write full report JSON here")
    ap.add_argument("--cliff_pt", type=float, default=5.0,
                    help="actionable |delta| threshold in points (default 5)")
    ap.add_argument("--gen_cap", type=int, default=0,
                    help="generation cap in tokens; when set, ALSO report "
                         "paired-complete scoring (rows where BOTH sides "
                         "finished under the cap) and per-side truncation "
                         "rates — truncation becomes a measured length "
                         "metric, never a graded miss (v2 protocol)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B",
                    help="tokenizer for the truncation metric (needs "
                         "transformers; runs fine from either lab venv)")
    args = ap.parse_args()

    tok = None
    if args.gen_cap:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            sys.exit(f"--gen_cap needs transformers: {e}")
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    base_path, base_res = load_results(args.base)
    quant_path, quant_res = load_results(args.quant)
    tasks = sorted(set(base_res) & set(quant_res))
    if not tasks:
        sys.exit(f"no common tasks between sides\n base: {sorted(base_res)}\n quant: {sorted(quant_res)}")

    print(f"base : {base_path}\nquant: {quant_path}")
    print(f"{'task':34s} {'metric':24s} {'n':>4s} {'base':>7s} {'quant':>7s} "
          f"{'delta':>7s} {'95% CI':>17s} cliff")
    report = {"cliff_pt": args.cliff_pt, "tasks": {}}
    worst = None
    for task in tasks:
        # resolve per side and require the SAME metric key on both — a
        # differing filter suffix (exact_match,none vs ,flexible-extract)
        # would compare unlike quantities; skip loudly instead.
        kb = pick_metric(task, base_res[task])
        kq = pick_metric(task, quant_res[task])
        if kb is None or kq is None or kb != kq:
            print(f"{task:34s} (unresolvable metric — skipped; base key: {kb}, quant key: {kq})")
            continue
        # a pinned filter candidate that didn't win means the report is
        # about to run on some OTHER filter — say so instead of going quiet
        pinned = next((c for c in TASK_METRICS.get(task, []) if "," in c), None)
        if pinned and kb != pinned:
            print(f"  ({task}: pinned '{pinned}' absent — using '{kb}')", file=sys.stderr)
        b, q = base_res[task][kb], quant_res[task][kb]
        delta = (q - b) * 100.0

        bs, qs = load_samples(args.base, task, kb), load_samples(args.quant, task, kb)
        common = sorted(set(bs) & set(qs))
        lo = hi = None
        if common:
            diffs = [(qs[i] - bs[i]) * 100.0 for i in common]
            lo, hi = paired_bootstrap(diffs)
            ci = f"[{lo:+.1f}, {hi:+.1f}]"
        else:
            ci = "(no aligned samples)"

        cliff = abs(delta) >= args.cliff_pt and lo is not None and (lo > 0 or hi < 0)
        name = kb.split(",")[0]
        n_str = str(len(common)) if common else "-"
        print(f"{task:34s} {name:24s} {n_str:>4s} "
              f"{b * 100:6.1f} {q * 100:6.1f} {delta:+6.1f} {ci:>17s} "
              f"{'YES' if cliff else ''}")
        entry = {
            "metric": name, "metric_key": kb, "n_paired": len(common),
            "base": b, "quant": q, "delta_pt": delta,
            "ci_lo_pt": lo, "ci_hi_pt": hi, "cliff": cliff,
        }
        # v2: truncation is a length metric, never a graded miss. Score the
        # pair on rows where BOTH sides finished; report per-side cut rates.
        if args.gen_cap and common:
            bl = load_resp_lengths(args.base, task, kb, args.gen_cap, tok)
            ql = load_resp_lengths(args.quant, task, kb, args.gen_cap, tok)
            cut = args.gen_cap - 50
            bcap = {i for i in common if bl.get(i, 0) >= cut}
            qcap = {i for i in common if ql.get(i, 0) >= cut}
            keep = [i for i in common if i not in bcap and i not in qcap]
            pdelta = ((sum(qs[i] for i in keep) - sum(bs[i] for i in keep)) / len(keep)) * 100.0 if keep else None
            plo, phi = paired_bootstrap([(qs[i] - bs[i]) * 100.0 for i in keep]) if keep else (None, None)
            print(f"{'  pc (both sides complete)':34s} {'':24s} {len(keep):>4d} "
                  f"{(sum(bs[i] for i in keep) / len(keep)) * 100:6.1f} "
                  f"{(sum(qs[i] for i in keep) / len(keep)) * 100:6.1f} "
                  f"{pdelta:+6.1f} [{plo:+.1f}, {phi:+.1f}]" if keep else
                  f"{'  pc (both sides complete)':34s} {'':24s}    0  (no complete pairs)")
            print(f"{'  trunc@cap':34s} base {len(bcap)}/{len(common)} "
                  f"({100.0 * len(bcap) / len(common):.0f}%)  quant {len(qcap)}/{len(common)} "
                  f"({100.0 * len(qcap) / len(common):.0f}%)  [length metric, not scored]")
            entry |= {"n_paired_complete": len(keep),
                      "pc_base": (sum(bs[i] for i in keep) / len(keep)) if keep else None,
                      "pc_quant": (sum(qs[i] for i in keep) / len(keep)) if keep else None,
                      "pc_delta_pt": pdelta, "pc_ci_lo_pt": plo, "pc_ci_hi_pt": phi,
                      "trunc_base": len(bcap), "trunc_quant": len(qcap), "n_common": len(common)}
            # decision metric under --gen_cap: paired-complete delta
            delta, lo, hi = pdelta, plo, phi
            cliff = abs(delta) >= args.cliff_pt and lo is not None and (lo > 0 or hi < 0)
            entry["cliff"] = cliff
        report["tasks"][task] = entry
        if worst is None or abs(delta) > abs(worst["delta_pt"]):
            worst = report["tasks"][task] | {"task": task}
    report["worst"] = worst

    compared = [t for t, r in report["tasks"].items() if r["ci_lo_pt"] is not None]
    if not compared:
        sys.exit("\nno task produced a paired comparison — see skipped tasks and "
                 "sample-alignment warnings above (missing --log_samples on a side? "
                 "metric-key drift?)")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.json}")
    if worst:
        ci = (f"(CI [{worst['ci_lo_pt']:+.1f}, {worst['ci_hi_pt']:+.1f}])"
              if worst["ci_lo_pt"] is not None else "(no aligned samples)")
        print(f"\nworst: {worst['task']} {worst['delta_pt']:+.1f}pt {ci}")


if __name__ == "__main__":
    main()
