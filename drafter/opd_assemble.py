"""Assemble Draft-OPD boundary records from opd_collect wave output.

Layout hypothesis (validated per wave; mismatch drops the wave loudly):
  - logits rows per request = n_i draft positions + 1 bonus row, requests
    back-to-back; sampled_token_ids are row-aligned with those rows.
  - A request's step output = the leading sampled draft-position rows that
    match its draft_token_ids (accepted prefix) + the first non-matching row
    (target correction) or the bonus row on full accept.
  - Concatenating step outputs per request must reproduce its returned
    output_token_ids EXACTLY.

Output: anchors.jsonl — one record per drafted block:
  {id, stream_offset, block_tokens, accepted_len, correction_token,
   p_top: [[token, logprob] per block position],   # target on draft prefix
   q_top: [[token, logprob] per block position]}   # draft distribution
plus summary stats (mean accepted length, first-rejection histogram).

Usage: python drafter/opd_assemble.py data/opd_boundaries/static
"""
import json
import sys
from collections import Counter
from pathlib import Path


def assemble_wave(steps, outputs):
    """Returns (anchors, streams_match) — anchors across all requests."""
    reqs = {o["id"]: {"ptr": 0, "stream": o["output_token_ids"],
                      "prompt_len": len(o["prompt_token_ids"]), "anchors": []}
            for o in outputs}
    live_order = [o["id"] for o in outputs]

    for step in steps:
        n_drafts = step["num_draft_tokens"]
        cu = step["cu_num_draft_tokens"]
        drafts = step["draft_token_ids"]
        sampled = step["sampled_token_ids"]
        p_top, q_top = step["p_top"], step["q_top"]
        # logits rows: per request n_i draft rows + 1 bonus row
        row = 0
        # requests still generating appear in step order; match by stream
        active = [rid for rid in live_order
                  if reqs[rid]["ptr"] < len(reqs[rid]["stream"])]
        if len(active) < len(n_drafts):
            return None  # cannot attribute confidently — drop wave
        for slot, n in enumerate(n_drafts):
            rid = active[slot]
            r = reqs[rid]
            block = drafts[cu[slot] - n:cu[slot]]
            acc = 0
            while acc < n - 1 and sampled[row + acc] == block[acc]:
                acc += 1
            # step output: acc accepted + 1 correction/bonus token
            chunk = sampled[row:row + acc + 1]
            stream = r["stream"]
            if stream[r["ptr"]:r["ptr"] + acc + 1] != chunk:
                return None
            r["anchors"].append({
                "id": rid,
                "stream_offset": r["ptr"],
                "block_tokens": block,
                "accepted_len": acc,
                "correction_token": chunk[-1] if acc < n - 1 else None,
                "p_top": p_top[row:row + n],
                "q_top": q_top[cu[slot] - n:cu[slot]] if q_top else None,
            })
            r["ptr"] += acc + 1
            row += n + 1  # draft rows + bonus row
    # every request must be fully consumed
    if any(r["ptr"] != len(r["stream"]) for r in reqs.values()):
        return None
    anchors = [a for r in reqs.values() for a in r["anchors"]]
    return anchors, reqs


def main():
    d = Path(sys.argv[1])
    all_anchors = []
    waves = sorted(d.glob("wave*_steps.jsonl"))
    dropped = 0
    for wf in waves:
        of = wf.parent / wf.name.replace("_steps", "_outputs")
        steps = [json.loads(x) for x in open(wf)]
        outputs = [json.loads(x) for x in open(of)]
        res = assemble_wave(steps, outputs)
        if res is None:
            dropped += 1
            print(f"[opd_assemble] DROP {wf.name} (attribution mismatch)")
            continue
        anchors, reqs = res
        all_anchors.extend(anchors)
    with open(d / "anchors.jsonl", "w") as f:
        for a in all_anchors:
            f.write(json.dumps(a) + "\n")

    accs = [a["accepted_len"] + 1 for a in all_anchors]  # incl. correction
    first_rej = Counter(
        a["accepted_len"] if a["accepted_len"] < len(a["block_tokens"]) - 1
        else "full" for a in all_anchors)
    print(f"[opd_assemble] {len(all_anchors)} anchors from "
          f"{len(waves) - dropped}/{len(waves)} waves "
          f"(mean accepted+1 = {sum(accs) / max(len(accs), 1):.3f})")
    print(f"[opd_assemble] first-reject position histogram: "
          f"{dict(sorted(first_rej.items(), key=lambda x: str(x[0])))}")


if __name__ == "__main__":
    main()
