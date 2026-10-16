"""Assemble Draft-OPD boundary records from opd_collect wave output.

Verified semantics (dry-run 2026-10-16, vLLM 0.31.0 dflash):
  - Each step record: cu_num_logits slices draft_sampled/target rows per
    request SLOT; slot order can permute mid-wave.
  - draft_sampled[cu[i]][0] is an ANCHOR ECHO: the request's last emitted
    token. Slots are attributed per step by matching the echo against live
    requests' last tokens.
  - draft_sampled[cu[i]+1 : cu[i]+16] are the 15 proposals (block includes
    the anchor slot; num_speculative_steps=15).
  - sampled[i][:num_sampled[i]] = tokens emitted this step: accepted
    proposals followed by the target's correction (or bonus on full accept).
  - Full stream = output_token_ids[0] (prefill token) + concatenation.

Output anchors.jsonl, one record per drafted block:
  {id, stream_offset, anchor_token, block_tokens (15 proposals),
   accepted_len, correction_token, p_top (per proposal position)}
Validation: every request's assembled stream must equal its returned
output_token_ids exactly; any mismatch drops the whole wave.

Usage: python drafter/opd_assemble.py data/opd_boundaries/<arm>
"""
import json
import sys
from collections import Counter
from pathlib import Path


def assemble_wave(steps, outputs):
    reqs = {i: {"ptr": 1, "stream": o["output_token_ids"],
                "id": o["id"], "anchors": []}  # ptr=1: prefill token consumed
           for i, o in enumerate(outputs)}
    for step in steps:
        cu = step["cu_num_logits"]
        drafts = step["draft_sampled"]
        sampled = step["sampled"]
        n_sampled = step["num_sampled"]
        p_top = step["p_top"]
        live = [i for i, r in reqs.items() if r["ptr"] < len(r["stream"])]
        used = set()
        for slot in range(len(cu) - 1):
            if slot >= len(n_sampled):
                break
            lo = cu[slot]
            echo = drafts[lo]
            # attribute by anchor echo == request's last emitted token
            target = None
            for i in live:
                if i in used:
                    continue
                last = reqs[i]["stream"][reqs[i]["ptr"] - 1]
                if last == echo:
                    target = i
                    break
            if target is None:
                continue
            r = reqs[target]
            n = n_sampled[slot]
            chunk = sampled[slot][:n]
            if r["stream"][r["ptr"]:r["ptr"] + n] != chunk:
                return None  # attribution failure — drop wave
            block = drafts[lo + 1: lo + 16]  # 15 proposals after anchor echo
            # accepted = leading chunk tokens that came from the block
            acc = 0
            while acc < len(chunk) - 1 and acc < len(block) \
                    and chunk[acc] == block[acc]:
                acc += 1
            r["anchors"].append({
                "id": r["id"],
                "stream_offset": r["ptr"] - 1,  # position of the anchor token
                "anchor_token": echo,
                "block_tokens": block,
                "accepted_len": acc,
                "correction_token": chunk[-1] if acc < len(block) else None,
                "p_top": p_top[lo: lo + 15],  # rows lo+k = target dist
                # conditioned through draft position k (proposal k's target)
            })
            r["ptr"] += n
            used.add(target)
    if any(r["ptr"] != len(r["stream"]) for r in reqs.values()):
        # trailing overhang after finish is fine; deficit is not
        if any(r["ptr"] < len(r["stream"]) for r in reqs.values()):
            return None
    anchors = [a for r in reqs.values() for a in r["anchors"]]
    return anchors


def main():
    d = Path(sys.argv[1])
    all_anchors = []
    waves = sorted(d.glob("wave*_steps.jsonl"))
    dropped = 0
    for wf in waves:
        of = wf.parent / wf.name.replace("_steps", "_outputs")
        try:
            steps = [json.loads(x) for x in open(wf)]
            outputs = [json.loads(x) for x in open(of)]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            dropped += 1
            print(f"[opd_assemble] DROP {wf.name} (unreadable wave: {e})")
            continue
        try:
            res = assemble_wave(steps, outputs)
        except (KeyError, IndexError, ValueError) as e:
            dropped += 1
            print(f"[opd_assemble] DROP {wf.name} (malformed record: {e})")
            continue
        if res is None:
            dropped += 1
            print(f"[opd_assemble] DROP {wf.name} (attribution mismatch)")
            continue
        all_anchors.extend(res)

    tmp = d / "anchors.jsonl.tmp"
    with open(tmp, "w") as f:
        for a in all_anchors:
            f.write(json.dumps(a) + "\n")
    tmp.replace(d / "anchors.jsonl")  # atomic: no truncated anchors.jsonl

    accs = [a["accepted_len"] + 1 for a in all_anchors]
    full = sum(1 for a in all_anchors if a["correction_token"] is None)
    hist = Counter(a["accepted_len"] for a in all_anchors)
    print(f"[opd_assemble] {len(all_anchors)} anchors from "
          f"{len(waves) - dropped}/{len(waves)} waves "
          f"(mean accepted+1 = {sum(accs) / max(len(accs), 1):.3f}, "
          f"full-accept {full})")
    print(f"[opd_assemble] accepted-len histogram: {dict(sorted(hist.items()))}")


if __name__ == "__main__":
    main()
