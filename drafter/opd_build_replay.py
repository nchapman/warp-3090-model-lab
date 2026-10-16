"""Build Draft-OPD replay training rows from collected boundary anchors.

Layout (verified against opd_collect records, vLLM 0.31.0):
  - anchors.jsonl rows carry stream_offset = OUTPUT position of the anchor
    token; proposal k targets output position stream_offset + 1 + k.
  - p_top rows (one per proposal) are [tokens(K), logprobs(K)] lists;
    -inf logprobs (masked vocab region) are dropped.

Per anchor (arXiv:2605.29343 replay): input_ids = prompt + accepted tokens
including the anchor + the drafter's 15 proposals (draft tokens in context —
extract_states then computes target states on draft-induced prefixes, the
paper's replay state computation). loss_mask marks the proposal positions;
supervision rides in opd_* columns consumed by the draft_opd trainer patch:
  opd_kl_mode      1 = accepted (CE toward the stream token — forward-KL
                     equivalent), 2 = rejected suffix (reverse-KL vs stored
                     target top-k)
  opd_target_ids   mode-1 CE target (int32, -1 elsewhere)
  opd_pos_weight   1.0 on accepted, gamma^(k-acc-1) on the rejected suffix
  opd_p_topk_*     [seq_len, K] target top-k tokens/logprobs (-1 padding)

Usage: python drafter/opd_build_replay.py data/opd_boundaries/<arm> \
          --out data/opd_replay_<arm>
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("boundaries",
                    help="opd_collect out dir with anchors.jsonl + wave outputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gamma", type=float, default=0.8,
                    help="rejected-suffix decay (paper: 0.8, robust 0.5-0.8)")
    args = ap.parse_args()

    d = Path(args.boundaries)
    streams = {}
    prompt_lens = {}
    for of in sorted(d.glob("wave*_outputs.jsonl")):
        for line in open(of):
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[opd_build_replay] skip malformed line in "
                      f"{of.name}: {e}")
                continue
            streams[r["id"]] = r["prompt_token_ids"] + r["output_token_ids"]
            prompt_lens[r["id"]] = len(r["prompt_token_ids"])
    anchors = []
    try:
        anchors = [json.loads(x) for x in open(d / "anchors.jsonl")]
    except FileNotFoundError:
        raise SystemExit(f"[opd_build_replay] {d / 'anchors.jsonl'} missing "
                         "— run opd_assemble first")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[opd_build_replay] anchors.jsonl unreadable: {e}")
    if not anchors:
        raise SystemExit("[opd_build_replay] anchors.jsonl is empty — "
                         "collection or assembly upstream produced nothing")

    K = len(anchors[0]["p_top"][0][0])
    n_rows = n_acc = n_rej = n_skip = 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "replay.jsonl.tmp"
    with open(tmp, "w") as f:
        for a in anchors:
            rid = a["id"]
            if rid not in streams:
                n_skip += 1
                continue
            stream = streams[rid]
            base = prompt_lens[rid] + a["stream_offset"] + 1  # first proposal
            block = a["block_tokens"]
            acc = a["accepted_len"]
            n = len(block)
            if base + n > len(stream) + 1:  # block overruns the stream
                n_skip += 1
                continue
            tail = stream[base + n] if base + n < len(stream) else block[-1]
            input_ids = list(stream[:base]) + list(block) + [tail]
            L = len(input_ids)

            kl_mode = np.zeros(L, dtype=np.int8)
            target_ids = np.full(L, -1, dtype=np.int32)
            pos_weight = np.zeros(L, dtype=np.float32)
            p_tok = np.full((L, K), -1, dtype=np.int32)
            p_lp = np.full((L, K), -1.0, dtype=np.float32)

            for k in range(n):
                pos = base + k
                if k <= acc:  # accepted proposals + the correction token
                    kl_mode[pos] = 1
                    target_ids[pos] = stream[pos]
                    pos_weight[pos] = 1.0
                else:  # rejected suffix: reverse-KL vs stored target top-k
                    kl_mode[pos] = 2
                    pos_weight[pos] = args.gamma ** (k - acc - 1)
                if k < len(a["p_top"]):
                    toks, lps = a["p_top"][k][0], a["p_top"][k][1]
                    j = 0
                    for t, lp in zip(toks, lps):
                        if lp == float("-inf") or j >= K:
                            continue
                        p_tok[pos, j] = t
                        p_lp[pos, j] = lp
                        j += 1
            n_rows += 1
            n_acc += int((kl_mode == 1).sum())
            n_rej += int((kl_mode == 2).sum())
            f.write(json.dumps({
                "id": f"{rid}@{a['stream_offset']}",
                "input_ids": [int(x) for x in input_ids],
                "loss_mask": [1 if base <= i < base + n else 0
                              for i in range(L)],
                "opd_kl_mode": kl_mode.tolist(),
                "opd_target_ids": target_ids.tolist(),
                "opd_pos_weight": pos_weight.tolist(),
                "opd_p_topk_tokens": p_tok.tolist(),
                "opd_p_topk_logprobs": p_lp.tolist(),
            }) + "\n")
    if n_rows == 0:
        raise SystemExit("[opd_build_replay] 0 rows built — anchors reference "
                         "missing/overrun streams; nothing written downstream "
                         "is usable")
    tmp.replace(out / "replay.jsonl")  # atomic: no truncated replay.jsonl
    print(f"[opd_build_replay] {n_rows} anchor rows -> {out}/replay.jsonl "
          f"(accepted {n_acc}, rejected-suffix {n_rej} positions, K={K}, "
          f"skipped {n_skip})")


if __name__ == "__main__":
    main()
