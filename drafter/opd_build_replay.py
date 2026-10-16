"""Build Draft-OPD replay training rows from collected boundary anchors.

Per anchor (arXiv:2605.29343 replay): input_ids = prompt + accepted tokens so
far + the drafter's full proposed block (draft tokens in context — the
teacher-forced target states on this prefix are computed downstream by
extract_states, which is exactly the paper's "replay" state computation).
loss_mask = 1 on the block positions only. Supervision rides in opd_* columns
consumed by the draft_opd trainer patch:
  opd_kl_mode      1 = accepted (forward-KL/CE to stream token),
                   2 = rejected suffix (reverse-KL vs stored target top-k)
  opd_target_ids   mode-1 CE target (int32, -1 elsewhere)
  opd_pos_weight   rejected-suffix decay w_k = gamma^k (gamma default 0.8)
  opd_p_topk_*     [seq_len, K] target top-k tokens/logprobs (renormalized in
                   the loss), -1 padding

Rows need one trailing context token after the block for the predecessor
machinery (masked 0).

Usage: python drafter/opd_build_replay.py data/opd_boundaries/<arm> \
          --out data/opd_replay_<arm>
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("boundaries", help="opd_collect out dir with anchors.jsonl + wave outputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gamma", type=float, default=0.8,
                    help="rejected-suffix decay (paper: 0.8, robust 0.5-0.8)")
    args = ap.parse_args()

    d = Path(args.boundaries)
    # request id -> full stream (prompt + accepted trajectory)
    streams = {}
    for of in sorted(d.glob("wave*_outputs.jsonl")):
        for line in open(of):
            r = json.loads(line)
            streams[r["id"]] = r["prompt_token_ids"] + r["output_token_ids"]
    anchors = [json.loads(x) for x in open(d / "anchors.jsonl")]

    K = max(len(a["p_top"][0]) if a.get("p_top") else 0 for a in anchors)
    prompt_lens = {}
    for of in sorted(d.glob("wave*_outputs.jsonl")):
        for line in open(of):
            r = json.loads(line)
            prompt_lens[r["id"]] = len(r["prompt_token_ids"])

    n_rows = n_acc = n_rej = 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "replay.jsonl", "w") as f:
        for a in anchors:
            rid = a["id"]
            if rid not in streams:
                continue
            stream = streams[rid]
            base = prompt_lens[rid] + a["stream_offset"]
            block = a["block_tokens"]
            acc = a["accepted_len"]
            n = len(block)
            # trailing context token (masked) for predecessor machinery
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
                if k <= acc:  # accepted positions (incl. anchor token match)
                    kl_mode[pos] = 1
                    target_ids[pos] = stream[base + k]
                    pos_weight[pos] = 1.0
                else:  # rejected suffix: reverse-KL vs stored target top-k
                    kl_mode[pos] = 2
                    pos_weight[pos] = args.gamma ** (k - acc - 1)
                if a.get("p_top") and k < len(a["p_top"]):
                    pairs = a["p_top"][k][:K]
                    for j, (t, lp) in enumerate(pairs):
                        p_tok[pos, j] = t
                        p_lp[pos, j] = lp
            n_acc += acc + 1
            n_rej += n - acc - 1
            f.write(json.dumps({
                "id": f"{rid}@{a['stream_offset']}",
                "input_ids": [int(x) for x in input_ids],
                "loss_mask": [1 if base <= i < base + n else 0 for i in range(L)],
                "opd_kl_mode": kl_mode.tolist(),
                "opd_target_ids": target_ids.tolist(),
                "opd_pos_weight": pos_weight.tolist(),
                "opd_p_topk_tokens": p_tok.tolist(),
                "opd_p_topk_logprobs": p_lp.tolist(),
            }) + "\n")
    print(f"[opd_build_replay] {n_rows} anchor rows -> {out}/replay.jsonl "
          f"(accepted {n_acc}, rejected-suffix {n_rej} positions, K={K})")


if __name__ == "__main__":
    main()
