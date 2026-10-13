"""Teacher-forced regeneration: rewrite assistant turns with the DEPLOYED QUANT.

Quant-alignment doctrine: the drafter must hit the exact deployed model, so
regen, states, training, and eval all run against the quant — never the bf16
base. Serve it first:  vllm serve <quant> --port 8138

Two steps, one script (both resume-safe):
  1. regen: hit the vLLM REST endpoint for every slice row (skips done ids)
  2. merge: write full conversations (ctx + regenerated assistant turn) for
     prep_dataset.py — one rendering path end-to-end, the v2+ convention.

Usage:
  python drafter/regen.py slices/v4_train.jsonl slices/v4_regen.jsonl \
      slices/v4_merged.jsonl --model 4b
  # endpoint/concurrency via REGEN_ENDPOINT / REGEN_CONC env (defaults below)
"""

import argparse
import concurrent.futures
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import resolve  # noqa: E402

# v1 conventions: temp 0.7 / top_p 0.95 / seed 42, max_tokens 640
DEFAULTS = {"temperature": 0.7, "top_p": 0.95, "seed": 42, "max_tokens": 640}


def regen(rows, out_path, endpoint, conc):
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["id"] for l in open(out_path)}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} rows, {len(done)} already done, {len(todo)} to go", flush=True)

    def hit(r):
        body = json.dumps({
            "messages": r["messages"], **DEFAULTS,
        }).encode()
        req = urllib.request.Request(endpoint, body,
                                     {"Content-Type": "application/json"})
        err = None
        for _ in range(4):
            try:
                with urllib.request.urlopen(req, timeout=900) as resp:
                    out = json.load(resp)
                txt = out["choices"][0]["message"].get("content") or ""
                if txt:
                    return r["id"], txt, None
                return r["id"], None, "empty"
            except Exception as e:
                err = str(e)[:200]
        return r["id"], None, err

    n_ok = n_err = 0
    with open(out_path, "a") as f, open(out_path + ".errors.jsonl", "a") as ef:
        with concurrent.futures.ThreadPoolExecutor(conc) as ex:
            for i, (rid, txt, err) in enumerate(ex.map(hit, todo)):
                if txt:
                    f.write(json.dumps({"id": rid, "response": txt}) + "\n")
                    n_ok += 1
                else:
                    ef.write(json.dumps({"id": rid, "error": err}) + "\n")
                    n_err += 1
                if (i + 1) % 100 == 0:
                    print(f"{i+1}/{len(todo)} ok={n_ok} err={n_err}", flush=True)
    print(f"regen DONE ok={n_ok} err={n_err}", flush=True)


def merge(rows, regen_path, out_path):
    ctxs = {r["id"]: r["messages"] for r in rows}
    n_ok = n_skip = 0
    with open(out_path, "w") as f:
        for line in open(regen_path):
            r = json.loads(line)
            ctx = ctxs.get(r["id"])
            if ctx is None:
                n_skip += 1  # stale row from a previous slice
                continue
            f.write(json.dumps({
                "id": r["id"],
                "messages": [*ctx, {"role": "assistant", "content": r["response"]}],
            }) + "\n")
            n_ok += 1
    print(f"merged {n_ok} conversations ({n_skip} stale skipped) -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slice", help="slice jsonl ({id, messages=ctx})")
    ap.add_argument("regen_out", help="{id, response} jsonl (resume log)")
    ap.add_argument("merged_out", help="full-conversations jsonl for prep_dataset.py")
    ap.add_argument("--model", default="4b")
    ap.add_argument("--quant", default=None,
                    help="optional: sanity-print the quant to serve (default from models.py)")
    ap.add_argument("--endpoint",
                    default=os.environ.get("REGEN_ENDPOINT",
                                           "http://localhost:8138/v1/chat/completions"))
    ap.add_argument("--conc", type=int,
                    default=int(os.environ.get("REGEN_CONC", "8")))
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    _, info = resolve(args.model)
    quant = args.quant or info.get("quant_repo")
    print(f"endpoint: {args.endpoint}")
    print(f"make sure the DEPLOYED QUANT is serving there (e.g. {quant})")

    rows = [json.loads(l) for l in open(args.slice)]
    if not args.merge_only:
        regen(rows, args.regen_out, args.endpoint, args.conc)
    merge(rows, args.regen_out, args.merged_out)


if __name__ == "__main__":
    main()
