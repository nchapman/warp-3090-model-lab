"""Hidden-state extraction — pipelined: GPU-side fp8 cast + overlapped writers.

Hardware notes (README "Hardware"): runs anywhere the quant loads via
transformers (train on Blackwell, deploy on Ampere — states are
hardware-portable modulo bf16 GEMM tiling noise, measured cos >= 0.9998).

CRITICAL RULES (paid-for, see experiments/drafters/):
  1. transformers-only forward. vLLM extract_hidden_states on W4A16 quants
     produces NaN tails AND silently-wrong states — never use it on a quant.
  2. States must be N+1 taps: aux layers + the FINAL layer
     (e.g. --model 4b gives [2,10,17,24,29,32] = 5 aux + final). The trainer
     splits hs[:, :-1] -> aux input, hs[:, -1] -> verifier_last stream.
  3. Run with --parity on a small sample before any bulk job.

Usage (two shards per GPU for overlap; sharded across GPUs):
    EXTRACT_SHARD=0 EXTRACT_N=4 CUDA_VISIBLE_DEVICES=0 \
      python drafters/extract_states.py --dataset ds_v4 --model 4b --out ds_v4 \
      [--resume] [--parity 20]
"""

import argparse
import concurrent.futures
import json
import os
import struct
import sys
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import resolve  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--verifier", required=True,
                   help="quant dir/hub id — states come FROM the deployed quant")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=None,
                   help="model key — fills --layers from models.py taps_v4")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="explicit tap list (aux + final, N+1 entries)")
    p.add_argument("--dtype", choices=["fp8", "bf16"], default="fp8",
                   help="state storage; fp8 is 0.5x disk, parity-proven "
                        "(cos >= 0.9997; needs drafters/patches/speculators_fp8_load_cast.patch)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-batch-tokens", type=int, default=16384)
    p.add_argument("--max-batch-rows", type=int, default=16,
                   help="row cap per forward batch — raise for better GPU "
                        "utilization (data-scaling ladder benchmark)")
    p.add_argument("--writers", type=int, default=4)
    p.add_argument("--resume", action="store_true",
                   help="skip rows whose existing file already has the right tap count")
    p.add_argument("--parity", type=int, default=0,
                   help="verify N written files decode with matching token_ids/shapes")
    return p.parse_args()


def tap_count_ok(path, want):
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        s = h.get("hidden_states", {}).get("shape", [])
        return len(s) == 3 and s[1] == want
    except Exception:
        return False


def main():
    args = parse_args()
    if args.layers is None:
        if not args.model:
            sys.exit("pass --model (taps from models.py) or --layers explicitly")
        _, info = resolve(args.model)
        args.layers = info["taps_v4"]
    print(f"taps: {args.layers} ({len(args.layers) - 1} aux + final)")

    shard = int(os.environ.get("EXTRACT_SHARD", "0"))
    n_shards = int(os.environ.get("EXTRACT_N", "1"))
    hs_dir = os.path.join(args.out, "hidden_states")
    os.makedirs(hs_dir, exist_ok=True)

    from transformers import Qwen3_5ForConditionalGeneration
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.verifier, dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()

    ds = load_from_disk(args.dataset)
    idxs = list(range(shard, len(ds), n_shards))
    if args.limit:
        idxs = idxs[:args.limit]
    if args.resume:
        idxs = [i for i in idxs
                if not tap_count_ok(f"{hs_dir}/hs_{i}.safetensors", len(args.layers))]
    rows = sorted(((i, torch.as_tensor([int(x) for x in ds[i]["input_ids"]],
                                       dtype=torch.long)) for i in idxs),
                  key=lambda t: t[1].numel())
    print(f"shard {shard}/{n_shards}: {len(rows)} rows", flush=True)

    pool = concurrent.futures.ThreadPoolExecutor(args.writers)
    pending = []
    store_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16

    def write_row(i, tok_cpu, hs_cpu):
        save_file({"token_ids": tok_cpu, "hidden_states": hs_cpu},
                  f"{hs_dir}/hs_{i}.safetensors")

    b_ids, b_meta = [], []
    n_done, t0 = 0, time.time()

    def flush_blocking():
        nonlocal n_done
        if not b_ids:
            return
        maxlen = max(t.numel() for t in b_ids)
        pad = torch.full((len(b_ids), maxlen), 1, dtype=torch.long)
        mask = torch.zeros((len(b_ids), maxlen), dtype=torch.long)
        for k, t in enumerate(b_ids):
            pad[k, : t.numel()] = t
            mask[k, : t.numel()] = 1
        with torch.no_grad():
            out = model(pad.cuda(), attention_mask=mask.cuda(),
                        output_hidden_states=True, logits_to_keep=1)
        # (B, L, T, H) stacked on GPU; fp8 cast HERE, not on the CPU tail
        hs = torch.stack([out.hidden_states[j] for j in args.layers], dim=2)
        hs = hs.to(store_dtype).cpu()
        futures = [pool.submit(write_row, i, t.cpu(),
                               hs[k, : t.numel()].clone())
                   for k, (i, t) in enumerate(b_meta)]
        pending.extend(futures)
        # backpressure: never let more than 3 batches of writes queue
        while len(pending) > 3 * args.writers:
            done = [f for f in pending if f.done()]
            for f in done:
                f.result()  # raise on failure
                pending.remove(f)
                n_done += 1
        b_ids.clear(); b_meta.clear()
        if n_done % 1000 < len(futures):
            rate = n_done / (time.time() - t0)
            print(f"shard {shard}: {n_done}/{len(rows)} ({rate:.1f} rows/s)", flush=True)

    for i, t in rows:
        b_ids.append(t); b_meta.append((i, t))
        if sum(x.numel() for x in b_ids) > args.max_batch_tokens or len(b_ids) >= args.max_batch_rows:
            flush_blocking()
    flush_blocking()
    for f in pending:
        f.result()
        n_done += 1

    if args.parity:
        bad = 0
        for i, t in rows[: args.parity]:
            p = f"{hs_dir}/hs_{i}.safetensors"
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            want = (t.numel(), len(args.layers), h["hidden_states"]["shape"][2])
            if tuple(h["hidden_states"]["shape"]) != want:
                bad += 1
        print(f"parity: {args.parity - bad}/{args.parity} ok")
    rate = n_done / (time.time() - t0)
    print(f"shard {shard} DONE {n_done} rows at {rate:.1f} rows/s")


if __name__ == "__main__":
    main()
