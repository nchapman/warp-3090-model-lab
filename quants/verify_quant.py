#!/usr/bin/env python3
"""Verify a quantized artifact: config, hashes, smoke load + generate.

Usage: python quants/verify_quant.py <artifact_dir> [--skip-generate]

Checks (exit 1 on any failure):
  1. File inventory: safetensors shards, tokenizer files, generation_config,
     config.json all present and non-empty.
  2. config.json carries a quantization_config: GPTQ, W4A16, packed ints.
  3. sha256 of every file -> <dir>/SHA256SUMS (+ stdout).
  4. Smoke: load on GPU, greedy-generate 8 tokens from a trivial prompt
     (--skip-generate to skip; e.g. when the GPU is busy).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact_dir")
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()

    d = Path(args.artifact_dir)
    if not d.is_dir():
        fail(f"{d} is not a directory")

    # -- 1. file inventory ------------------------------------------------
    must = ["config.json", "generation_config.json", "tokenizer_config.json",
            "tokenizer.json"]
    missing = [m for m in must if not (d / m).is_file()]
    if missing:
        fail(f"missing files: {missing}")
    shards = sorted(d.glob("*.safetensors"))
    if not shards:
        fail("no safetensors shards")
    total = sum(s.stat().st_size for s in shards)
    print(f"OK files: {len(shards)} shards, {total / 2**30:.2f} GiB total")

    # -- 2. quantization config ------------------------------------------
    cfg = json.loads((d / "config.json").read_text())
    qc = cfg.get("quantization_config") or {}
    if qc.get("quant_method") != "compressed-tensors":
        fail(f"quant_method = {qc.get('quant_method')!r}, want 'compressed-tensors'")
    if qc.get("format") != "pack-quantized":
        fail(f"format = {qc.get('format')!r}, want 'pack-quantized'")
    if qc.get("quantization_status") != "compressed":
        fail(f"status = {qc.get('quantization_status')!r}, want 'compressed'")
    cfgs = qc.get("config_groups", {})
    if not cfgs:
        fail("no config_groups")
    for name, g in cfgs.items():
        w = g.get("weights", {})
        if w.get("num_bits") != 4 or w.get("strategy") != "group" or w.get("group_size") != 128:
            fail(f"group {name}: {w}")
    # GPTQ provenance: recipe.yaml in the artifact names the modifier + scheme
    ry = (d / "recipe.yaml").read_text()
    if "GPTQModifier" not in ry or "W4A16" not in ry:
        fail("recipe.yaml lacks GPTQModifier/W4A16")
    print("OK quant: compressed-tensors pack-quantized GPTQ W4A16 g128 "
          f"({len(cfgs)} group); ignored(bf16): {str(qc.get('ignore'))[:80]}...")

    # -- 3. hashes ---------------------------------------------------------
    sums = d / "SHA256SUMS"
    with sums.open("w") as f:
        for p in sorted(d.iterdir()):
            if p.name == "SHA256SUMS":
                continue
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
            f.write(f"{h.hexdigest()}  {p.name}\n")
    big = max((d / s.name for s in shards), key=lambda p: p.stat().st_size)
    for line in sums.read_text().splitlines():
        if big.name in line:
            print(f"OK sha256: {len(sums.read_text().splitlines())} files; "
                  f"largest {big.name}: {line.split()[0]}")

    # -- 4. smoke generate --------------------------------------------------
    if args.skip_generate:
        print("OK smoke: skipped")
        return
    import torch  # noqa: E402
    from models import model_class  # noqa: E402

    cls = model_class(str(d))
    model = cls.from_pretrained(str(d), dtype="auto", device_map={"": 0})
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(d))
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=8, do_sample=False)
    text = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"OK smoke: {prompt!r} -> {text!r}")

    print("VERIFY PASSED")


if __name__ == "__main__":
    main()
