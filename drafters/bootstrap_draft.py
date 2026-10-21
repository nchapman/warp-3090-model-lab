"""Bootstrap a fresh DFlash2 draft init (v4-exact convention).

Creates <out>: 5 plain qwen3 draft layers at the target's dims, speculators
config embedded, bf16-stripped checkpoint (verifier-owned weights omitted —
byte-convention identical to the production drafters). Requires the
speculators package (requirements-drafter.txt) — run in the drafter venv.

Usage:
  python drafters/bootstrap_draft.py --model 4b --quant ./Qwen3.5-4B-W4A16-GPTQ-cascade \
      --out draft_init
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import resolve  # noqa: E402

# verifier-owned tensors dropped from the init (regression-tested convention)
DROP = {"embed_tokens.weight", "lm_head.weight",
        "verifier_lm_head.weight", "verifier_norm.weight"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="4b")
    ap.add_argument("--quant", required=True,
                    help="quant dir/hub id — dims + tokenizer conventions come from it")
    ap.add_argument("--out", default="draft_init")
    ap.add_argument("--block-size", type=int, default=16,
                    help="16 = deployment config (serves down to 8 freely)")
    ap.add_argument("--taps", choices=["v1", "v4"], default="v4",
                    help="aux tap set from models.py (v4 = 5 aux, the champion era)")
    ap.add_argument("--num-layers", type=int, default=5)
    args = ap.parse_args()

    _, info = resolve(args.model)
    taps = info[f"taps_{args.taps}"][:-1]  # aux only for the draft config
    mask_token_id = info["mask_token_id"]
    vocab = info["vocab_size"]

    from speculators.models.dflash2.core import DFlash2DraftModel
    from speculators.train.cli import create_transformer_layer_config

    tl_cfg = create_transformer_layer_config(
        verifier_name_or_path=args.quant, num_layers=args.num_layers,
        draft_arch="qwen3", hidden_act=None, sliding_window=-1,
        full_attention_indices=[])
    m = DFlash2DraftModel.from_training_args(
        tl_cfg, verifier_name_or_path=args.quant, draft_vocab_size=vocab,
        block_size=args.block_size, target_layer_ids=taps,
        mask_token_id=mask_token_id)
    m.save_pretrained(args.out)

    # match production checkpoints: bf16, verifier-owned weights omitted
    sd = load_file(f"{args.out}/model.safetensors")
    kept = {k: v.to(torch.bfloat16) for k, v in sd.items() if k not in DROP}
    save_file(kept, f"{args.out}/model.safetensors")
    print(f"draft init saved: {args.out}")
    print(f"  {len(kept)} tensors, {sum(v.numel() for v in kept.values()) / 1e6:.1f}M params")
    print(f"  taps {taps} (states must be {len(taps) + 1} taps incl. final layer!)")
    print(f"  block {args.block_size} ({args.block_size - 1} spec tokens)")


if __name__ == "__main__":
    main()
