"""W4A16 quantization for the Qwen 3.5 release family using llm-compressor.

Champion recipe (what we published): plain GPTQ, cascade-v1 calibration,
999 samples x 4096 tokens. Algorithm/precision rationale lives in the README
(warp-3090: 24GB Ampere budget).

Usage:
  python quant/quantize.py --model 4b                      # champion cascade quant
  python quant/quantize.py --model 27b --algo gptq
  python quant/quantize.py --model 4b --algo awq --tag awq-cascade
  python quant/quantize.py --model 4b --calib data/calib-cascade-v1.jsonl --num-samples 512

Calibration: --calib takes a jsonl path (default: the frozen cascade-v1 set in
data/) or a comparison-set name (cyankiwi | hermes | ultrachat, fetched from HF).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import resolve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

from datasets import Dataset, load_dataset  # noqa: E402
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration  # noqa: E402

from llmcompressor import oneshot  # noqa: E402
from llmcompressor.modifiers.autoround import AutoRoundModifier  # noqa: E402
from llmcompressor.modifiers.gptq import GPTQModifier  # noqa: E402
from llmcompressor.modifiers.quantization import QuantizationModifier  # noqa: E402
from llmcompressor.modifiers.transform.awq import AWQModifier  # noqa: E402

MAX_SEQUENCE_LENGTH = 4096


def normalize_messages(conversation):
    # flatten OpenAI-style tool_calls; Qwen3.5 template wants dict arguments
    messages = []
    for message in conversation:
        msg = {"role": message["role"], "content": message.get("content") or ""}
        if message.get("tool_calls"):
            tool_calls = []
            for call in message["tool_calls"]:
                call = dict(call.get("function") or {})
                args = call.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                call["arguments"] = args
                tool_calls.append(call)
            msg["tool_calls"] = tool_calls
        messages.append(msg)
    return messages


def load_jsonl_calib(path, num_samples):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            msgs = r.get("messages") or r.get("conversations")
            rows.append(normalize_messages(msgs))
    return rows[:num_samples]


def load_cyankiwi(num_samples):
    ds = load_dataset("cyankiwi/calibration", split="train")
    return [normalize_messages(c) for c in ds["conversations"][:num_samples]]


def load_hermes(num_samples):
    # code-agent blend: real API tool calls, multi-turn function calling,
    # and agentic structured output
    shares = {"func_calling": 0.4, "glaive_func_calling": 0.4, "json_mode_agentic": 0.2}
    role_map = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool"}

    parts = []
    for name, share in shares.items():
        ds = load_dataset(
            "NousResearch/Hermes-Function-Calling-V1", name, split="train"
        ).shuffle(seed=42)
        msgs = [
            [{"role": role_map[m["from"]], "content": m["value"]} for m in conv]
            for conv in ds["conversations"]
        ]
        parts.append(msgs[: int(num_samples * share)])

    combined = [m for part in parts for m in part]
    return combined[:num_samples]


def load_ultrachat(num_samples):
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split="train_sft"
    ).shuffle(seed=42)
    return [m for m in ds["messages"][:num_samples]]


NAMED_CALIBRATIONS = {
    "cyankiwi": load_cyankiwi,
    "hermes": load_hermes,
    "ultrachat": load_ultrachat,
}


def get_calibration_dataset(processor, calib, num_samples, max_seq_length):
    tokenizer = processor.tokenizer
    if calib in NAMED_CALIBRATIONS:
        conversations = NAMED_CALIBRATIONS[calib](num_samples)
    else:
        path = Path(calib)
        if not path.is_absolute():
            path = ROOT / path
        conversations = load_jsonl_calib(path, num_samples)

    def tokenize(sample):
        try:
            text = processor.apply_chat_template(
                normalize_messages(sample["messages"]), tokenize=False
            )
        except Exception:
            return {"input_ids": [], "attention_mask": []}
        return tokenizer(
            text,
            padding=False,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = Dataset.from_list([{"messages": m} for m in conversations])
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["input_ids"]) > 0)
    print(f"calibration samples ({calib}): {len(ds)}")
    return ds


def build_recipe(algo):
    # vision tower and lm_head stay BF16
    ignore = ["lm_head", "re:.*visual.*"]
    if algo == "gptq":
        return GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore)
    if algo == "duo":
        # the library's canonical recipe verbatim (examples/quantization_w4a16/
        # qwen3_8_gptq_awq_example.py): AWQ pre-scaling + GPTQ, with the
        # library's own ignore list (keeps linear_attn in_proj_a/b at BF16).
        # NOTE: AWQ calibration caches scale linearly with tokens — at 999x4096
        # even 275GB VRAM OOMs; offload to CPU is protocol-safe. We measured
        # duo at library budget (512x384) LOSES to plain GPTQ at 999x4096
        # (weightedKL 0.1209 vs 0.1087; agentic domains worst). Kept for
        # reproduction only.
        ignore = [
            "re:visual.*", "re:model.visual.*", r"re:.*lm_head",
            r"re:.*embed_tokens$", r"re:.*linear_attn\.in_proj_a$",
            r"re:.*linear_attn\.in_proj_b$",
        ]
        return [
            AWQModifier(duo_scaling="both", offload_device=torch.device("cpu")),
            GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore),
        ]
    if algo == "awq":
        # AWQModifier scales salient channels; QuantizationModifier quantizes
        return [
            AWQModifier(),
            QuantizationModifier(targets="Linear", scheme="W4A16", ignore=ignore),
        ]
    if algo == "autoround":
        # sign-gradient optimization of rounding offsets and clipping ranges
        return AutoRoundModifier(
            targets="Linear", scheme="W4A16", ignore=ignore, iters=200
        )
    raise ValueError(f"unknown algo: {algo}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="4b",
                        help="model key from models.py (4b|9b|27b) or a full "
                             "hub id / local path")
    parser.add_argument("--algo", choices=["gptq", "duo", "awq", "autoround"],
                        default="gptq")
    parser.add_argument("--calib", default="data/calib-cascade-v1.jsonl",
                        help="calibration jsonl path or comparison-set name "
                             "(cyankiwi|hermes|ultrachat)")
    parser.add_argument("--num-samples", type=int, default=999,
                        help="champion regime is 999; library default is 512")
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQUENCE_LENGTH,
                        help="calibration truncation length (library default "
                             "regime is 384)")
    parser.add_argument("--tag", default=None, help="output dir suffix")
    args = parser.parse_args()

    hub_id, info = resolve(args.model)
    base_name = hub_id.split("/")[-1]
    save_dir = f"{base_name}-W4A16-{args.algo.upper()}"
    if args.calib.endswith(".jsonl") and "cascade" in Path(args.calib).stem:
        save_dir += "-cascade"
    if args.tag:
        save_dir += f"-{args.tag}"

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        hub_id, dtype="auto", device_map={"": 0}
    )
    processor = AutoProcessor.from_pretrained(hub_id)

    oneshot_kwargs = dict(
        model=model,
        dataset=get_calibration_dataset(
            processor, args.calib, args.num_samples, args.max_seq_length
        ),
        recipe=build_recipe(args.algo),
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_samples,
    )
    if args.algo == "autoround":
        # per llm-compressor example: slightly better accuracy unshuffled
        oneshot_kwargs["shuffle_calibration_samples"] = False

    oneshot(**oneshot_kwargs)

    model.save_pretrained(save_dir, save_compressed=True)
    processor.save_pretrained(save_dir)
    print(f"saved to {save_dir}")
    if info.get("quant_repo"):
        print(f"(published reference artifact: {info['quant_repo']})")


if __name__ == "__main__":
    main()
