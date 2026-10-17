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

# B200 (2026-10-17, 35B first attempt): cuDNN SDPA allocates per-shape
# workspace; at >99% HBM the allocation fails inside the calibration forward
# as CUDNN_STATUS_INTERNAL_ERROR — an OOM wearing a cuDNN costume. The
# flash/efficient/math backends need no such workspace. Mechanics-only fix;
# recipe untouched.
torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import model_class, resolve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

from datasets import Dataset, load_dataset  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from llmcompressor import oneshot  # noqa: E402
from llmcompressor.modeling.moe.linearize import load_quantizable_moe  # noqa: E402
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


def build_recipe(algo, moe=False):
    # vision tower and lm_head stay BF16
    ignore = ["lm_head", "re:.*visual.*"]
    # MoE-family (qwen3_5_moe) additions — verified empirically against
    # llmcompressor 0.14.1a (mini e2e): the router needs NO ignore entry
    # (Qwen3_5MoeTopKRouter is a raw nn.Parameter, never matched by
    # targets="Linear"; llmcompressor still materializes it into the
    # artifact's explicit ignore list automatically). Kept at BF16:
    # shared expert + its [1,2048] gate (always-active dense path; Intel
    # AutoRound MoE recipe + palmfuture GPTQ precedent, ~180MB) and GDN
    # in_proj_a/b ([32,2048]; vLLM fuses a+b into in_proj_ba — sub-kernel
    # width; the library duo recipe ignores the same two).
    moe_ignore = [
        "re:.*shared_expert.*",
        "re:.*linear_attn.in_proj_a$",
        "re:.*linear_attn.in_proj_b$",
    ]
    ignore = ignore + (moe_ignore if moe else [])
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
            GPTQModifier(targets="Linear", scheme="W4A16",
                         ignore=ignore + (moe_ignore if moe else [])),
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


def _base_snapshot_safetensors(hub_id: str):
    """Base-checkpoint shard paths (hub snapshot dir or a local model dir)."""
    from huggingface_hub import snapshot_download

    base = Path(hub_id) if Path(hub_id).exists() else Path(
        snapshot_download(hub_id, local_files_only=True)
    )
    return sorted(base.glob("*.safetensors"))


def _evict_file_cache(paths) -> int:
    """posix_fadvise(DONTNEED) each file so the kernel drops its clean pages.
    Returns total bytes evicted. No root needed; only clean cache is dropped."""
    import ctypes
    import os as _os

    POSIX_FADV_DONTNEED = 4
    libc = ctypes.CDLL(None, use_errno=True)
    total = 0
    for p in paths:
        fd = _os.open(str(p), _os.O_RDONLY)
        try:
            libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
            total += _os.fstat(fd).st_size
        finally:
            _os.close(fd)
    return total


def _start_cache_reclaimer(gap_gb=4.0, interval_s=5.0):
    """Daemon that keeps the CUDA caching allocator honest on GB10 unified
    memory. During transformers-5.x MoE loads, one large transient (~29GB
    observed on 35B) is allocated and freed but stays RESERVED (non-
    releasable: outstanding stream refs) — reserved-but-unallocated memory
    still counts against MemAvailable, so the load dies at ~50% with a
    perfectly healthy live set. synchronize() + empty_cache() returns
    inactive blocks to the system (2026-10-17 snapshot probe: reserved
    65.4GB vs live 36.4GB, single 29.7GB inactive block)."""
    import threading

    stop = threading.Event()
    reclaimed = [0.0]

    def loop():
        while not stop.wait(interval_s):
            try:
                gap = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                if gap > gap_gb * 2**30:
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_reserved()
                    torch.cuda.empty_cache()
                    got = (before - torch.cuda.memory_reserved()) / 2**30
                    reclaimed[0] += got
                    with open("/proc/self/statm") as f:
                        rss = int(f.read().split()[1]) * 4096 / 2**30
                    print(f"[reclaimer] gap {gap / 2**30:.1f}GB -> "
                          f"released {got:.1f}GB "
                          f"(reserved {torch.cuda.memory_reserved() / 2**30:.1f}GB "
                          f"alloc {torch.cuda.memory_allocated() / 2**30:.1f}GB "
                          f"rss {rss:.1f}GB)", flush=True)
            except Exception as e:
                print(f"[reclaimer] error: {e!r}", flush=True)

    threading.Thread(target=loop, daemon=True, name="cuda-cache-reclaimer").start()
    return stop, reclaimed


def _kill_uma_warmup():
    """No-op transformers' caching_allocator_warmup (modeling_utils.py:4418,
    defined ~5106). It pre-allocates a full-model-sized buffer (65.4GiB at
    35B: byte_count//2 as fp16) purely to defragment the CUDA cache, then
    frees it. On discrete GPUs that's free; on GB10 unified memory the freed
    block stays cache-reserved as real RAM — every big-model load died at
    43-55% with exactly this much memory missing (runs 1-7, 2026-10-17).
    Upstream already skips warmup on MPS/Neuron; same reasoning here."""
    import transformers.modeling_utils as _mu
    _mu.caching_allocator_warmup = lambda *a, **k: None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="4b",
                        help="model key from models.py (0.8b|4b|9b|27b|35b) or "
                             "a full hub id / local path")
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
    from lablog import log_event
    log_event("RUN_START", model=args.model, algo=args.algo, calib=args.calib)

    hub_id, info = resolve(args.model)
    base_name = hub_id.split("/")[-1]
    save_dir = f"{base_name}-W4A16-{args.algo.upper()}"
    # naming rule: quants are named for WHAT THEY ARE (model + recipe), never
    # for the calib experiment that produced them — calib provenance lives in
    # run logs and data/MANIFEST.sha256, not the dir name (retired 2026-10-16:
    # the old '-cascade' auto-suffix)
    if args.tag:
        save_dir += f"-{args.tag}"

    model_cls = model_class(hub_id)
    is_moe = model_cls.__name__ == "Qwen3_5MoeForConditionalGeneration"

    reclaim_stop, reclaimed = _start_cache_reclaimer()
    _kill_uma_warmup()
    if is_moe:
        # MoE: linearize fused experts at LOAD time — this also registers the
        # save-side repack mapping (the post-load fallback inside oneshot does
        # not). oneshot then routes calibration through all experts by default
        # (moe_calibrate_all_experts=True) and re-copies mtp.* tensors from the
        # base checkpoint on save.
        with load_quantizable_moe(model_cls):
            model = model_cls.from_pretrained(
                hub_id, dtype="auto", device_map={"": 0}
            )
    else:
        model = model_cls.from_pretrained(hub_id, dtype="auto", device_map={"": 0})
    reclaim_stop.set()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    log_event("CACHE_RECLAIMED", gb=round(reclaimed[0], 1))

    # GB10 unified memory: the just-read base shards (~67GB for 35b) sit in
    # page cache NEXT TO the resident model copy — ~2x the model footprint
    # before any workspace, and pressure-driven reclaim has already once
    # arrived as a global OOM that shot random userspace (2026-10-17).
    # Weights are copied onto the device (the mappings are closed: RSS is
    # tiny), so dropping the cache is safe. load happens via mmap -> device
    # copy; a re-fault would only mean a slow re-read, never corruption.
    evicted = _evict_file_cache(_base_snapshot_safetensors(hub_id))
    log_event("PAGE_CACHE_EVICT", files_gb=round(evicted / 2**30, 1))

    processor = AutoProcessor.from_pretrained(hub_id)

    oneshot_kwargs = dict(
        model=model,
        dataset=get_calibration_dataset(
            processor, args.calib, args.num_samples, args.max_seq_length
        ),
        recipe=build_recipe(args.algo, moe=is_moe),
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_samples,
    )
    if args.algo == "autoround":
        # per llm-compressor example: slightly better accuracy unshuffled
        oneshot_kwargs["shuffle_calibration_samples"] = False

    oneshot(**oneshot_kwargs)

    model.save_pretrained(save_dir, save_compressed=True)
    processor.save_pretrained(save_dir)
    # stamp canonical serving params into the artifact so `vllm serve` picks
    # them up as request defaults (models.SERVING_PARAMS = single source)
    import models as _m
    gc = {"do_sample": True, **_m.SERVING_PARAMS,
          "eos_token_id": processor.tokenizer.eos_token_id,
          "pad_token_id": processor.tokenizer.eos_token_id}
    import json as _json
    with open(f"{save_dir}/generation_config.json", "w") as f:
        json.dump(gc, f, indent=2)
    print(f"saved to {save_dir} (serving params stamped: {_m.SERVING_PARAMS})")

    if is_moe:
        # MTP audit: transformers drops mtp.* on load; llmcompressor's
        # save_mtp_tensors_to_checkpoint re-copies them from the base. Verify
        # the artifact actually carries them (spec-decode baselines need it).
        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        def mtp_keys(path):
            p = Path(path)
            idx = p / "model.safetensors.index.json"
            shards = (
                {(p / s).resolve() for s in json.loads(idx.read_text())["weight_map"].values()}
                if idx.exists()
                else sorted(p.glob("*.safetensors"))
            )
            keys = set()
            for s in shards:
                with safe_open(s, framework="pt") as f:
                    keys.update(k for k in f.keys() if k.startswith("mtp."))
            return keys

        base_path = hub_id if Path(hub_id).exists() else snapshot_download(hub_id)
        base_keys = mtp_keys(base_path)
        saved_keys = mtp_keys(save_dir)
        missing = base_keys - saved_keys
        log_event("MTP_AUDIT", base=len(base_keys), saved=len(saved_keys),
                  missing=len(missing))
        # registry says this model has MTP → the copy MUST have happened;
        # unknown provenance (raw hub id/path) → require at least parity
        # with whatever the base carries
        required = bool(info.get("mtp_num_hidden_layers"))
        if (required and (not base_keys or saved_keys != base_keys)) or (
            not required and missing
        ):
            raise RuntimeError(
                f"MTP tensors wrong in artifact (base {len(base_keys)}, saved "
                f"{len(saved_keys)}, missing {len(missing)}); spec-decode "
                f"baselines would break")
        print(f"MTP audit: {len(saved_keys)}/{len(base_keys)} mtp.* tensors copied")

    from lablog import run_done
    run_done(save_dir=save_dir)
    if info.get("quant_repo"):
        print(f"(published reference artifact: {info['quant_repo']})")


if __name__ == "__main__":
    main()
