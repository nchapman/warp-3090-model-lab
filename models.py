"""Qwen 3.5 release family registry — the single source of truth for scripts.

All checkpoints across the 3.5 / 3.6-upgrade / 3.8-only releases share the same
architecture class (model_type qwen3_5, Qwen3_5ForConditionalGeneration), so one
code path covers the family; per-model facts live here.

Cross-generation caveat: absolute KL / acceptance numbers are NOT comparable
across release generations (measured: Qwen3.8-27B shows a higher conv_agent KL
wall than Qwen3.5-4B despite more scale). Keep results tables per-model.
"""

MODELS = {
    "4b": {
        "hub": "Qwen/Qwen3.5-4B",
        "generation": "3.5",
        "num_hidden_layers": 32,
        "hidden_size": 2560,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 9216,
        "vocab_size": 248320,
        "mask_token_id": 248077,          # auto-detected from tokenizer (len 248078)
        # drafter taps: N aux + final layer (states are always N+1 taps!)
        "taps_v1": [2, 16, 29, 32],               # 3 aux, block-8 era
        "taps_v4": [2, 10, 17, 24, 29, 32],       # 5 aux + final, champion era
        "draft_config": "drafter/draft_config_4b.json",
        # published artifacts (live on HF)
        "quant_repo": "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        # validated results (see RESULTS.md; blank = untested)
        "quant_weightedkl": 0.1087,
        "drafter_ladder": 3.443,   # armB-ep2, b16-native, interval-weighted
    },
    "9b": {
        "hub": "Qwen/Qwen3.5-9B",
        "generation": "3.5",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 12288,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "taps_v1": [2, 16, 29, 32],               # same depth as 4B
        "taps_v4": [2, 10, 17, 24, 29, 32],
        "draft_config": "drafter/draft_config_9b.json",
        "quant_repo": "nchapman/Qwen3.5-9B-W4A16-GPTQ",
        "quant_weightedkl": 0.0889,
        "drafter_ladder": None,    # init config built, never trained
    },
    "27b": {
        "hub": "Qwen/Qwen3.8-27B",
        "generation": "3.8",
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        # even-spread default rule [2, L/2, L-3]; 5-aux spread for the v4-style
        # drafter. UNVALIDATED for us — vendor premium 27B drafter used
        # [5,19,33,47,61] (block 8). Treat 27B drafter as experimental.
        "taps_v1": [2, 32, 61, 64],
        "taps_v4": [2, 17, 32, 47, 61, 64],
        "draft_config": "drafter/draft_config_27b.json",
        "quant_repo": "nchapman/Qwen3.8-27B-W4A16-GPTQ",
        "quant_weightedkl": 0.1264,   # cross-gen: not comparable to 3.5 numbers
        "drafter_ladder": None,       # NO-GO decision (budget); premium ref 2.959
    },
}


def get(key):
    """Resolve a model key ('4b') or pass through a full hub id/path."""
    if key in MODELS:
        return MODELS[key]
    return None


def resolve(key):
    """Return (hub_id, info) where info is the registry entry or {} for raw ids."""
    info = get(key)
    if info is not None:
        return info["hub"], info
    return key, {}
