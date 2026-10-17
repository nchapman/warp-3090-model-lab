# warp-3090-quant-lab — agent rules

Project-specific doctrine for any agent (or human) working in this repo. The
global engineering guidelines still apply; this file adds what this lab
learned the hard way.

## Runs and processes

- Every eval/server/training run goes through `lib/runlib.sh` (bash) or
  `quant/lablog.py` (python) — no exceptions. If a script runs anything
  GPU-bound or longer than a minute without `run_init`/events, it's wrong.
- Inspect any run with `bash scripts/tail_run.sh [prefix]` — never go hunting
  for log paths manually.
- Servers die by the PID recorded in the SERVER_LAUNCH event (`stop_server`).
  Never `pkill`/`pgrep -f` by pattern — it has killed our own shell FOUR
  times (pattern text appears in your own cmdline). Recorded PIDs only; if
  you must pattern-match, build the pattern from concatenated shell variables
  so your own cmdline doesn't contain it.
- Never edit a script while an instance of it is running.
- vLLM serving with speculative decoding can wedge silently during CUDA-graph
  capture — pass `EXTRA_SERVE_ARGS="--enforce-eager"` if a spec-decode server
  hangs at startup.

## Measuring τ (acceptance)

- `TEMP` env on `drafter/eval_ladder.sh` / `eval_mtp.sh` selects sampling
  temperature (default 1.0). For checkpoint selection, use
  `drafter/sweep_epochs.sh` — one command, full campaign. Three regimes,
  three purposes:
  - **t=0 (greedy)** — the low-noise instrument (repeat spread ~0.05). Use to
    sanity-check stability, not to pick shipping checkpoints.
  - **t=0.7 (serving)** — the deployment point and the corpus generation
  config. **Selection happens here**, n≥4 per contender (spread ~0.1).
  - **t=1.0** — heavy-tailed (0.34 spread observed on identical inputs).
  Never rank checkpoints on small n at t≥0.7; never quote a single draw as a
  result. Report n and ranges with every number.
- Ranking is temperature-dependent. State the temperature with every τ. A
  checkpoint that wins at greedy may lose at serving — that's data, not error.
- Fresh GB10 numbers are the lab's anchors (`RESULTS.md`). Don't publish
  cross-machine τ comparisons; don't resurrect superseded numbers.

## Publishing

- One recipe per artifact. No internal run names ("v4b", "armB", epoch
  counts) in public Hub text — the card describes the recipe and the shipped
  checkpoint, nothing else.
- **Naming rule (2026-10-16):** quants are named for what they ARE —
  `Qwen3.5-<size>-W4A16-GPTQ` — never for the calib experiment that
  produced them (the `-cascade` dir suffix is retired; historical dirs
  keep their names, nothing new gets one). Dataset files are named for
  their PURPOSE (`calib-*`, regen pool) with the source documented
  inside (`calib-cascade-v1` = nvidia/Nemotron-Cascade-2-SFT-Data —
  source-derived name on the dataset is fine). Calib provenance lives
  in run logs + `data/MANIFEST.sha256`, not quant dir names.
- Cards carry `model.safetensors` sha256 and full provenance
  (train_command, run.yaml, training_state, val_metrics).

## Environment

- Fresh pod: `scripts/pod_prep.sh <user@host> [port]` — survey, HF token,
  guarded rsync (refuses surprise gigabytes), frozen-data manifest check,
  venv build, speculators patches, GPU check, model pulls. Re-anchor any
  baseline on the pod itself before comparing numbers across machines.

- Two venvs: `.venv-quant` (torch 2.14.1 / llmcompressor), `.venv-dflash`
  (torch 2.13.0 / speculators 0.8.1 / vllm 0.31.0). `vllm serve` needs the
  venv's bin on PATH.
- Everything caches via standard `~/.cache/huggingface`. No repo-local model
  litter; transient outputs are disposable and stay out of git.
- Frozen data in `data/` is load-bearing for every published number —
  `sha256sum -c` before any measurement campaign.

## Corpus & capability facts (MEASURED — do not re-derive, do not assume)

**Data currency.** Always state amounts in **unique supervised tokens**
(loss_mask=1, once per conv), with conversations as the diversity co-unit.
SPEAK TOKENS, NOT CHARACTERS — never report text lengths in chars
(they read as token counts and mislead; the 2400-char ≈ 640-tok proxy
confusion is the canonical example). Nominal row-tokens ≠ unique ≠
supervised: rung 1's "10M" was 7.5M unique (25% prefix dup) = **5.2M
supervised** (1.93× exchange). `drafter/data_audit.py` measures all of
these — run it, never guess.

**Corpus** (`data/context_pool.jsonl`, sha-pinned; read-only copy of the
v4 regen input): 145,018 rows / ~100K convs / 260M nominal. **THIS FILE IS THE REGEN
INPUT, NOT A CORPUS** — it is slice_a+slice_b regrouped by conversation:
bare contexts, every row ends user/tool, ZERO regenerated responses.
The final assistant turn of every conversation is absent; single-turn
rows (51% of file — ALL math + science, plus 60K chat/IF/agent) carry no
supervision at all and are dropped by prep. The true corpus (contexts +
on-policy finals; what the champion trained on) was pod-side
/root/dflash_data_v4 and is GONE with the pod — only regen inputs and
the champion ckpt survive.
- Mix — the file's row labels (chat 65/math 12.5/science 8/…) describe
  PROMPT POOL composition, not training data. Ladder-side effective mix
  (supervised tokens in context turns): chat ~87% / agents ~11%; math 0,
  science 0 (single-turn → no supervision). Champion-side mix includes
  regenerated finals (rate unknown, node logs gone). Code ≈ 6% of
  context content (measured).
- **τ 1.05 (rung 1) is NOT a clean dead-zone number**: it trained
  off-policy (source responses only, zero on-policy finals) and scored
  BELOW the epoch-0 control (3.407) — distribution-mismatch signature,
  volume confounded with data-kind. Re-measure on regen-fixed corpus
  before quoting it.
- Length: mean 2,115 / median 1,676 / p90 4,588 / p99 6,207 tokens —
**uncapped**; DFlash paper spec caps at 3072. DSpark reference preps ≤1024.
- Shape tooling: `drafter/reshape_tokens.py` (prefix/merged/first/short),
`prep_dataset.py --loss-mode all|last --max-tokens N` (drop cap).

**Eval.** `data/ladder-20.jsonl` is GENERAL CHAT ONLY — every τ ever
quoted (incl. champion 3.79) is general-chat τ. No code/agentic/long
eval exists. Don't claim deployment relevance without one.

**Measured anchors (τ @ t=0.7 n=4 unless noted):** 4B 5.2M supervised →
1.05 flat (UNSOUND — off-policy artifact, see corpus section; do not
quote as a dead-zone number); 4B ~150M → **3.79**; epoch-0 control 3.407.
0.8B: 0.7M → 1.024, 3.5M → 1.08–1.09 (n=1) — dead zone extends to ≥3.5M at
0.8B; MTP k=15 untrained head = 2.868 on the same ladder (the bar a
trained 0.8B drafter must beat). External: DSpark 100K prompts ≤1024 →
3.41; DFlash 100K-sample ablations → 4.9–5.6 greedy. Dead zone < ~5M
supervised (two sources); known-good 20–60M; rising to ~1B (DFlare).

**Loss-weighting footgun:** speculators 0.8.1 defaults dflash to
`--per-position-loss-weight dpace` — the documented γ=7 static decay runs
only when pinned (all our scripts pin it; unpinned commands silently train
D-PACE weights). Draft-OPD trainer patch is inert without opd_ columns —
default SFT path unaffected (docs/draft_opd.md).

**Trainer behavior.** Multipack sampler re-shuffles per epoch (seed+epoch)
→ epoch step-counts vary (rung 1: 1,310–1,850+); `step_ms` excludes
inter-step overhead — wall is ~1.7–2× step_ms. Plan in hours/epoch
(GB10: ~3.5h per 10M nominal), never in step counts.

**GB10 reality.** Bandwidth-starved (~17.6× off H200): extraction and
training both ~2k tok/s class. nvidia-smi memory reads [N/A]; 30–70W
at 96% util is normal load, not idle. A hung nvidia-smi can hold
/dev/nvidia0 — kill by exact PID (never pkill patterns). **UNIFIED
memory (121 GiB, CPU+GPU shared): vLLM's default 0.9 gpu-mem-util
reserves ~109 GiB and OOM-kills the whole userspace once client load
lands (2026-10-16: pipewire/llama-swap/session died mid-run). Always
serve with an explicit --gpu-memory-utilization ≤ 0.6 (all serve sites
default GPU_MEM_UTIL=0.6; env-overridable — use 0.9 on dedicated pod
hardware) — the box is shared with llama-swap and a
desktop session.**

**Unified-memory big-load doctrine (2026-10-17, cost: 3 dead runs + 2
desktop sessions).** Any >30GB model load on this box needs ALL of:
- `HF_DEACTIVATE_ASYNC_LOAD=1` — transformers 5.x async load (4 reader
  threads) piles completed CPU tensors into converter futures → ~2×
model footprint → OOM at ~45%. Default in quant/run_quant.sh.
- The mem_guard fence (runlib.sh, `MEM_GUARD_MB`): GPU-side allocations
  are INVISIBLE to RSS and to the kernel OOM killer's scoring — under
  pressure it shoots random userspace while the 67GB-holder shows
  rss=21MB. /proc/meminfo MemAvailable DOES see unified allocations;
  guard kills OUR pid before the kernel fires. EVERY GPU-touching probe
  or run gets a floor — no exceptions (an unfenced probe killed the
  desktop again on 2026-10-17).
- Page-cache hygiene during load: `scripts/evict_load_cache.py` (loop)
  + PAGE_CACHE_EVICT (post-load, in quantize.py) — just-read base
  shards sit in cache NEXT TO the resident copy unless dropped.
- NEVER load big MoE via baseline transformers `from_pretrained` here:
its weight_mapping merges ALL experts into ONE ~70GB tensor (watched
it request 70,214,746,112 bytes with 21.8GB free). The llmcompressor
`load_quantizable_moe` wrap path linearizes per-layer (~1.6GB
transients) and never mega-allocs — use it for 35B-class loads.

**Models registry.** `models.py` keys: 4b (champion, taps_v4
[2,10,17,24,29,32]), 0.8b (testbed, taps [2,8,12,16,20,24], local quant
`Qwen3.5-0.8B-W4A16-GPTQ/`), 9b, 27b (taps UNVALIDATED).
Quant recipe: `quant/run_quant.sh <key>` (W4A16 GPTQ, cascade-v1 calib).
**Calib truth: cascade-v1 = 999×4096 windows, 100%
nvidia/Nemotron-Cascade-2-SFT-Data (checksum-pinned in data/MANIFEST.sha256;
build_calib_mix.py's 9-bucket spec does NOT match the frozen file — provenance
gap, document-before-rebuild). All quants to date share this lineage.**

**Regen doctrine (the standing per-model process).** Contexts are the
shared invariant; every target model regenerates finals ON-POLICY with its
deployed quant: `drafter/regen_pool.sh <model-key>`. Sampling =
`models.py SERVING_PARAMS` (t 0.7 / top_p 0.95 / top_k 20 — the deployment
point + DFlash2 distribution pair), also stamped into every quant's
generation_config.json at quant time. Outputs land in `data/` (this repo).
GB10 measured: conc 128, MTP off (spec-decode LOSES at saturation:
313 vs 249 rows/min), GPU_MEM_UTIL=0.6. Historical: the champion's v1
regen predates SERVING_PARAMS and lacked top_k.

**Literature idiom (verified from primary sources).** On-policy regen
by the exact deployed target; one conv per sample (first-turn common);
short or capped lengths (paper 3072; DSpark ≤1024); PerfectBlend =
 ecosystem default mix and NOBODY ablates mix; `minimum-valid-tokens`
filter (speculators) — we lack it.
