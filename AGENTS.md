# warp-3090-quant-lab — agent rules

Project-specific doctrine for any agent (or human) working in this repo. The
global engineering guidelines still apply; this file adds what this lab
learned the hard way.

**Maintenance contract:** every number below cites instrument, temp, and n.
When a new instrument, artifact, or decision lands, its line here changes in
the same commit. A doctrine file that lags the repo is worse than none —
"no multi-domain eval exists" survived three days after `bench_subsets.sh`
shipped, and misinformed a planning session.

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
  - **t=0.7 (serving)** — the deployment point and the selection temperature.
  **Selection happens here**, n≥4 per contender (spread ~0.1).
  - **t=1.0** — heavy-tailed (0.34 spread observed on identical inputs).
  Never rank checkpoints on small n at t≥0.7; never quote a single draw as a
  result. Report n and ranges with every number.
- Ranking is temperature-dependent. State the temperature with every τ. A
  checkpoint that wins at greedy may lose at serving — that's data, not error
  (rung 1 is the canonical case: greedy improved 3.407 (t=0, n=1) →
  4.157 (t=0, n=2) while serving sat flat at ~1.05, t=0.7 n=4).
- Fresh GB10 numbers (or same-machine re-anchors) are the lab's anchors
  (`RESULTS.md`). Don't publish cross-machine τ comparisons; don't resurrect
  superseded numbers.

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

- Two venvs: `.venv-quant` (torch 2.14.1 / llmcompressor 0.14.1a),
  `.venv-dflash` (torch 2.13.0 / speculators 0.8.1 / vllm 0.31.0).
  `vllm serve` needs the venv's bin on PATH.
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
v4 regen input): 145,018 rows, every row ends user/tool, ZERO regenerated
responses. **THIS FILE IS THE REGEN INPUT, NOT A CORPUS.** The final
assistant turn of every conversation is absent; single-turn rows (51% —
ALL math + science, plus 60K chat/IF/agent) carry no supervision at all
and are dropped by prep. The champion's corpus (contexts + on-policy
finals, ~150M supervised) was pod-side and is GONE — contexts survive
here; the next corpus is built fresh per the campaign spec below.
- Mix — the file's row labels (chat 65/math 12.5/science 8/…) describe
  PROMPT POOL composition, not training data. Ladder-side effective mix:
  chat ~87% / agents ~11% / IF ~2%; math 0, science 0 (single-turn → no
  supervision). Code ≈ 6% of context content (measured).
- Length: mean 2,115 / median 1,676 / p90 4,588 / p99 6,207 tokens —
  **uncapped**; DFlash paper spec caps at 3072, DSpark reference ≤1024.
- Shape tooling: `drafter/reshape_tokens.py` (prefix/merged/first/short),
  `prep_dataset.py --loss-mode all|last --max-tokens N` (drop cap).
- Full corpus narrative: `docs/data_scaling.md`.

**Eval instruments** (two exist; long-context: none yet — don't claim
deployment relevance on long inputs without building one):
- `data/ladder-20.jsonl` + `drafter/eval_ladder.sh` — GENERAL CHAT ONLY;
  the selection instrument (t=0.7, n≥4). Every ladder-quoted τ (incl.
  champion 3.79) is general-chat τ.
- `drafter/bench_subsets.sh` — multi-domain profile: RedHatAI
  speculator_benchmarks, 9 subsets / 924 prompts, revision-pinned
  `2ae86af` (repins are logged, never silent), t=0 default, interval-
  weighted τ from vLLM logs; `TEMP`/`SUBSET_N`/`SPEC`/`FLOOR_PER_PROMPT`
  envs. GB10 confirmed profile (2026-10-14, n=2): champion weighted
  **4.65** vs native MTP k=3 **3.21**, wins 9/9 — strongest
  math_reasoning 6.64 / HumanEval 4.85 / rag 5.03; weak tail translation
  +0.08 (chat-heavy-corpus signature). (A first-pass read of 4.16/3.26
  8/9 was superseded by the fixed-window pass — quote 4.65/3.21.)

**τ bars and the dead-zone map** (τ @ t=0.7 n≥4, ladder-20, unless noted;
external references — full provenance in `docs/external_references.md`):
- **THE BAR: untrained MTP head = 2.868** (4B quant's native MTP,
  **k=3**, ladder-20 @ t=0.7, single run, GB10 — `results/results.csv`
  `spec=mtp-k3`; consistent with the GB10 anchor table's k=3 → 2.86).
  Naming trap: the producing run is named `mtp-15` because `eval_mtp.sh 15
  8151` consumed the 15 as the run NAME — arg 3 is SPEC (default 3, capped
  at 8). "MTP k=15" never existed; every doc that said so was citing the
  run name. Quoted as the floor in every later campaign incl. the B200
  closeout (not re-measured there). Any shipped drafter must beat it;
  nothing at ≤1.4 ships.
- **Reference recipes:** DSpark-class ≥3.41 (citation under reconciliation —
  paper trains 1.3M prompts/10 epochs, chat τ 3.29–3.64; resolve before
  quoting externally, `docs/external_references.md` §7); DFlash paper
  100K-sample ablations → 4.9–5.6 greedy.
- Dead zone: <~5M supervised (two sources); known-good 20–60M; rising to
  ~1B (DFlare). 0.8B: 0.7M → 1.024, 3.5M → 1.08–1.09 (n=1) — dead zone
  extends to ≥3.5M at 0.8B.
- Rung 1 (4B, 5.2M supervised): **not a clean dead-zone number** —
  off-policy data (source responses only). At t=0.7 it sat flat at 1.05
  across all epochs (n=4) while its greedy sweep *improved* over the
  epoch-0 control (3.407→4.157, t=0, n=1 vs n=2) — greedy gains that
  don't transfer to serving are the distribution-mismatch signature. Do not quote as a
  dead-zone datapoint.
- Historical only, do not chase: a 4B run reached 3.79 @ t=0.7 n=4 (GB10;
  pod re-anchor 3.714 n=4) — proof the recipe-class can reach ≥3.7, and
  nothing more.

**Loss-weighting footgun** (speculators 0.8.1, `schema.py`): unset
`--per-position-loss-weight` → **dpace for plain `dflash`**, fixed-exp-decay
for dflash2/dspark — BUT `--dflash-decay-gamma` defaults to **4.0**, so the
documented γ=7 static recipe still requires pinning the flag. All our
scripts pin both explicitly (champion lineage = fixed-exp-decay γ7; dpace
additionally requires `--loss-fn=ce`). Draft-OPD trainer patch is inert
without opd_ columns — default SFT path unaffected (`docs/draft_opd.md`).

**Trainer behavior.** Multipack sampler re-shuffles per epoch (seed+epoch)
→ epoch step-counts vary (rung 1: 1,310–1,850+); `step_ms` excludes
inter-step overhead — wall is ~1.7–2× step_ms. Plan in hours/epoch
(GB10: ~3.5h per 10M nominal; B200: 111 min at 197.5M total), never in
step counts.

**GB10 reality.** Bandwidth-starved (~17.6× off H200): extraction and
training both ~2k tok/s class. nvidia-smi memory reads [N/A]; 30–70W
at 96% util is normal load, not idle. A hung nvidia-smi can hold
/dev/nvidia0 — kill by exact PID (never pkill patterns). **UNIFIED
memory (121 GiB, CPU+GPU shared): vLLM's default 0.9 gpu-mem-util
reserves ~109 GiB and OOM-kills the whole userspace once client load
lands (2026-10-16: pipewire/llama-swap/session died mid-run). Always
serve with an explicit --gpu-memory-utilization ≤ 0.6 (every serve
script defaults GPU_MEM_UTIL=0.6; env-overridable — use 0.9 on dedicated
pod hardware) — the box is shared with llama-swap and a
desktop session.**

**Models registry** (`models.py` — single source of truth for per-model
facts): `4b` (imported champion ckpt — warm-start source ONLY, taps_v4
[2,10,17,24,29,32]), `0.8b` (testbed, taps [2,8,12,16,20,24], local quant
`Qwen3.5-0.8B-W4A16-GPTQ/` at repo root), `9b` (init config built, never
trained), `27b` (taps UNVALIDATED; measured NO-GO — bounded upside vs
~5× pod cost). Quant recipe: `quant/run_quant.sh <key>` (W4A16 GPTQ).
**Calib truth: cascade-v1 = 999×4096 windows, 100%
nvidia/Nemotron-Cascade-2-SFT-Data (checksum-pinned in
data/MANIFEST.sha256; `quant/build_calib.py`'s bucket spec (7 domains,
CARD_TOTALS) does NOT match the frozen file — provenance gap,
document-before-rebuild). All quants to date share this lineage.**

**Regen doctrine.** Contexts are the shared invariant; every target model
regenerates finals ON-POLICY with its deployed quant:
`drafter/regen_pool.sh <model-key>`. Generator config follows the
reference idiom — **t 0.7 / top_p 0.95, NO top_k**, stated length cap,
one conv per sample, minimum-valid-tokens filter ON. CAUTION:
`regen_pool.sh` as committed still stamps `models.py SERVING_PARAMS`
(t 0.7 / top_p 0.95 / **top_k 20**) into the generator — the exact
deviation the 2026-10-17 closeout flagged (~$170 lesson; serving params
in a data generator is an untested divergence from every reference, and
`models.py`'s "consumed by regen" docstring predates the lesson).
Parameterize the generator config before the next campaign. Outputs land
in `data/` (this repo), sha-pinned. GB10 measured: conc 128, MTP off
(spec-decode LOSES at saturation: 313 vs 249 rows/min),
GPU_MEM_UTIL=0.6. This finals-only process is **v4 lineage**; the
Open-PerfectBlend campaign replaces it with full sequential regen —
every assistant turn, history rebuilt wave by wave (see Next
campaign) — same generator idioms (t 0.7 / top_p 0.95, no top_k,
stated cap), but prep-side knobs are stock speculators defaults
(`--seq-length` 8192, `--minimum-valid-tokens` OFF).

**Next campaign (decided 2026-10-17/18) — reference conformance first.**
- Data source = **Open-PerfectBlend** (HF `mlabonne/open-perfectblend`,
  apache-2.0, ~1.3M samples, ShareGPT `conversations` + per-row `source`
  column for stratification): public, sha-pinnable, rebuildable, and
  DSpark's corpus — kills data variability. User turns verbatim;
  assistant turns regenerated by the deployed quant (full sequential
  regen, below). Stratify rungs
  by `source` so every scale holds the same mix.
- **Datashape = stock speculators preprocessing — no bespoke prep
  (decided 2026-10-18).** Feed `conversations` as-is to `speculators
  prepare-data --render-endpoint <quant vLLM>`: the library normalizes
  ShareGPT `from`/`value`, fans out **one row per assistant turn**, each
  with a boundary loss mask (0 on context, 1 on that turn) and history
  re-rendered exactly as inference sees it; trailing non-assistant turns
  drop; a turn whose context alone exceeds the cap is skipped per-turn,
  not per-conv. Verified 0.8.1 knob defaults: `--seq-length` 8192,
  `--minimum-valid-tokens` OFF (None), seed 0. No `--loss-mode`, no
  `reshape_tokens.py`, no custom caps — that machinery (and its churn)
  is retired for this campaign. This supersedes the collapse-vs-keep
  prep question: the answer is neither, it's per-turn fan-out.
- **Regen scope = EVERYTHING (decided 2026-10-18).** Stock fan-out
  supervises every assistant turn present in a conversation, so
  all-on-policy requires sequential whole-conversation regen: keep
  source user turns verbatim, regenerate each assistant turn
  conditioned on the already-regened history, emit the rebuilt
  conversation. Fan-out then supervises only target-generated text,
  with contexts exactly as inference sees them. Frozen-context
  per-turn regen is NOT viable on the conversations path (earlier
  source turns would get supervised off-policy) and would force the
  bespoke `(input_ids, loss_mask)` passthrough — rejected. Single-turn
  rows (the math half) are unchanged: one turn = one regen. Regen runs
  in batched waves across conversations (turn-1 for all convs, then
  turn-2, …) — vLLM batches each wave; multi-turn convs chain wave to
  wave. Assistant-turn volume ≈ 1.6–1.8M turns (~1.3M single +
  multi-turn extras; measure exact count from the pinned dataset and
  budget pod regen in waves). New script pattern required —
  `regen_pool.sh` finals-mode is v4 lineage only.
- Reference recipe = paper optimizer (6 ep / 6e-4 cosine / γ7 / b16 /
  taps_v4), CE loss, quant verifier, stock datashape + knob defaults.
  Warm-start eligible from ctrl ep2 (`pod_snapshot/ctrl_closeout/`,
  sha-verified).
- Gate = reference conformance: beat the MTP floor 2.868, target
  DSpark-class. Never gate on a predecessor checkpoint.
- Re-divergence queue (one axis at a time, only after the reference run
  conforms): KL-vs-CE, bf16-vs-quant verifier, taps 6→9.
- External numbers, bars, and open reconciliation items live in
  `docs/external_references.md` — check §7 before quoting any external
  bar.
