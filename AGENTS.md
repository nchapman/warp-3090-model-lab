# warp-3090-quant-lab — agent rules

Project-specific doctrine for any agent (or human) working in this repo. The
global engineering guidelines still apply; this file adds what this lab
learned the hard way.

**Maintenance contract:** every number below cites instrument, temp, and n.
When a new instrument, artifact, or decision lands, its line here changes in
the same commit. A doctrine file that lags the repo is worse than none —
"no multi-domain eval exists" survived three days after `bench_subsets.sh`
shipped, and misinformed a planning session.

## Repo layout (per-activity monorepo, 2026-10-20)

- `quants/` — quant workflow (calib build, quantize, verify, probe,
  publish, capability battery, model cards; `requirements-quant.txt`).
  Local quant artifacts are
  untracked bare root dirs named for what they are (`*-W4A16-GPTQ*`,
  gitignored) — there is no tracked `quants/` artifact directory.
- `drafters/` — drafter workflow: slicing, on-policy regen, prep,
  extraction, ladders/evals, `train.md` recipes, draft_config taps;
  self-contained with `requirements-drafter.txt`, `patches/`
  (speculators fp8 load-cast, draft-OPD), `data_scaling.md`,
  `external_references.md` (speculator literature — check its §7
  before quoting any external claim).
- `finetune/` — managed/external fine-tuning workflows (scaffold; charter
  in its README).
- `lib/` — shared run plumbing: `runlib.sh` (bash), `lablog.py` (python).
- `scripts/` — pod-side and repo ops (pod_prep, podsh, tail_run, upload).
- `experiments/` — closed campaigns, code+docs+records frozen together:
  `experiments/quants/` (calib-v2 arm, battery records),
  `experiments/drafters/` (draft-OPD, dflash2 research, overnight
  campaign). Records and writeups there are frozen receipts — they may
  cite pre-reorg paths (`drafter/`, `quant/`, the old root `quants/`
  artifact dir); read them as historical.
- Untracked conventions (gitignored): `runs/` `results/` `slices/`
  `states/` `ds_*/` `draft_*/` `eval_*/` `data-raw/` `smoke/` and root
  `*-W4A16-GPTQ*` dirs. `results/results.csv` is live eval output;
  anything worth keeping graduates to `experiments/*/records/`.
- Venv and requirements file names keep their historical identities
  (`.venv-quant`, `.venv-dflash`, `requirements-drafter.txt`) — renaming
  forces rebuild churn on every pod for zero information. The files live
  inside their subprojects (`quants/requirements-quant.txt`,
  `drafters/requirements-drafter.txt`); the venvs themselves stay at the
  repo root.

## Runs and processes

- Every eval/server/training run goes through `lib/runlib.sh` (bash) or
  `lib/lablog.py` (python) — no exceptions. If a script runs anything
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

- `TEMP` env on `drafters/eval_ladder.sh` / `eval_mtp.sh` selects sampling
  temperature (default 1.0). For checkpoint selection, use
  `drafters/sweep_epochs.sh` — one command, full campaign. Three regimes,
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
  **No sequenced v-numbers for calib variants either** (2026-10-18):
  name the recipe delta — `calib-cascade-agent3x` (agentic domains
  tripled), `calib-cascade-if3x` (IF share tripled) — never `v2`/`v4`.
  Historical references keep a recipe gloss on first mention
  (cascade-v2 = agent-tripled, cascade-v3 = agent×1.5+ML).
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
supervised** (1.93× exchange). `drafters/data_audit.py` measures all of
these — run it, never guess.

**Corpus** (`data/context_pool.jsonl`, sha-pinned; read-only copy of the
v4 regen input): 145,018 rows, every row ends user/tool, ZERO regenerated
responses. **THIS FILE IS THE REGEN INPUT, NOT A CORPUS.** The final
assistant turn of every conversation is absent; single-turn rows (51% —
ALL math + science, plus 60K chat/IF/agent) carry no supervision at all
and are dropped by prep. The champion's exact pod-side MERGED corpus
is gone, but its ingredients all survive locally: contexts here + the
recovered regen lineage — `data/regen_4b{,_final,_sft,_opd}.jsonl` +
empty errors receipt (0 errors), synced pod→local 2026-10-16 (06b1890),
sha-pinned in data/MANIFEST.sha256, dual-homed in the main checkout
(disk-only blobs, same pattern as context_pool.jsonl). Byte-exact prep
is NOT recoverable; rebuild from ingredients if ever needed. The next
corpus is built fresh per the campaign spec below.
- Mix — the file's row labels (chat 65/math 12.5/science 8/…) describe
  PROMPT POOL composition, not training data. Ladder-side effective mix:
  chat ~87% / agents ~11% / IF ~2%; math 0, science 0 (single-turn → no
  supervision). Code ≈ 6% of context content (measured).
- Length: mean 2,115 / median 1,676 / p90 4,588 / p99 6,207 tokens —
  **uncapped**; DFlash paper spec caps at 3072, DSpark reference ≤1024.
- Shape tooling: `drafters/reshape_tokens.py` (prefix/merged/first/short),
  `prep_dataset.py --loss-mode all|last --max-tokens N` (drop cap).
- Full corpus narrative: `drafters/data_scaling.md`.

**Eval instruments** (two exist; long-context: none yet — don't claim
deployment relevance on long inputs without building one):
- `data/ladder-20.jsonl` + `drafters/eval_ladder.sh` — GENERAL CHAT ONLY;
  the selection instrument (t=0.7, n≥4). Every ladder-quoted τ (incl.
  champion 3.79) is general-chat τ.
- `drafters/bench_subsets.sh` — multi-domain profile: RedHatAI
  speculator_benchmarks, 9 subsets / 924 prompts, revision-pinned
  `2ae86af` (repins are logged, never silent), t=0 default, interval-
  weighted τ from vLLM logs; `TEMP`/`SUBSET_N`/`SPEC`/`FLOOR_PER_PROMPT`
  envs. GB10 confirmed profile (2026-10-14, n=2): champion weighted
  **4.65** vs native MTP k=3 **3.21**, wins 9/9 — strongest
  math_reasoning 6.64 / HumanEval 4.85 / rag 5.03; weak tail translation
  +0.08 (chat-heavy-corpus signature). (A first-pass read of 4.16/3.26
  8/9 was superseded by the fixed-window pass — quote 4.65/3.21.)

**Quant-damage instruments** (battery + rollout ruler; the deployment
side, distinct from the τ instruments above). The rollout ruler:
`data/traj-v1.jsonl` (301 rows, agentic 100 / math 60 / chat+long 80 /
IF 25 / Belebele 36) read by `quants/traj_probe.py` — paired greedy 32-
token rollouts; report agree@1, exact@32, per-domain (the compounding
regime; Unsloth Divergence-300-style). Teacher-forced KL (probe.py) is a
different instrument — never average the two, compare per-domain
directions. Runbook: needs `.venv-quant` (the dflash venv lacks
accelerate and crashes at load); ~11 min/leg on GB10 for the 4B.
- **Battery tiering (2026-10-19, user doctrine):** the full capability
battery is the 4B-tier instrument — it calibrates probe↔battery
agreement. Larger models (27b flagship, 35b class) verify by PROBE
first; battery only for anomalies or a contested shipping call.
Rationale: larger models should be more quant-resilient; a 4-side
battery at 27B dense ≈ $75–150/arm on one H200 (~1–2k tok/s), and probe
ranking already transfers. If a big-model battery ever runs: reuse the
base side, right-size hardware (A100-80 fits 27B BF16), reserve
full-length ifeval/gpqa for the finalist.
- **Battery protocol (v2, 2026-10-19):** generation cap 16384, mlen 20480;
truncation-at-cap is a LENGTH metric, never a graded miss —
paired-complete scoring (`battery_report.py --gen_cap`) excludes docs
where either side capped (measured over-16k at 4B: ifeval 23–32%, gpqa
4.5–6%, math 0.4%, code 0%; quant truncs more than base — the gap
itself is damage). 8k cap ruled out (ifeval >54% would truncate).
- **Battery protocol modes (2026-10-19 verdict):** chat-template
  (`EXTRA_LMEVAL_ARGS=--apply_chat_template`) and raw-completion runs
  are DIFFERENT instruments — completion-16k hammers everyone (champion
  −11.8 ifeval vs chat-mode −7.6, no-think −1.7) and inflates quant
  gaps; a cross-mode champion-vs-contender comparison manufactured a
  false 10pt "damage cliff". Deployment mode (chat template, SERVING
  regime) is the decision protocol. capability_battery.sh stamps
  chat_template/think_args/max_gen_toks/max_model_len into events;
  battery_report.py HARD-FAILS cross-protocol pairing. Matched-mode
  verdict: v1rebuild is champion-class (−6.5 as-built vs −7.6 champion,
  n=541) — a/b BF16 exclusion retained, no rebuild needed.
- **Battery-number hygiene (2026-10-20, user doctrine):** quote mode
  with every battery number (comp-4096 / comp-16k / chat16k-think /
  chat16k-nothink); never mix modes in one comparison; completion-mode
  numbers are regime diagnostics, never deployment claims. Pre-10-20
  runs carry no protocol stamp — verify mode from run logs before
  pairing. Authoritative standing of every historical battery claim:
  `experiments/quants/records/battery-protocol-audit-20261020.md`.
- **Battery noise floor (measured 2026-10-19, two independent base runs
at identical config):** ifeval strict is format-robust and stable
(0.2pt base-vs-base) — the trustworthy discriminator. gpqa
flexible-extract swings ±9.6pt (only flexible-extract is nonzero;
strict-match ≈0 for thinking models). **math500 is UNUSABLE for
ranking on this instrument** — vLLM batch nondeterminism flips
reasoning paths → answer-format flips → minerva strict AND math_verify
swing 35–57pt base-vs-base (81.4/24.6 strict, 91.8/56.8 verify; 34.8%
doc agreement). Never quote math500 battery deltas. Correction: earlier
resps-proxy over-16k rates (ifeval 23–32%) undercounted — battery_report
measures base 57–58% / champion 72% / v1rebuild 67% on ifeval.
- **Probe↔battery predictive power: UNMEASURED (2026-10-20 framing).**
The 10-19 "−35pt PC cliff the probe missed" was a cross-mode artifact
(matched-mode verdict: −6.5, champion-class); the clean test is the
10-20 gate — probes ranked qf far better, battery/McNemar said null
(p=0.57). That is ONE null pair, not a refutation: probe→battery
predictive power is unmeasured in both directions (negative direction
never cleanly tested; earlier non-correlations were protocol defects,
see the audit). Probe-first tiering stands; probe-negative ≠ ship-safe
(a shipping call needs the format-robust subset: ifeval + code).

**τ bars and the dead-zone map** (τ @ t=0.7 n≥4, ladder-20, unless noted;
external references — full provenance in `drafters/external_references.md`):
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
  quoting externally, `drafters/external_references.md` §7); DFlash paper
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
without opd_ columns — default SFT path unaffected (`experiments/drafters/draft_opd.md`).

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
facts). Keys — six: 4b, 0.8b, 9b, 27b, 35b, 35b-ornith.
- `4b` — imported champion ckpt, taps_v4 [2,10,17,24,29,32]. RETIRED as
  any kind of spec (2026-10-18, main-line closeout): warm-start source
  ONLY; the shipped champion quant predates it.
- `0.8b` — testbed, taps [2,8,12,16,20,24], local quant
  `Qwen3.5-0.8B-W4A16-GPTQ/` at repo root.
- `9b` — init config built, never trained.
- `27b` — taps UNVALIDATED; measured NO-GO (bounded upside vs ~5× pod
  cost).
- `35b` — Qwen3.6-35B-A3B MoE, quant published 2026-10-17, no taps —
  drafter tooling untested.
- `35b-ornith` — Ornith-1.5-35B-A3B, cross-lineage fork, quant
  published 2026-10-17, vision tower excluded from quant.
4B artifact warning (2026-10-17): the champion calib-recipe artifact lives in
~/Code/llm-compressor/Qwen3.5-4B-W4A16-GPTQ-**cascade** (suffix REQUIRED);
the unsuffixed dir there is an ablation-era leftover that ablations overwrote
in place — never probe it as the champion.
Champion era caveat (2026-10-19): the shipped champion is a PRE-d281deb
build — its GDN in_proj_a/b are W4-packed (3.9M params): vLLM-servable,
NOT SGLang/Marlin-safe. All post-10-18 builds keep a/b BF16 (compat
requirement, see quantize.py ignore history). Champion-calib rebuilds
under today's code are tagged `v1rebuild`.
New-class 4B artifacts (2026-10-20, all local in ~/Code/llm-compressor/):
`Qwen3.5-4B-W4A16-GPTQ-v1rebuild` (cascade-v1 calib, current code — the
publish candidate) and `Qwen3.5-4B-W4A16-GPTQ-gdnfix-qwenfinals` (teacher-
authored finals: qwen/qwen3.8-flash via regen_calib_finals.py). Battery-gate
verdict (n=541, chat16k-think, McNemar p=0.57): the healthy-quant family
(champion / v1rebuild / qwenfinals) is FLAT — same-box traj ruler confirms
(RESULTS.md 2026-10-20b). Calib-authorship swaps are probe-only screens.
Quant recipe: `quants/run_quant.sh <key>` (W4A16 GPTQ, cascade-v1 calib).
**Release metadata (2026-10-20, user decision):** every quant ships the
CUSTOM release chat template (froggeric v22.5, byte-pinned as
quants/release_chat_template.jinja, sha-guarded in quantize.py, standalone-file
mechanism identical to the published champion) + the SERVING_PARAMS
generation_config stamp. Stock template does NOT ship. Measured 0.5pt-neutral
on ifeval (n=541) — consistency with the published family wins.
**Calib truth: cascade-v1 = 999 rows (truncated 4096 tok), 100%
nvidia/Nemotron-Cascade-2-SFT-Data, stratified to the dataset's own card
proportions — safety dropped (0.02%), chat 58.7 / math 18.5 / science 11.4 /
IF 5.2 / conv_agent 3.5 / terminal 2.0 / swe 0.6 (verified against the frozen
file to the row, 2026-10-17). Original builder = build_cascade_slices.py in
~/Code/llm-compressor (build_calib_mix.py is the earlier ncblend 9-bucket
builder — different artifact; that conflation caused the old "provenance gap"
note). RECONSTRUCTION VERIFIED 2026-10-17: `quants/build_calib.py --mode
v1-repro` rebuilds the frozen calib AND its day-0 probe companion byte-exact
from data-raw/cascade/ (gitignored raw pool, 21 shards first/mid/last per
subset, hash-pinned in data/MANIFEST.raw-pools.sha256); the same mode is the
tuning entry point — change --seed/targets for new variants, calib mode for
leak-proof sets. (The script's DEFAULT CARD_TOTALS mode re-derives the card
mix rather than reproducing the frozen file — `--mode v1-repro` is the exact
path; superseded main's old "provenance gap" note.) Deliberate remixes all
lost to the card mix: v2 agentic×3 0.1120, v3 agentic×1.5+ML 0.1123,
perfectblend 0.1432. All quants share this lineage.**

**Regen doctrine.** Contexts are the shared invariant; every target model
regenerates finals ON-POLICY with its deployed quant:
`drafters/regen_pool.sh <model-key>`. Generator config follows the
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

**Calib-authorship vocabulary + measured null (2026-10-20 — naming
conflation burned us twice in one day; do not repeat).** The arms:
- `gdnfix-qwenfinals` = TEACHER-authored finals (qwen/qwen3.8-flash
  via OpenRouter, `regen_calib_finals.py`). NEVER call this "selfgen".
- `4bself` = the ONLY true selfgen arm (4B base regenerates its own
  finals, `regen_calib_self.sh`, calib-cascade-4bself.jsonl).
- champion/v1rebuild lineage = curated cascade-v1 (Nemotron SFT).
MEASURED: teacher-authored ≈ curated in deployment mode — ifeval chat16k
same-base pairing, qf −8.7 vs v1r −7.4, paired McNemar exact p=0.57
(RESULTS.md 2026-10-20). The probe gain was real (dPPL 0.98 vs 3.33)
and transferred nowhere. Calib-authorship swaps are probe-only screens.
Null discipline: when the paired test says no difference, the entry
says no difference — no direction words ("worse"), no mechanism
stories on unmeasured effects. 4bself remains unmeasured in
deployment mode (artifact lost 2026-10-20 — the parked H200 pod was
deleted; arm rebuildable from the banked calib + regen_calib_self.sh).

**Literature idiom (verified from primary sources).** On-policy regen
by the exact deployed target; one conv per sample (first-turn common);
short or capped lengths (paper 3072; DSpark ≤1024); PerfectBlend =
 ecosystem default mix and NOBODY ablates mix; `minimum-valid-tokens`
filter (speculators) — we lack it.

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
  `drafters/external_references.md` — check §7 before quoting any external
  bar.
