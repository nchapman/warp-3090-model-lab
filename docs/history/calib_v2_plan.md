# Calibration v2 — Hypotheses & Experiment Plan

Status: DISCUSSION ONLY — no changes until the 9B ncblend battery finishes and we've
made the ship/iterate call. This doc captures what we believe and how we'd test it.

## Where v1 (ncblend) landed

Deployment target: **Qwen3.5-27B W4A16 on 3090 Ti (24GB)** — ~15GB of weights,
so the remaining ~8GB must cover KV + activations; selective-precision budget is
+1-2GB max (first-N-layers down_proj @W8 fits; full down_proj @W8 is borderline).
4B/9B are iteration proxies only. Note quantization sensitivity *decreases* with
scale, so 4B results are a conservative upper bound on 27B divergence. Final
ladder: iterate on 4B → confirm on 9B → single 27B confirmation run (quant +
probe + battery subset) before shipping.

- 949 rows, 1.23M effective tokens @4096, 0 template failures
- Effective token mix: 67% SWE-agentic, 12% multilingual SFT, 7% UltraInteract,
  <5% each math/code/chat/MCQ (row shares were 21/16/16/... — tokens are what the
  Hessian sees; rows were the wrong mental model)
- Non-EN 26% (vs cyankiwi's 67%)
- 4B: KL 0.0774 (cyankiwi 0.0838) ✓, ΔPPL +1.78% (cyankiwi +0.29%) ✗,
  HE/HE+ 0.832/0.797 vs 0.832/0.781 (tie/+1.6pp, not significant)
- 9B: KL 0.0521 (cyankiwi 0.0616) ✓, ΔPPL +1.72% (cyankiwi +1.08%) ✗,
  HE greedy 0.872/0.823 vs 0.896/0.835 (statistical tie), battery pending

Consistent signature: ncblend preserves distribution shape better (KL, HE+),
cyankiwi preserves next-token likelihood better (ΔPPL). Hypothesis: reasoning-chain
density (long-CoT math) drives ΔPPL; agentic/tool-loop token mass drives KL/HE+.

## Probe upgrade first (before any v2 data work)

Current probe (eval_set.jsonl, 120 code-agent EN convs) is domain-biased; a
fidelity-first objective needs a global fidelity estimate.

- P1. Multi-domain probe: ~50 held-out convs each from {code-agent, multilingual
  (non-EN), math long-CoT, agentic tool-loop} — no source overlap with calibration.
- P2. Add p99.9 KL (tail) alongside mean — Unsloth's 99.9% KLD shows tails matter;
  occasional garbage tokens come from tail blowups, not the mean.
- Decision rule: v2 candidates rank by mean KL across P1 domains + no domain
  regressing p99.9 by >2x baseline.

## Data hypotheses (v2 mix)

- H1 (SWE windowing): head-truncation wastes the 4096-token budget on task-setup
  boilerplate; the agentic regime (dense assistant→tool→assistant cycles, deep
  DeltaNet state) lives mid-trajectory. v2: window-slice tool-dense mid-regions
  of ~150-200 traces instead of heads. Prediction: agentic-domain KL and HE+
  improve at same token budget; other domains flat.
- H2 (long-CoT source): cyankiwi's ΔPPL edge comes from Nemotron-Post-Training-style
  long reasoning chains (~25% of its rows), absent from v1. v2: add ~150 long-CoT
  math rows (EN, 2-4K tokens). Prediction: ΔPPL closes to ≤+0.8% without KL regression.
- H3 (token-share rebalance): 67% agentic token share overshoots; the sweet spot is
  ~40-50%. Achieved via H1 (shorter windows) + H2 (more reasoning mass) + non-EN
  to ~35-40%. Prediction: both probe metrics improve together (no tradeoff).
- H4 (sequence length ceiling, optional): raising max_seq_length to 8192 for a
  subset exercises deeper DeltaNet state accumulation, matching deployment contexts.
  One controlled run, compare at matched token budget. Prediction: small but real
  agentic-domain KL gain.

## Method hypotheses (independent of data; Unsloth-informed)

- M1 (selective precision): leave `down_proj` (and/or first-N layers, attn-adjacent
  tensors — sensitive on hybrid architectures) at W8A16 while the rest go W4A16.
  Costs ~10-15% size (fits 24GB target). Prediction: mean+p99.9 KL drop ≥20% for
  the size cost; test before dataset changes since it's data-independent.
- M2 (metric): adopt p99.9 KL everywhere (see P2).

## Experiment order (once battery verdict is in)

1. P1+P2 probe upgrade (ratchet before we turn any knobs)
2. M1 selective-precision run on v1 data (cheapest, orthogonal lever)
3. v2 data build: H1 + H2 (+H3 targets), one variable at a time, 4B first,
   promote to 9B only on clear probe wins
4. H4 as a single optional side experiment

## Decision framework (unchanged)

KL + ΔPPL (multi-domain) rank candidates; HE/battery confirm no catastrophes.
Fidelity-first: optimize "matches the original model," not task-pulling.

## v2 result: Cascade-2 slice (4B, 2026-10-09)

| metric | cyankiwi | ncblend | cascade |
|---|---|---|---|
| cascade probe meanKL | 0.2082 | 0.1997 | **0.1612** |
| cascade probe p99.9 | 19.06 | 19.35 | 18.11 |
| guard dPPL | +1.62% | +1.66% | **+0.76%** |
| guard meanKL | 0.1037 | 0.0875 | **0.0821** |

Per-domain wins across chat/IF/math/science (-16..-32%); conv_agent wall 2.0 -> 1.58;
swe slightly worse; tails unchanged. Guard improved alongside in-distribution ->
genuine fidelity gain, no gaming signature. Next: HE n20 sanity, then 9B chain.

## v2 reweight result (cascade-v2, 4B, 2026-10-09 evening)

Agentic 9% -> 29% of effective tokens (940 rows): conv_agent 1.58->1.36,
terminal 0.75->0.67, swe 1.03->0.91 (partial). chat/IF/math/science FLAT
(cost check passed). Aggregate 0.1598->0.1463 (best), dPPL +1.76% (best),
p99.9 17.04 (best). Costs: ML mean 0.1034->0.1155 (arb/hin worst), guard
gave back ~6%. Open items for v3: ~8-10% ML rows (arb/hin-weighted) from
chat; swe still >0.9 despite 40 rows.

## Probe-v4 frozen (power-sized ruler, 378 rows, analytic blend weights)

| | cyankiwi | ncblend | cascade-v1 | cascade-v2 |
|---|---|---|---|---|
| weightedKL | 0.1340 | 0.1273 | **0.1087** | 0.1120 |

Row-share aggregate crowned v2; blend-weighted score crowns v1 — agentic
token share in probe (~50%) masked the chat/IF/ML givebacks. v3 target:
v2 mix + ~100 ML rows (arb/hin-weighted) + ~40 chat rows restored, agentic
untouched. Per-row dumps saved for all 4 quants (rows_v4_*.jsonl).

## Qwen3.8-27B cascade quant (first 27B, 2026-10-10 morning)

Artifact: Qwen3.8-27B-W4A16-GPTQ-cascade (19GB, 6h15m quant, rc=0).
probe-v4: weightedKL 0.1264, dPPL +1.00%, p99.9 24.66, raw 0.357
guard: weightedKL 0.1111, dPPL -0.52% (!), raw 0.0867
Per-domain: chat 0.0326 (best ever), IF 0.0529, math 0.0981, science 0.0967,
ML 0.1086, terminal 0.4860, conv_agent 2.2032, swe 1.0207.
Notes: guard dPPL negative. conv_agent wall HIGHER than 3.5-4B (2.20 vs 1.60)
and p99.9 elevated (24.7) — cross-generation caveat (3.8 != 3.5; absolute KL
not comparable across model generations). dPPL by domain benign except swe +12%
(known wall). Verdict: gate passed.

## 9B cascade verdict (2026-10-10): gate PASSED
weightedKL 0.0889 (best anywhere), conv_agent 1.25 (wall falls with scale
within 3.5: 1.60->1.25), guard dPPL +0.77%. Scale-robustness confirmed
same-generation; 27B domain spikes attributed to 3.8 generation.

## B300 duo@999 attempt (2026-10-10): UNRESOLVED — pod stopped, no verdict

Rented B300 (275GB) to run duo (AWQ+GPTQ) at matched 999 cascade samples vs
champion 0.1087. Outcome: no numbers. Run #1 OOM'd at layer 2 (261GB/267GB
allocated, mlp.up_proj grid search) — even 275GB VRAM does not fit AWQ caches
at 999x4096; offload_device=cpu re-added (matches local commit) for run #2,
which was ~30 min in (PCIe-bound, ETA 3-6h, ~$40-55 total projected) when the
pod was stopped. duo@999 is CLOSED-UNANSWERED; plain GPTQ uploads stand.

Durable learnings:
- AWQ cache = raw per-sample activations, linear in tokens. 999x~2k avg =
  ~2M tokens vs library default 512x384 (~200k): 10x. down_proj mapping alone
  (9216-wide input) is ~37-75GB at our seq lengths. GPTQ is immune (Hessian
  O(d^2)=13MB) — the recipe asymmetry that killed GB10 x3 and B300 x1.
- Peak inflated ~4x by cross-subgraph cache retention (261GB observed vs
  ~60GB/layer ideal). Grid-search error is a SUM over samples: a chunked
  evaluator would be bit-identical at O(chunk) memory — unimplemented
  upstream; candidate issue/patch, NOT applied (verbatim-recipe rule).
- With offload, VRAM need = model + temporaries (~15-20GB at 4B): if ever
  reopened, a small-GPU/big-RAM box at ~$1-2/hr runs duo@999; no B300 needed.
- Remote env root cause: `pip install -e .` gave a no-deps editable (deps in
  setup.py, pyproject has no [project] table) — full dep list in HANDOFF_B300.md.

## Duo at library regime (2026-10-10): LOSES — question closed, uploads final

duo512x384 = canonical AWQ+GPTQ recipe, library-default data budget
(512 samples x 384 tokens, cascade data, GB10, 50min quant, first duo run
to ever complete: 5th attempt across GB10/B300).

| | cascade-v1 GPTQ (999x4096) | duo512x384 (512x384) |
|---|---|---|
| weightedKL | **0.1087** | 0.1209 (+11%) |
| dPPL | +3.35% | +3.05% |
| guard dPPL | **+0.76%** | +1.38% (~2x worse) |

Paired bootstrap (rows_v4_3 vs rows_v4_duo512x384): duo significantly
WORSE on conv_agent (+0.547 KL, ~29% rel), terminal_agent (+0.086),
math (+0.0033), chat (+0.0020); significantly better only on science/
IF/swe (small); ML pooled tie. The agentic domains — the entire point
of cascade — are exactly where their token budget starves the scaling
search: 384 tokens amputates agentic context and AWQ can't recover what
the data never shows.

Verdict: their out-of-the-box pipeline at their recommended budget loses
to our tuned plain GPTQ. Gate = LOSE -> stop; live GPTQ uploads stand
final. Matched-999 duo question stays formally open (needs chunked
grid-search patch; cheap small-GPU/big-RAM path documented above) but
has no remaining practical motivation.
