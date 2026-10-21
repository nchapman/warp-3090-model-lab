# data/ — frozen rulers, pinned context, regen outputs

**Frozen deliberately:** every published number is measured on these exact
files (weightedKL on probe-v4, rollout agreement on traj-v1, τ on
ladder-20), so drifting them silently breaks cross-run comparability.
Verify your checkout:

```bash
# the gate — complete copy; REQUIRED before any measurement campaign
(cd data && sha256sum -c MANIFEST.sha256 --quiet && echo "data OK — complete copy")
# fresh clone: verify the tracked subset, failing closed on anything absent
# except the six disk-only pinned files (pull them from the HF mirror)
(cd data && sha256sum -c \
  <(grep -vE '  (context_pool\.jsonl|regen_4b[^ ]*)$' MANIFEST.sha256) \
  --quiet && echo "data OK — tracked subset")
```

The gate form fails on a fresh clone by design: the manifest also pins the
**disk-only** files below (`context_pool.jsonl`, `regen_4b*.jsonl`, its
`.errors.jsonl` sidecar — gitignored, absent until pulled). The subset form
skips exactly those six names and nothing else — any other missing or
corrupted pinned file fails the check.

## The frozen sets

| file | what | provenance |
|---|---|---|
| `calib-cascade-v1.jsonl` | champion calibration set (999 rows) | 100% nvidia/Nemotron-Cascade-2-SFT-Data, dataset-native proportions (verified 2026-10-16) |
| `probe-v4.jsonl` | frozen 378-row fidelity ruler (the weightedKL metric) | cascade pool per V4_TARGETS + Belebele 10×9 langs |
| `probe-guard.jsonl` | OOD guard (no_robots 40 + Belebele 4×9 langs) | no_robots **train** split (verified vs cached upstream arrows) + facebook/belebele |
| `traj-v1.jsonl` | frozen 301-row rollout-agreement ruler (greedy 32-token trajectory match vs base — the compounding regime teacher-forced KL can't see) | cascade pool per TRAJ_TARGETS (agentic 100: terminal 50/swe 25/conv_agent 25; math 60, IF 25, chat 40, chat_long 40 @ 12–24K chars, Belebele 4×9 fresh rows) |
| `traj-v1.jsonl.sources.json` | provenance sidecar for traj-v1 (19 upstream shard hashes) | written by the build that froze it |
| `ladder-20.jsonl` | the 20 held-out eval prompts (the τ metric) | rows 50–69 of the v1-era 100-row cascade dev set (`dflash_data_v1`), frozen since the first DFlash2 run; hash-banned from drafter training, disjoint from the probes. Prompt texts recur inside calib rows (both cascade-derived) — affects no quoted number |
| `conv-ids/` | conversation-id manifests for the v3/v4/v5 training slices | exact `slice_convs.py` preset outputs — deterministically reproducible (seeds, per-subset rounding, val sampling all verified against the manifests) |
| `MANIFEST.sha256` | hash pin for everything it lists: the frozen sets above, the calib-v2-era experimental calib builds + their provenance sidecars (`calib-cascade-if3x*`, `calib-cascade-4bself*`, `calib-cascade-qwenfinals*`), the disk-only files below, and the raw-pool pin file `MANIFEST.raw-pools.sha256`. Not pinned: the manifest itself, this README (on the Hub the `README.md` path is the dataset card — `hf/frozen-data/`), and untracked scratch | prefix-free paths relative to `data/`. The uploader enforces on-disk set == pin list (after those two top-level skips); scratch left in `data/` aborts the upload by design. Grown only by deliberate append — never regenerate from a `find .` sweep |

## Where each set is used — three processes, three roles, walled off

All frozen sets are pairwise hash-disjoint; the builders enforce it.

| process | uses | notes |
|---|---|---|
| quantization (`quants/quantize.py`) | `calib-cascade-v1` only | activations decide GPTQ grouping; no_robots/Belebele never enter — keeps the guard honest |
| drafter training (`drafters/slice_convs.py` → `regen.py` → …) | cascade pool prompts via `conv-ids` presets | slices hash-banned from every frozen set; `regen.py` rewrites assistant turns with the deployed quant — training data is quant-written completions + quant states, never cascade's original assistant text, never bf16 |
| evaluation (`quants/probe.py`, `quants/traj_probe.py`, `drafters/eval_ladder.sh`) | `probe-v4` → headline weightedKL; `traj-v1` → rollout agreement (agree@1, exact@32, per-domain — where looping/tool-format drift shows); `probe-guard` → OOD anti-gaming check; `ladder-20` → headline τ + per-epoch checkpoint selection | ladder-20 is hash-banned from training; probe-guard and traj-v1 are consumed by nothing upstream — pure held-out referees |

One documented exception: ladder-20 prompt texts recur inside calib rows
(both cascade-derived) — τ compares drafters against the same target model,
so the overlap can't game it. `regression.sh` regenerates its own smoke
dataset and never reads `data/`.

## Reproducibility contract

`quants/build_calib.py` builds *new* sets of each calib/probe/guard kind —
always hash-disjoint from the frozen ones (freshness by design, not
regeneration). Every build writes a `<out>.sources.json` sidecar pinning
each upstream file it read (dataset revision + SHA256), so silent upstream
drift (HF reshards, split-order flips) is detectable by diffing sidecars.
The no_robots split is explicit (`--no-robots-split`, default `train` = the
frozen set's actual source split, verified against the cached upstream
arrows; the converted-parquet API has served split lists in unstable order).

One exception, closed 2026-10-17: `calib-cascade-v1` **is** regenerable —
`--mode v1-repro` rebuilds it **byte-exact** from the raw pool
(`data-raw/cascade/`, 21 parquet shards, gitignored like all raw pools and
hash-pinned in `MANIFEST.raw-pools.sha256` — verify with
`sha256sum -c data/MANIFEST.raw-pools.sha256` from the repo root; the pin
file itself is registered in `MANIFEST.sha256`). The mode refuses to write
into `data/` and self-verifies both artifacts — the calib against the
frozen file's bytes, the co-built day-0 probe companion against the
original ruler's pinned sha256 — exiting nonzero on any drift. probe-v4 and
probe-guard stay freshness-by-design: new builds are hash-disjoint, never
regenerations.

## Pinned but disk-only (gitignored)

These are hash-pinned in `MANIFEST.sha256` but not in git — present on the
lab machines, restored from the HF mirror or pod pipeline elsewhere:

- `context_pool.jsonl` — the
  **regen input, not a corpus**: 145,018 rows / ~100K convs / 260M nominal;
  bare contexts, every row ends user/tool, zero regenerated responses. Read
  the corpus-truth section of `AGENTS.md` before quoting any number derived
  from it.
- `regen_4b*.jsonl` (incl. the `.errors.jsonl` sidecar) — on-policy regen
  outputs from `drafters/regen_pool.sh`, frozen by the 4b campaign.

Unpinned scratch also lives here — `regen_0.8b.jsonl`, `smoke*_corpus*.jsonl`,
`smoke_corpus_clean.jsonl`, `opd_dryrun/` — regression fixtures and probe
outputs. The uploader aborts on them until removed; the gate does not cover
them.

## Licenses

Check before redistributing: cascade-derived sets → NVIDIA Open Model
License; probe-guard no_robots rows → CC-BY-NC-4.0 (non-commercial);
Belebele rows → CC-BY-SA-4.0. Regen outputs are Qwen model outputs.

## HF mirror

The frozen sets are dual-homed at
`nchapman/warp-3090-quant-lab-frozen-data` (**public**, CC-BY-NC-4.0
top-level per the card — non-commercial research use; see license note
above). Pinned at commit `945980f`; verify against `MANIFEST.sha256` after
pulling. Good reason to freeze *on the Hub* rather than git alone:
revision-hash citability, pull-by-pods without a repo clone, and a
license/provenance surface git can't provide.
`scripts/upload_frozen_data.py` performs the upload.

## Dataset naming

Dataset files are named for their **purpose** (`calib-*`, context pool,
regen outputs) with sources documented in the table above — never for the
experiment that produced them. (Quants follow the same rule; see
`quants/README.md`.)
