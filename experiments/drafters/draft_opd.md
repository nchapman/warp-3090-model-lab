# Draft-OPD — on-policy distillation from verification boundaries

Implementation of [arXiv:2605.29343](https://arxiv.org/abs/2605.29343)
(Draft-OPD) for our DFlash2 drafters, plus the supporting collector/assembler.
**Experimental, opt-in**: nothing here runs unless invoked explicitly, and the
trainer patch is behaviorally inert on data without `opd_*` columns.

## What it is

SFT plateaus because the drafter trains on target trajectories but serves on
its own proposals. Draft-OPD post-trains on real verification outcomes:
target-assisted rollouts are collected with the drafter in the loop, and
drafting is *replayed* from each anchor with the draft's own tokens in
context — the drafter learns from target feedback on both accepted proposals
(forward-KL-equivalent CE) and rejected suffixes (reverse-KL vs the stored
target distribution, weights `w_k = γ^(k-1)`, γ=0.8, λ_rej∈[0.5,1]).

Both stages are **training-time only** — serving is unchanged; a result is a
checkpoint swap.

## Pipeline (all verified end-to-end on real data, 2026-10-16)

```
1. drafter/opd_collect.py   serve+generate with the hook; waves of N
2. drafter/opd_assemble.py  validated attribution -> anchors.jsonl
3. drafter/opd_build_replay.py  per-anchor replay rows (draft tokens in ctx)
4. drafter/prep_dataset.py  (pre-tokenized passthrough) -> arrow ds
5. drafter/extract_states.py   target states on draft-induced prefixes
6. speculators train ... --from-pretrained <SFT ckpt>   (lr ~5e-5, 1-2 ep)
7. eval_ladder vs the SFT baseline checkpoint
```

Apply the trainer patch first (inert by default):

```bash
SP=$(./.venv-dflash/bin/python -c 'import speculators,os;print(os.path.dirname(os.path.dirname(speculators.__file__)))') \
  && patch -p1 -d "$SP" < drafters/patches/speculators_draft_opd.patch
```

## Verified semantics (vLLM 0.31.0, dflash path)

Discovered by probe (`drafter/opd_probe.py`) — do not assume the obvious
class is the right hook:

- The verify boundary is **`vllm.v1.worker.gpu.spec_decode.
  rejection_sampler_utils.rejection_sample`** — NOT
  `vllm.v1.sample.rejection_sampler.RejectionSampler` (never called for
  dflash).
- The engine runs in a subprocess; the hook requires
  `VLLM_ENABLE_V1_MULTIPROCESSING=0` (set before importing vllm — a
  `setdefault` after import silently no-ops).
- Per step: `cu_num_logits` slices rows per request **slot**; slot order can
  permute mid-wave. Attribution is by **anchor echo**: `draft_sampled[
  cu[i]][0]` equals the request's last-emitted token.
- `draft_sampled[cu[i]+1 : cu[i]+16]` are the 15 proposals (the 16-slot
  block includes the anchor echo at slot 0; `num_speculative_steps=15`).
- `sampled[i][:num_sampled[i]]` = tokens emitted this step (accepted
  proposals + correction/bonus). Full stream = `output_token_ids[0]`
  (prefill token) + concatenation. Any wave whose assembled streams don't
  exactly reproduce the returned ids is dropped loudly.
- `draft_logits` (the drafter's own q) is None on this path — fine: replay
  training recomputes q from the *current* drafter; only the target top-k
  (p) is stored (top-64, -inf entries from the masked vocab region dropped).

**Instrument cross-check**: boundary mean accepted+1 = 1.100 vs ladder τ
1.081 for the same checkpoint (smoke/draft_static, 2026-10-16) — collection
and eval agree independently.

## Trainer patch behavior

`drafters/patches/speculators_draft_opd.patch` (three files, applied to the venv):

- `train/data.py`: attaches `opd_*` columns from the ds to samples.
- `models/dflash2/core.py`: gathers opd tensors to the **anchor grid**
  (`block_indices` — unary_logits live there, not on the sequence grid);
  masks the standard loss at opd positions; adds the opd term + metrics
  (`opd/loss`, `opd/acc_ce`, `opd/rej_rkl`).
- `losses/utils.py`: `draft_opd_loss` — CE on mode-1 positions, `λ_rej · w_k`
  reverse-KL vs renormalized stored top-k on mode-2 positions; raw position
  counts in the denominator (no phantom weight when a mode is absent);
  `DRAFT_OPD_LAMBDA_REJ` env (default 1.0).

Inert without opd columns — the standard SFT path (incl. both
per-position-loss-weight modes) is byte-identical.

## Known approximations (deliberate, smoke-scale)

- **Top-k truncated reverse KL**: p is stored as top-64 only; q mass outside
  the stored support is uncounted, so the surrogate can read slightly
  negative. Still alignment-minimizing; noted wherever results are quoted.
- **Single OPD stage** (collect → train once), like the paper; not iterated.
- **Collection at serving temp** (SERVING_PARAMS), matching the paper; our
  rollout uses greedy-vs-sample acceptance as vLLM implements it.

## Results so far

See RESULTS.md (0.8B smoke section) — collection/assembly/training verified
end-to-end on real data; no quality verdict yet (awaiting the full-scale
0.8B round on the Blackwell pod).
