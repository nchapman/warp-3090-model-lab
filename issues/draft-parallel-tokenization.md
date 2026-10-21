# data_audit / prep_dataset tokenization is single-threaded (~25 min for 145K rows on a 192-core B200 pod)

Measured (2026-10-17, B200 pod, regen_4b_final.jsonl 902MB / 145,018 rows):
`data_audit.py --jsonl` took ~25 min at 100% of ONE core (192 available).
GPU idle throughout. Same cost lands again in `prep_dataset.py` — same
`tokenize_with_masks` path — on every full-corpus prep, i.e. on the
critical path of every campaign.

## Cause
`tokenize_with_masks` (drafters/prep_dataset.py) tokenizes each message
incrementally: for message i it re-renders the whole prefix via
`apply_chat_template` and tokenizes the delta — O(n_msgs²) jinja renders
per conversation, pure Python, serial across rows.

## Fix 1 (quick, safe): parallelize over rows
Row-parallel multiprocessing pool (chunked, deterministic order preserved
on write). Audit and prep both drop ~25 min → ~20-30s at 64-192 workers.
No change to per-row outputs — bit-identical ids/masks.

## Fix 2 (right, needs parity test): single-pass assistant masks
`tok.apply_chat_template(msgs, tokenize=True, return_assistant_tokens_mask=True)`
produces assistant-token masks in one render — removes the O(n_msgs²)
entirely and the jinja overhead with it. BUT it must be proven equivalent
to the incremental masks on real data before replacing the validated path
(the 46c2331 prep template fix history is exactly why this needs a parity
gate: same masks on a frozen sample, incl. think-tag turns and tool rows).

Suggested acceptance: parity harness over a 5K-row frozen sample
(incremental vs single-pass: identical id sequences + identical masks),
then Fix 1 for parallelism regardless.
