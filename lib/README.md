# lib/ — shared run plumbing

One lifecycle for every run in the lab, bash and python. The process
failures this project has hit (silent input mismatch, wedged servers,
results in unexpected places) are failure modes of unmanaged runs, not of
the models — so every script sources or imports one of these two. No
exceptions (see `AGENTS.md` "Runs and processes").

| file | language | used by |
|---|---|---|
| `runlib.sh` | bash | `regression.sh`, `drafters/eval_ladder.sh`, `drafters/eval_mtp.sh`, `quants/run_quant.sh`, `quants/capability_battery.sh`, … |
| `lablog.py` | python | `quants/probe.py`, `quants/quantize.py`, `quants/build_calib.py`, … |

## The contract

- **Every run** — evals, regression, quantize, probe, calib builds — gets
  `runs/<name>.<UTC ts>.<pid>/` with `cmd.txt` (exact invocation) and
  `events.jsonl` (structured lifecycle: `RUN_START`, `PREFLIGHT_OK/FAIL`,
  `SERVER_LAUNCH/HEALTHY/STUCK/FAILED`, stage markers,
  `RUN_DONE/RUN_FAILED`).
- **Preflight or die**: inputs are validated by name before anything
  launches (`require_file`).
- **Watchdogged servers**: health waits fail fast — a server whose log
  goes idle *and* whose GPU is 0% for two consecutive probes is
  `SERVER_STUCK`, killed by its recorded PID, with a process snapshot in
  the log. No 15-minute silent hangs. (`EXTRA_SERVE_ARGS="--enforce-eager"`
  dodges cudagraph-capture wedges when throughput doesn't matter.)
- **Servers die by PID** — never `pkill` scatter.
- **events are structured JSON**, one object per line: `ts` (single clock
  read — seconds and ms fields can't disagree), `type`, then key=value
  fields. `log_event` accepts both `"k1=v1" "k2=v2"` and the legacy packed
  `"k1=v1 k2=v2"` form (split only when every whitespace token looks like
  `key=value`; free-text values stay one field).

## Inspecting runs

```bash
bash scripts/tail_run.sh [name-prefix]   # cmd + events + server log tail
```

Never go hunting for log paths manually — the run dir is
`runs/<name>.<UTC ts>.<pid>/`.
