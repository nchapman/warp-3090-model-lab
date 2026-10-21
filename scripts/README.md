# scripts/ — pod & repo ops

Lab-wide operational helpers (shared by all subprojects). Workflow code
lives in `quants/`, `drafters/`, `finetune/`, `experiments/`.

| script | role |
|---|---|
| `pod_prep.sh <user@host> [port]` | fresh-pod bootstrap: survey, HF token, guarded rsync (refuses surprise gigabytes), frozen-data manifest check, both venvs, speculators patches, GPU check, model pulls. Re-anchor any baseline on the pod itself before comparing numbers across machines. |
| `podsh.py` | run commands on a RunPod pod through the ssh.runpod.io console — the gateway is a PTY-demanding console, not an exec channel; this drives the proven OpenSSH path with pexpect. |
| `pod_status.sh` | one-shot machine utilization snapshot (run ON the pod): GPU util/mem/power, GPU processes (zombie spotting), load, RAM, disk, newest run event. |
| `pod_pull_pass.sh` | one local sync pass from the pod (run repeatedly, chained); verifies tar integrity each pass; touches `/root/PULL_CONFIRMED` when PIPELINE_TERMINAL + manifest verify — the green light for ship + teardown. |
| `tail_run.sh [prefix]` | inspect lab runs without guessing where anything went: cmd.txt + events.jsonl + server log tail (no arg = latest run). |
| `evict_load_cache.py` | incremental page-cache evictor for GB10 unified memory — caps the ~2× double-hold while safetensors shards stream in during model load. |
| `upload_frozen_data.py` | push the frozen data sets to the HF mirror (`nchapman/warp-3090-quant-lab-frozen-data`; see `data/README.md`). |

Pod workflow wrapper: `pod.sh` at the repo root (push / run / pull).
Process doctrine (run logging, PID kills, watchdogs): `AGENTS.md` +
`lib/README.md`.
