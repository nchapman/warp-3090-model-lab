"""Standard event logging for python lab tools — mirrors lib/runlib.sh.

Every tool emits the same event vocabulary into runs/<tool>.<UTC ts>/events.jsonl
(RUN_START / RUN_DONE / RUN_FAILED / tool-specific stages). Values must be
simple strings/numbers.

Usage:
    from lablog import log_event, run_done, run_fail
    log_event("RUN_START", model="4b", eval_set="data/probe-v4.jsonl")
    ...
    run_done(weighted_kl=0.1087)
"""
import json
import os
import sys
import time

_TOOL = os.path.splitext(os.path.basename(sys.argv[0]))[0]
_ROOT = os.environ.get("RUNS_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
_DIR = None


def _dir():
    global _DIR
    if _DIR is None:
        _DIR = os.path.join(_ROOT, f"{_TOOL}.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.{os.getpid()}")
        os.makedirs(_DIR, exist_ok=True)
    return _DIR


def log_event(etype, **kw):
    obj = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
           + f".{int(time.time()*1000)%1000:03d}Z", "type": etype, "tool": _TOOL}
    obj.update(kw)
    with open(os.path.join(_dir(), "events.jsonl"), "a") as f:
        f.write(json.dumps(obj) + "\n")
    print(f"[{obj['ts']}] [{_TOOL}] {etype} "
          + " ".join(f"{k}={v}" for k, v in kw.items()), file=sys.stderr)


def run_done(**kw):
    log_event("RUN_DONE", status="ok", **kw)


def run_fail(**kw):
    log_event("RUN_FAILED", status="fail", **kw)
