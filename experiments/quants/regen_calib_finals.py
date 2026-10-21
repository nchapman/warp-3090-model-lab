#!/usr/bin/env python3
"""Regenerate the FINAL assistant turn of a frozen calib file with a teacher
model served via OpenRouter.

Regen doctrine, ported to calib construction: contexts are the shared
invariant — every message except the last assistant turn is preserved
byte-identical from the source file; only the final answer is rewritten by
the teacher. The output file keeps the source schema (conversations/source/
domain/generator) so quantize.py consumes it unchanged.

Design notes (2026-10-18):
- Thinking stays ON: the source finals (Nemotron) carry <think> blocks, so
  the shipped recipe already calibrates over think-formatted text; stripping
  them from the teacher arm would change the stream shape, not just the
  answer policy. --no-think flips this (kept for a future arm).
- Sampling matches the student's deployment point (models.py SERVING_PARAMS:
  t 0.7 / top_p 0.95 / top_k 20), not the teacher's vendor recs — we want
  finals that look like what the deployed quant emits.
- tool-role messages (no tool_calls linkage in this SFT format) are sent to
  the API as user turns; the frozen row itself is untouched.
- Per-row seed = --seed + row index, pinned into the sidecar.
- Resumable: rows already present in --out (matched by source-row sha) are
  skipped on rerun.
- Cost guard: aborts before exceeding --cost-cap USD (usage.cost from the
  API when present, else token-count estimate).

Usage:
  OPENROUTER_API_KEY=... python3 quant/regen_calib_finals.py \
      [--calib data/calib-cascade-v1.jsonl] \
      [--out data/calib-cascade-qwenfinals.jsonl] \
      [--limit 3]   # smoke first, always
      [--api-base http://localhost:8137/v1 --model Qwen/Qwen3.5-4B]  # self-authored
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib.lablog import log_event, run_done, run_fail  # noqa: E402

DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
# fallback price table (USD/1M tok) if usage.cost is absent — qwen3.8-flash
PRICE_IN, PRICE_OUT = 0.15, 0.47

_state = {"cost": 0.0, "tokens_in": 0, "tokens_out": 0, "done": 0,
           "fail": 0, "trunc": 0}
_lock = threading.Lock()


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:16]


def api_messages(context, no_think):
    msgs = []
    for m in context:
        role = m["role"]
        if role == "tool":
            role = "user"  # SFT format has no tool_calls linkage; see docstring
        msgs.append({"role": role, "content": m["content"]})
    if no_think and msgs and msgs[0]["role"] == "system":
        msgs[0]["content"] += " /no_think"
    return msgs


def call_openrouter(key, model, msgs, seed, params, max_tokens, retries,
                    api_base=DEFAULT_API_BASE, local=False):
    url = api_base.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": msgs,
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "top_k": params["top_k"],
        "seed": seed,
        "max_tokens": max_tokens,
    }
    if not params.get("thinking", True):
        body["reasoning"] = {"enabled": False}
    data = json.dumps(body).encode()
    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                out = json.load(resp)
            choice = out["choices"][0]["message"]
            text = choice.get("content") or ""
            reasoning = choice.get("reasoning") or ""
            finish = out["choices"][0].get("finish_reason", "")
            # OpenRouter strips reasoning out of `content` into its own
            # field. Reconstruct the Qwen-native format the source finals
            # (and the shipped calib stream) already use, so the only delta
            # of this arm is WHO wrote the tokens, not the stream shape.
            if reasoning:
                text = f"<think>\n{reasoning}\n</think>\n\n{text}"
            if not text.strip():
                raise ValueError("empty completion content")
            usage = out.get("usage", {})
            if local:
                cost = 0.0  # self-served vLLM: no meter
            else:
                cost = usage.get("cost")
                if cost is None:
                    cost = (usage.get("prompt_tokens", 0) / 1e6 * PRICE_IN
                            + usage.get("completion_tokens", 0) / 1e6 * PRICE_OUT)
            return text, cost, usage, finish == "length"
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError,
                KeyError, IndexError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            code = getattr(e, "code", None)
            body = ""
            if code is not None:
                try:
                    body = e.read().decode()[:200]
                except Exception:
                    pass
            log_event("BACKOFF", http=code, attempt=attempt,
                      err=body or str(e)[:120])
            # 429/5xx: exponential backoff; be gentle with workers
            time.sleep(min(2 ** attempt * 2, 60))
    raise RuntimeError(f"exhausted {retries} retries: {last_err}")


def process_row(idx_row, args, params, key):
    idx, row = idx_row
    context = row["conversations"][:-1]
    try:
        text, cost, usage, trunc = call_openrouter(
            key, args.model, api_messages(context, args.no_think),
            seed=args.seed + idx, params=params,
            max_tokens=args.max_tokens, retries=args.retries,
            api_base=args.api_base, local=args.local)
    except RuntimeError as e:
        return idx, None, str(e)
    new_row = dict(row)
    new_row["conversations"] = context + [
        {"role": "assistant", "content": text}]
    new_row["generator"] = (f"{args.model}@"
                            f"{'local-vllm' if args.local else 'openrouter'}")
    with _lock:
        _state["cost"] += cost
        _state["tokens_in"] += usage.get("prompt_tokens", 0)
        _state["tokens_out"] += usage.get("completion_tokens", 0)
        _state["done"] += 1
        if trunc:
            _state["trunc"] += 1
    return idx, new_row, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calib", default="data/calib-cascade-v1.jsonl")
    p.add_argument("--out", default="data/calib-cascade-qwenfinals.jsonl")
    p.add_argument("--model", default="qwen/qwen3.8-flash")
    p.add_argument("--api-base", default=DEFAULT_API_BASE,
                   help="OpenAI-compatible base URL; point at a local vLLM "
                        "(http://localhost:PORT/v1) for self-authored arms")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--cost-cap", type=float, default=15.0)
    p.add_argument("--limit", type=int, default=0, help="0 = all rows")
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--no-think", action="store_true")
    args = p.parse_args()

    args.local = args.api_base.startswith(("http://127.", "http://localhost"))
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        if args.local:
            key = "local"  # self-served vLLM: auth unused
        else:
            print("OPENROUTER_API_KEY not set — add it to ~/.config/secrets.env "
                  "and open a fresh shell (see AGENTS.md secrets section)",
                  file=sys.stderr)
            sys.exit(2)

    rows = [json.loads(l) for l in open(args.calib)]
    if args.limit:
        rows = rows[:args.limit]
    src_hash = hashlib.sha256(open(args.calib, "rb").read()).hexdigest()

    done_hashes = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l)
            # resume key: the ORIGINAL context (all but final turn) + domain
            ctx = json.dumps(r["conversations"][:-1], sort_keys=True)
            done_hashes.add(hashlib.sha256(
                (ctx + r["domain"]).encode()).hexdigest()[:16])

    todo = []
    for i, r in enumerate(rows):
        ctx = json.dumps(r["conversations"][:-1], sort_keys=True)
        if hashlib.sha256((ctx + r["domain"]).encode()).hexdigest()[:16] \
                not in done_hashes:
            todo.append((i, r))

    params = {"temperature": 0.7, "top_p": 0.95, "top_k": 20,
              "thinking": not args.no_think}
    log_event("RUN_START", calib=args.calib, out=args.out, model=args.model,
              api_base=args.api_base, local=args.local,
              todo=len(todo), skipped=len(rows) - len(todo),
              sampling=json.dumps(params), seed=args.seed,
              thinking=params["thinking"])

    t0 = time.time()
    fail_rows = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex, \
            open(args.out, "a") as fout:
        for idx, new_row, err in ex.map(
                lambda t: process_row(t, args, params, key), todo):
            if err is not None:
                fail_rows.append((idx, err))
                _state["fail"] += 1
                log_event("ROW_FAIL", row=idx, error=err[:300])
            else:
                fout.write(json.dumps(new_row) + "\n")
                fout.flush()
            n = _state["done"] + _state["fail"]
            if n % 100 == 0:
                log_event("PROGRESS", done=n, total=len(todo),
                          cost_usd=round(_state["cost"], 2))
            if _state["cost"] >= args.cost_cap:
                log_event("COST_CAP", hit=round(_state["cost"], 2),
                          cap=args.cost_cap)
                ex.shutdown(wait=False, cancel_futures=True)
                break

    # sidecar provenance — lives next to the data file
    sidecar = {
        "teacher": (f"{args.model}@"
                    f"{'local-vllm' if args.local else 'openrouter'}"),
        "api_base": args.api_base,
        "sampling": params, "seed_base": args.seed,
        "max_tokens": args.max_tokens,
        "source_calib": args.calib, "source_sha256": src_hash,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows_requested": len(todo), "rows_done": _state["done"],
        "rows_failed": len(fail_rows), "rows_length_capped": _state["trunc"],
        "cost_usd_est": round(_state["cost"], 2),
        "tokens_in": _state["tokens_in"], "tokens_out": _state["tokens_out"],
        "note": "contexts byte-identical to source; only final assistant "
                "turn regenerated; tool-role msgs sent to API as user "
                "(frozen rows unchanged)",
    }
    sc_path = os.path.splitext(args.out)[0] + ".provenance.json"
    with open(sc_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    if fail_rows:
        run_fail(rows_failed=len(fail_rows), first=str(fail_rows[0])[:300])
        sys.exit(1)
    run_done(rows=_state["done"], cost_usd=round(_state["cost"], 2),
             tokens_in=_state["tokens_in"], tokens_out=_state["tokens_out"],
             wall_s=round(time.time() - t0))


if __name__ == "__main__":
    main()
