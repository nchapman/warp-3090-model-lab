#!/bin/bash
# 100-row end-to-end regression — run this BEFORE any paid GPU launch.
# Catches the whole bug class (tap shapes, template drift, save gate, fp8 cast)
# in <15 min on one GPU. Env: BASE (regen source), QUANT (verifier for
# extract/train), OUT (workspace, wiped), PORT, VENV (drafter venv bin).
set -eu
ROOT="$(cd "$(dirname "$0")" && pwd)"
BASE=${BASE:-nchapman/Qwen3.5-4B-W4A16-GPTQ}
QUANT=${QUANT:-$BASE}
OUT=${OUT:-/tmp/regression_warp3090}
PORT=${PORT:-8143}
VENV=${VENV:-$ROOT/.venv-dflash/bin}
export PATH="$VENV:$PATH"
cd "$ROOT"
T0=$(date +%s)

rm -rf "$OUT"; mkdir -p "$OUT"

echo "[1/4] regen: serve verifier + generate 100 short conversations"
setsid nohup "$VENV/vllm" serve "$QUANT" --port "$PORT" --max-model-len 4096 \
  --gpu-memory-utilization 0.55 > "$OUT/regen_server.log" 2>&1 &
for i in $(seq 1 40); do sleep 15; curl -s -m 3 "localhost:$PORT/health" >/dev/null 2>&1 && break; done
curl -s -m 3 "localhost:$PORT/health" >/dev/null 2>&1 || { echo REGRESSION_FAIL server; exit 1; }
"$VENV/python" - "$PORT" "$OUT" << 'PY'
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
rows = []
for i in range(100):
    q = f"Explain concept {i} from first principles in two sentences."
    body = json.dumps({"messages": [{"role": "user", "content": q}],
                       "max_tokens": 120, "temperature": 1.0}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    rows.append({"messages": [{"role": "user", "content": q},
                              {"role": "assistant",
                               "content": r["choices"][0]["message"]["content"]}]})
with open(f"{out}/regen.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("regen:", len(rows), "rows")
PY
pkill -f "vllm ser[v]e"; sleep 8

echo "[2/4] build 100-row dataset (assistant-masked) + extract (6-tap fp8)"
"$VENV/python" drafter/prep_dataset.py --jsonl "$OUT/regen.jsonl" --out "$OUT/states" \
  --quant "$QUANT" --limit 100 | tail -1
CUDA_VISIBLE_DEVICES=0 "$VENV/python" drafter/extract_states.py --dataset "$OUT/states" \
  --verifier "$QUANT" --out "$OUT/states" --model 4b --limit 100 --parity 20 2>&1 | tail -3

echo "[3/4] draft bootstrap + 10 train steps + val"
"$VENV/python" drafter/bootstrap_draft.py --model 4b --quant "$QUANT" --out "$OUT/draft_init"
CUDA_VISIBLE_DEVICES=0 "$VENV/speculators" train --speculator-type dflash2 \
  --draft-attn-impl sdpa --from-pretrained "$OUT/draft_init" \
  --verifier-name-or-path "$QUANT" --data-path "$OUT/states" \
  --epochs 1 --lr 1e-4 --max-steps 10 --checkpoint-freq 1 \
  --save-path "$OUT/draft_reg" > "$OUT/train.log" 2>&1
tr '\r' '\n' < "$OUT/train.log" | grep -a "loss" | tail -3
"$VENV/python" - "$OUT" << 'PY'
import re, sys
log = open(sys.argv[1] + "/train.log", errors="ignore").read()
losses = [float(m) for m in re.findall(r"loss\w*=([0-9]+\.[0-9]+)", log)]
assert losses and max(losses) > 0.01, f"training loss suspiciously zero: {losses[:3]}"
print(f"TRAIN_LOSS_OK max={max(losses):.3f}")
PY

echo "[4/4] tap-shape validator (states vs draft aux dim)"
"$VENV/python" - "$OUT" << 'PY'
import glob, json, sys
from safetensors.torch import load_file
out = sys.argv[1]
p = sorted(glob.glob(f"{out}/states/hidden_states/hs_0.safetensors"))[0]
hs = load_file(p)["hidden_states"].shape
ck = sorted(glob.glob(f"{out}/draft_reg/*/model.safetensors"))
print(f"states taps={hs[1]} hidden={hs[2]} (expect 6 taps; draft aux = 5 x {hs[2]})")
assert hs[1] == 6, "tap count mismatch"
assert ck, "no trained checkpoint saved"
print("VALIDATOR_OK", ck[0].split('/')[-2] if ck else "")
PY

echo "REGRESSION_PASS $(( $(date +%s) - T0 ))s"
