#!/usr/bin/env bash
# vLLM smoke test: serve the miniature qwen3_5_moe W4A16 artifact on GB10 and
# generate once. Validates the full llmcompressor -> vLLM MoE W4A16 path
# (per-expert split packed experts + qwen3_5_moe arch) before committing to
# the real 67GB run.
set -euo pipefail
cd "$(dirname "$0")/../.."
source lib/runlib.sh

run_init "mini-moe-vllm-smoke" "${BASH_SOURCE[0]}"
log_event RUN_START model=mini-qwen36moe-w4a16 purpose="serve-path-validation"

PORT=8137
ARTIFACT=${ARTIFACT:-/tmp/mini-qwen36moe-w4a16}
LOG="$RUN_DIR/vllm.log"
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}

[[ -f "$ARTIFACT/config.json" ]] || { log_event RUN_FAILED stage=artifact; echo "no artifact at $ARTIFACT (build it with /tmp/mini_e2e.py or set ARTIFACT=)" >&2; exit 1; }

launch_server "$LOG" env PATH="$PWD/.venv-dflash/bin:$PATH" .venv-dflash/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$ARTIFACT" \
  --served-model-name mini \
  --port "$PORT" --host 127.0.0.1 --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len 8192 --enforce-eager
wait_healthy "$PORT" 900 240 || finish_fail "stage=health"

log_event SMOKE_QUERY stage=generate
RESP=$(curl -s -m 120 localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mini", "max_tokens": 24,
  "messages": [{"role": "user", "content": "Say OK."}]
}')
echo "$RESP" | head -c 600; echo
stop_server
CONTENT=$(echo "$RESP" | .venv-quant/bin/python -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null || true)
if [[ -n "${CONTENT// }" ]]; then
  finish_ok "artifact=$ARTIFACT preview=$(echo "$CONTENT" | head -c 60)"
else
  finish_fail "stage=generate resp=$(echo "$RESP" | head -c 300)"
fi
