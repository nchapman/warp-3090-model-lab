#!/bin/bash
# Quantize a registered model with the champion W4A16 GPTQ recipe
# (cascade-v1 calibration). Usage: quant/run_quant.sh <model-key> [extra args]
set -euo pipefail
source "$(dirname "$0")/../lib/runlib.sh"

KEY=${1:?usage: run_quant.sh <model-key> [extra args...]}
shift || true

run_init "quant-$KEY"
log_event QUANT_START "model=$KEY args=$*"
VENV="$(dirname "$0")/../.venv-quant/bin/python3"

run_with_watchdog "$RUN_DIR/quant.log" \
  "$VENV" "$(dirname "$0")/quantize.py" --model "$KEY" "$@" \
  || finish_fail "stage=quantize"

log_event QUANT_DONE "model=$KEY out=$RUN_DIR/quant.log"
finish_ok "model=$KEY"
