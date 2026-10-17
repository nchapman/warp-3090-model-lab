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

# GB10 unified-memory hygiene (see mem_guard in runlib.sh + PAGE_CACHE_EVICT
# in quantize.py): cap allocator fragmentation, fence the job so the desktop
# survives a budget miss.
# 2026-10-17 (v2): expandable_segments retired; max_split_size_mb:64 pins it.
# Allocation >64MB never splits and gets a DEDICATED segment; when the load
# transient (~29GB observed at 35B) frees, empty_cache() can release that
# segment whole. Without it the free block co-resides with live tensors and
# reserved stays pinned at 65.4GB regardless of allocator flavor
# (measured: runs 5+6, released 0.0GB while gap was 20-34GB).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}
export MEM_GUARD_MB=${MEM_GUARD_MB:-6144}
# transformers 5.x async load: 4 reader threads pile completed CPU tensors
# into converter futures faster than the convert loop consumes them -> ~2x
# model footprint; without this the 35B load OOMs the box at ~45% (2026-10-17).
export HF_DEACTIVATE_ASYNC_LOAD=${HF_DEACTIVATE_ASYNC_LOAD:-1}

run_with_watchdog "$RUN_DIR/quant.log" \
  "$VENV" "$(dirname "$0")/quantize.py" --model "$KEY" "$@" \
  || finish_fail "stage=quantize"

log_event QUANT_DONE "model=$KEY out=$RUN_DIR/quant.log"
finish_ok "model=$KEY"
