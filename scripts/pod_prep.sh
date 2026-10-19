#!/usr/bin/env bash
# pod_prep.sh — standard prep for a fresh GPU pod (run from the LOCAL repo).
#
# Encodes the day-1 sequence so it is identical every time:
#   survey  : GPU / CUDA / disk / python gates
#   token   : ship the HF token quietly (stdin, never argv) — prefer
#             HF_TOKEN from ~/.config/secrets.env, fall back to the stored
#             CLI token ~/.cache/huggingface/token (huggingface-cli login).
#             Installed BOTH as /root/.cache/huggingface/token (what
#             huggingface_hub/datasets auto-read in ANY process — gated
#             datasets, private repos) and /root/.hub_env.sh via .bashrc
#             (interactive shells). Non-interactive SSH (bash without
#             SSH_SOURCE_BASHRC, i.e. Debian/Ubuntu) sources neither .bashrc
#             nor hub_env — the stored-token file is the one that counts.
#   sync    : rsync the repo EXCLUDING disposable testbed artifacts, behind a
#             dry-run size guard (the 281GB smoke/ lesson: refuse > MAX_SYNC_GB
#             of transfer without FORCE=1, and name the offenders)
#   verify  : sha256sum -c data/MANIFEST.sha256 — frozen data is load-bearing
#   venv    : .venv-dflash from requirements-drafter.txt (skipped if importable)
#   patches : speculators fp8_load_cast + draft_opd, idempotent
#   check   : torch device capability + matmul on the pod GPU
#   models  : hf download $MODELS (default: the 4B campaign set)
#
# Usage:  scripts/pod_prep.sh <user@host> [port]
# Env:    KEY (default ~/.ssh/id_ed25519)
#         REPO_DIR (default /root/warp-3090-quant-lab)
#         MODELS (space-separated hub ids)   SKIP_MODELS=1 to skip
#         MAX_SYNC_GB (default 10)           FORCE=1 to override the size guard
#         SKIP_TOKEN=1 if the pod must not get the HF token
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)

DEST=${1:?usage: pod_prep.sh <user@host> [port]}
PORT=${2:-22}
KEY=${KEY:-$HOME/.ssh/id_ed25519}
REPO_DIR=${REPO_DIR:-/root/warp-3090-quant-lab}
MAX_SYNC_GB=${MAX_SYNC_GB:-10}
MODELS=${MODELS:-"Qwen/Qwen3.5-4B nchapman/Qwen3.5-4B-W4A16-GPTQ nchapman/Qwen3.5-4B-DFlash2"}

SSH() { ssh -p "$PORT" -i "$KEY" -o LogLevel=ERROR "$DEST" "$@"; }
stage() { printf '\n== %s (%s)\n' "$1" "$(date +%H:%M:%S)"; }

# Everything here is either disposable output, machine-local, or shipped by a
# different channel: venvs, the pod's run registry, git objects, caches,
# 0.8B smoke artifacts (217GB!), old eval scratch, local quant dirs (models
# come from the Hub), and the raw cascade pool (data-raw/, local-only input
# to slice_convs.py / build_calib.py — too big for the sync guard).
# NOTE: slices/ and draft_init/ are EXCLUDED here too — the drafter flow's
# payload ships via `bash pod.sh push`, NOT via pod_prep sync; the verify
# stage below warns when they are missing.
EXCLUDES=(.venv-* runs .git __pycache__ .pi
          .pytest_cache .mypy_cache smoke 'eval_*' 'Qwen3.5-*-GPTQ*'
          '*.pyc' .DS_Store quants data-raw slices)
RSYNC_EXCLUDES=()
for x in "${EXCLUDES[@]}"; do RSYNC_EXCLUDES+=(--exclude "$x"); done

stage "survey"
SSH 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
      || { echo "NO GPU VISIBLE — is this a GPU pod?"; exit 1; }
     df -h / | tail -1; python3 -V; nproc; free -g | awk "NR==2{print \$2\" GB ram\"}"'

stage "token"
if [[ ${SKIP_TOKEN:-0} != 1 ]]; then
  # Resolve locally: secrets.env first, stored CLI token as fallback.
  TOKEN_SRC="" TOKEN_VAL=""
  if grep -q '^HF_TOKEN=' "$HOME/.config/secrets.env" 2>/dev/null; then
    TOKEN_SRC="secrets.env"
    TOKEN_VAL=$(grep '^HF_TOKEN=' "$HOME/.config/secrets.env" | head -1 | cut -d= -f2-)
  elif [[ -s "$HOME/.cache/huggingface/token" ]]; then
    TOKEN_SRC="stored CLI token (~/.cache/huggingface/token)"
    TOKEN_VAL=$(tr -d '[:space:]' < "$HOME/.cache/huggingface/token")
  fi
  if [[ -n "$TOKEN_VAL" ]]; then
    # stdin, not argv; value never echoed into any log. Bare value goes over
    # the wire; the remote side derives both the stored-token file (root of
    # huggingface_hub auth for every python process) and hub_env.sh.
    printf '%s' "$TOKEN_VAL" | SSH 'umask 177; T=$(cat)
        mkdir -p /root/.cache/huggingface
        printf "%s" "$T" > /root/.cache/huggingface/token
        printf "export HF_TOKEN=%s\n" "$T" > /root/.hub_env.sh
        touch /root/.bashrc
        grep -q hub_env /root/.bashrc || sed -i "1i source /root/.hub_env.sh" /root/.bashrc' \
      && echo "token installed from ${TOKEN_SRC} (/root/.cache/huggingface/token + hub_env.sh via .bashrc)"
  else
    echo "WARN: no HF_TOKEN in ~/.config/secrets.env and no stored CLI token at ~/.cache/huggingface/token — gated/private pulls will fail"
  fi
fi

stage "sync"
# dry-run first: refuse surprise gigabytes before they cost 30 minutes
DRY=$(rsync -azn --stats "${RSYNC_EXCLUDES[@]}" \
        -e "ssh -p $PORT -i $KEY" "$ROOT/" "$DEST:$REPO_DIR/")
BYTES=$(awk '/^Total transferred file size:/{gsub(/,/,"",$5); print $5}' <<<"$DRY")
GB=$(( ${BYTES:-0} / 1024 / 1024 / 1024 ))
echo "transfer: ${GB}GB (guard: ${MAX_SYNC_GB}GB)"
if (( GB > MAX_SYNC_GB )) && [[ ${FORCE:-0} != 1 ]]; then
  echo "REFUSING: ${GB}GB exceeds MAX_SYNC_GB — largest local dirs:"
  du -sh --exclude='.venv-*' --exclude=.git \
     --exclude=runs "$ROOT"/* "$ROOT"/.[!.]* 2>/dev/null | sort -rh | head -8
  echo "fix the excludes or FORCE=1"
  exit 2
fi
rsync -az "${RSYNC_EXCLUDES[@]}" -e "ssh -p $PORT -i $KEY" \
  "$ROOT/" "$DEST:$REPO_DIR/"
echo "rsync done"

stage "verify"
SSH "cd $REPO_DIR/data && sha256sum -c MANIFEST.sha256 >/dev/null \
     && echo 'frozen data manifest OK' \
     || { echo 'FROZEN DATA DRIFT — do not run campaigns'; exit 1; }"
# slices/ is pod_prep-excluded (size guard) and pod.sh push is its channel —
# flag its absence so a drafter pod doesn't discover the gap mid-run.
# draft_init rides in the same tar; check both.
SSH "cd $REPO_DIR && for d in slices draft_init; do \
       [ -e \"\$d\" ] || echo \"  WARN: \$d/ absent — drafter flow needs it; run slice_convs/bootstrap_draft locally and 'bash pod.sh push'\"; done"

stage "venv"
SSH "cd $REPO_DIR && { .venv-dflash/bin/python -c 'import torch, speculators, vllm, lm_eval, langdetect, immutabledict, math_verify, latex2sympy2_extended; from importlib.metadata import version; assert version(\"lm_eval\") == \"0.4.13\"; assert version(\"antlr4-python3-runtime\").startswith(\"4.11\")' \
        2>/dev/null && echo 'venv already importable — skipping pip'; } \
     || { python3 -m venv .venv-dflash \
          && .venv-dflash/bin/pip install -q -U pip \
          && .venv-dflash/bin/pip install -q -r requirements-drafter.txt \
          && echo 'venv built'; }"
# quant venv (torch 2.14 / llmcompressor) — separate interpreter, needed by
# run_quant.sh; skipped when already importable. Not part of the drafter
# venv above (2026-10-19: the trial pod silently lacked it and every
# run_quant died at exec).
SSH "cd $REPO_DIR && { .venv-quant/bin/python -c 'import torch, llmcompressor, transformers, gptqmodel' 2>/dev/null \
        && echo 'quant venv already importable — skipping pip'; } \
     || { python3 -m venv .venv-quant \
          && .venv-quant/bin/pip install -q -U pip \
          && .venv-quant/bin/pip install -q -r requirements-quant.txt \
          && echo 'quant venv built'; }"

stage "patches"
# ORDER MATTERS: draft_opd's train/data.py hunk is cut against the
# fp8-patched file — apply fp8_load_cast first, draft_opd second.
PATCHES="patches/speculators_fp8_load_cast.patch patches/speculators_draft_opd.patch"
SSH "cd $REPO_DIR && SP=\$(.venv-dflash/bin/python -c \
      'import speculators,os;print(os.path.dirname(os.path.dirname(speculators.__file__)))') \
     && .venv-dflash/bin/pip install -q --force-reinstall --no-deps speculators==0.8.1 \
     && find \"\$SP\" -name '*.rej' -o -name '*.orig' | xargs -r rm -f \
     && for p in $PATCHES; do \
          patch -p1 -d \"\$SP\" < \"\$p\" >/dev/null \
          || { echo \"  FAILED \$p\"; exit 1; }; \
          echo \"  \$p ok\"; done \
     && .venv-dflash/bin/python -c 'import speculators.train.data, \
        speculators.models.dflash2.core, speculators.losses.utils' \
     && echo 'patched modules import ok'"

stage "check"
SSH "cd $REPO_DIR && .venv-dflash/bin/python -c 'import torch; cc = torch.cuda.get_device_capability(0); print(torch.__version__, cc); x = torch.randn(4096, 4096, device=\"cuda\", dtype=torch.bfloat16); (x @ x).sum().item(); torch.cuda.synchronize(); print(\"matmul ok\")' \
     || { echo 'TORCH/GPU CHECK FAILED'; exit 1; }"

stage "models"
if [[ ${SKIP_MODELS:-0} != 1 ]]; then
  SSH "source /root/.hub_env.sh 2>/dev/null; cd $REPO_DIR && rc=0; \
       for m in $MODELS; do \
         .venv-dflash/bin/hf download \"\$m\" --exclude '*.pth' >/dev/null 2>&1 \
         && echo \"  pulled \$m\" || { echo \"  FAILED \$m\"; rc=1; }; done; exit \$rc"
else
  echo "skipped (SKIP_MODELS=1)"
fi

stage "done"
echo "pod ready: $DEST -> $REPO_DIR"
echo "next: re-anchor baselines on THIS machine before comparing anything"
