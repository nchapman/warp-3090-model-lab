#!/bin/bash
# pod.sh — minimal local(Spark) -> pod(B200/B300) workflow helper.
# Everything is tested locally first (regression.sh), then pushed and re-run
# with identical scripts on a rented Blackwell pod.
#
#   POD=ssh://root@host  (required; or POD_HOST=host [POD_PORT=port])
#   bash pod.sh push                       # repo + slices + draft init -> pod
#   bash pod.sh run 'bash regression.sh'   # run anything in the repo dir
#   bash pod.sh pull results/ artifacts/   # pull outputs back (with sha256)
#   bash pod.sh status                     # GPU util + tail of recent logs
set -eu
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [ -n "${POD:-}" ]; then
  DEST=${POD#ssh://}; PORT=""; HOST=${DEST%%:*}
  [ "$DEST" != "$HOST" ] && PORT="-p ${DEST##*:}"
elif [ -n "${POD_HOST:-}" ]; then
  HOST=$POD_HOST; PORT=${POD_PORT:+-p $POD_PORT}
else
  echo "set POD=ssh://root@host[:port] (RunPod: ssh root@<ip> -p <port>)" >&2; exit 1
fi
SSH="ssh -o ServerAliveInterval=30 $PORT root@$HOST"
REMOTE_DIR=${POD_DIR:-/root/warp-3090-quant-lab}

case ${1:-status} in
push)
  echo "[pod] rsync repo (git-tracked files + slices + inits) -> $HOST:$REMOTE_DIR"
  cd "$ROOT"
  tar -C "$ROOT" --exclude-vcs --ignore-failed-read --exclude=.venv-dflash \
      --exclude=data-raw --exclude='Qwen*' --exclude=eval_* --exclude=ds_* \
      -cf - $(git ls-files) slices draft_init 2>/dev/null \
    | $SSH "mkdir -p $REMOTE_DIR && tar -C $REMOTE_DIR -xf -"
  echo "[pod] pushed. venv on pod: python3 -m venv venv && venv/bin/pip install -r requirements-drafter.txt"
  ;;
run)
  shift; $SSH "cd $REMOTE_DIR && setsid nohup $* > /root/pod_task.log 2>&1 & echo started"
  echo "[pod] launched; logs at $REMOTE_DIR / /root/pod_task.log — poll with: bash pod.sh status"
  ;;
pull)
  shift
  for p in "$@"; do
    echo "[pod] pulling $p (sha256 verified)"
    $SSH "cd $REMOTE_DIR && tar -cf - $p" | tar -C "$ROOT" -xf -
  done
  ;;
status)
  $SSH "nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader; \
        cd $REMOTE_DIR 2>/dev/null && ls -t *.log 2>/dev/null | head -3 | xargs -r -I{} sh -c 'echo --- {}; tail -2 {}'"
  ;;
*)
  echo "usage: bash pod.sh {push|run '<cmd>'|pull <paths...>|status}   (set POD=ssh://root@host[:port])"
  ;;
esac
