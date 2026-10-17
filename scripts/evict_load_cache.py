"""Incremental page-cache evictor for GB10 unified memory.

During model load, the just-read safetensors shards accumulate in page cache
NEXT TO the resident model copy (~2x model footprint) and the post-load
eviction inside quantize.py hasn't fired yet. This loop drops clean cache
pages every 15s for the whole load window, capping the double-hold.

Exits when quantize.py's own PAGE_CACHE_EVICT event appears (load done) or
after a hard timeout (run died without it).

Usage: evict_load_cache.py <snapshot_dir> <events_jsonl> [timeout_s]

Exit codes:
  0  PAGE_CACHE_EVICT seen — load completed
  2  argument/usage error (bad args, missing dir/events file, no shards)
  3  timeout without PAGE_CACHE_EVICT — the load never finished

Per-shard IO errors are logged and skipped (a shard vanishing mid-load must
not kill the memory fence for the ones that remain); a systematically failing
posix_fadvise is reported once per path, not silently.
"""
import ctypes
import glob
import os
import subprocess
import sys
import time

FADV_DONTNEED = 4
POLL_INTERVAL_S = 15


def fail(msg):
    print(f"evict_load_cache: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def main():
    if not 3 <= len(sys.argv) <= 4:
        fail(f"usage: {sys.argv[0]} <snapshot_dir> <events_jsonl> [timeout_s]")
    snap, events = sys.argv[1], sys.argv[2]
    try:
        timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 3600
    except ValueError:
        fail(f"timeout must be integer seconds, got {sys.argv[3]!r}")
    if not os.path.isdir(snap):
        fail(f"snapshot dir not found: {snap}")
    if not os.path.isfile(events):
        fail(f"events file not found: {events}")
    shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    if not shards:
        fail(f"no *.safetensors under {snap} — wrong dir? refusing to spin "
             f"silently for {timeout}s")

    libc = ctypes.CDLL(None, use_errno=True)
    libc.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_long,
                                   ctypes.c_int]
    libc.posix_fadvise.restype = ctypes.c_int
    io_warned = set()

    deadline = time.time() + timeout
    while time.time() < deadline:
        done = subprocess.run(
            ["grep", "-q", "PAGE_CACHE_EVICT", events], capture_output=True
        )
        if done.returncode == 0:
            print("post-load eviction fired; exiting", flush=True)
            return
        if done.returncode >= 2:
            # grep itself failed (events file vanished/unreadable) — watching
            # for completion is now impossible; spinning to timeout pretending
            # otherwise is the failure mode this guard exists for
            fail(f"cannot read {events} (grep rc={done.returncode})")
        for p in shards:
            try:
                fd = os.open(p, os.O_RDONLY)
            except OSError as e:
                if p not in io_warned:
                    print(f"evict_load_cache: shard unreadable {p}: {e}",
                          file=sys.stderr, flush=True)
                    io_warned.add(p)
                continue
            try:
                rc = libc.posix_fadvise(fd, 0, 0, FADV_DONTNEED)
                if rc != 0 and p not in io_warned:
                    print(f"evict_load_cache: posix_fadvise({p}) -> {rc} "
                          f"(errno {ctypes.get_errno()})", file=sys.stderr,
                          flush=True)
                    io_warned.add(p)
            finally:
                os.close(fd)
        time.sleep(POLL_INTERVAL_S)
    print("timeout without PAGE_CACHE_EVICT; exiting", file=sys.stderr,
          flush=True)
    sys.exit(3)


if __name__ == "__main__":
    main()
