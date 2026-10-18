#!/usr/bin/env python3
"""podsh.py — run commands on a RunPod pod through the ssh.runpod.io console.

RunPod's SSH gateway is a console, not an exec channel: it demands a PTY,
ignores exec requests, and offers no sftp. This helper drives the PROVEN
OpenSSH path (ssh -tt, system client, registered account key) with pexpect:
wait for the root prompt, send the command, stream output until a unique
sentinel line arrives.

Usage:
  podsh.py 'command...' [--timeout 120] [--dest POD-USER@ssh.runpod.io]
  podsh.py --file script.sh
  POD_DEST=... podsh.py ...   (default dest: quant-arms-4b pod)
"""
import argparse
import os
import sys

import pexpect

SENTINEL = "__PODSH_DONE__"
DEFAULT_DEST = "rvu4jwezkpa20u-64411f5e@ssh.runpod.io"
PROMPT = r"root@[^\n]*[#$] "


def run(dest: str, command: str, timeout: float) -> int:
    child = pexpect.spawn(
        "ssh",
        ["-tt", "-i", os.path.expanduser("~/.ssh/id_ed25519"),
         "-o", "ControlMaster=no", "-o", "ControlPath=none",
         "-o", "LogLevel=ERROR", dest],
        encoding="utf-8", codec_errors="replace", timeout=30,
    )
    child.logfile_read = None
    try:
        idx = child.expect([PROMPT, pexpect.TIMEOUT, pexpect.EOF], timeout=40)
        if idx != 0:
            print(f"[podsh] no shell prompt (idx={idx})", file=sys.stderr)
            print(child.before or "", file=sys.stderr)
            return 2
        # disable echo + fancy prompt so output is parseable
        child.sendline("stty -echo; export TERM=dumb; PS1='PROMPT> '")
        child.expect_exact("PROMPT> ", timeout=15)
        child.sendline(f"{command}; echo {SENTINEL}:$?")
        out = []
        deadline = timeout
        child.timeout = deadline
        while True:
            idx = child.expect([SENTINEL, pexpect.TIMEOUT, pexpect.EOF])
            line = child.before or ""
            out.append(line)
            if idx == 0:
                tail = child.readline()  # ":<rc>\n"
                rc = tail.strip().lstrip(":")
                break
            if idx == 2:
                rc = "EOF"
                break
            # TIMEOUT inside command loop
            rc = "TIMEOUT"
            break
        text = "".join(out).replace("\r\n", "\n").replace("\r", "\n")
        lines = [l for l in text.split("\n")]
        # drop the echoed command (first physical line up to first newline after prompt)
        if lines and command[:40] in lines[0]:
            lines = lines[1:]
        while lines and lines[0].strip() == "":
            lines.pop(0)
        print("\n".join(lines))
        print(f"[podsh] rc={rc}")
        return 0 if rc == "0" else 1
    finally:
        try:
            child.sendline("exit")
            child.close(force=True)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?")
    ap.add_argument("--file")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--dest", default=os.environ.get("POD_DEST", DEFAULT_DEST))
    args = ap.parse_args()
    command = args.command
    if args.file:
        with open(args.file) as f:
            command = f.read()
    if not command:
        ap.error("need a command or --file")
    return run(args.dest, command, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
