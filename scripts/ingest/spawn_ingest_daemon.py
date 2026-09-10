#!/usr/bin/env python3
"""
Detach the ingestion supervisor from whatever session started it.

``nohup cmd &`` is not enough. It ignores SIGHUP, but the process stays in the
starting shell's process group and session, so anything that tears down that
group — a terminal closing, an editor exiting, an assistant session being cut
off — can still take the ingestion with it. A backfill that runs for a day must
not be a child of a session that lasts an hour.

So this performs the standard double-fork: fork, ``setsid`` to become a session
leader with no controlling terminal, fork again so the process can never reacquire
one, then redirect the standard streams to a log file and exec the supervisor.
The result is reparented to init and is genuinely independent of its caller.

For restart-on-crash and restart-at-login as well, install the launchd agent
instead — see ``scripts/ops/manage_ingest_daemon.sh install``. This script is the
no-privileges fallback that covers session independence only.

Usage:
    python scripts/ingest/spawn_ingest_daemon.py
    python scripts/ingest/spawn_ingest_daemon.py --waves "2 3 4" --rate 600
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAEMON = ROOT / "scripts" / "ingest" / "ingest_daemon.sh"
LOG = ROOT / "runtime" / "logs" / "ingest_daemon.log"


def already_running() -> bool:
    """Whether an ingester or supervisor is already alive."""
    for pattern in ("ingest_fmp.py", "ingest_daemon.sh"):
        result = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True, check=False
        )
        if result.stdout.strip():
            return True
    return False


def main() -> int:
    """Double-fork the supervisor into its own session and return its pid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waves", default="1 2 3 4")
    parser.add_argument("--rate", default="600")
    parser.add_argument("--workers", default="16")
    parser.add_argument("--force", action="store_true", help="spawn even if one is running")
    args = parser.parse_args()

    if already_running() and not args.force:
        print("an ingester or supervisor is already running; not starting a second")
        print("(the rate limiter is per-process, so two would double the call rate)")
        return 1

    LOG.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "INGEST_WAVES": args.waves,
        "INGEST_RATE": args.rate,
        "INGEST_WORKERS": args.workers,
    }

    read_fd, write_fd = os.pipe()

    if os.fork() > 0:
        # Original process: wait for the grandchild's pid, then return.
        os.close(write_fd)
        with os.fdopen(read_fd) as reader:
            pid = reader.read().strip()
        os.wait()
        print(f"ingestion supervisor detached as pid {pid}")
        print(f"logs: {LOG}")
        print("it is now independent of this session; check with scripts/ingest/ingest_status.py")
        return 0

    # Child: become a session leader so there is no controlling terminal.
    os.close(read_fd)
    os.setsid()

    if os.fork() > 0:
        os._exit(0)  # noqa: SLF001 - intermediate parent must not run atexit handlers

    # Grandchild: fully detached. Redirect the standard streams and exec.
    with os.fdopen(write_fd, "w") as writer:
        writer.write(str(os.getpid()))

    os.chdir(ROOT)
    with open(os.devnull, "rb", 0) as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())
    with open(LOG, "ab", 0) as log:
        os.dup2(log.fileno(), sys.stdout.fileno())
        os.dup2(log.fileno(), sys.stderr.fileno())

    os.execve("/bin/bash", ["/bin/bash", str(DAEMON)], environment)


if __name__ == "__main__":
    raise SystemExit(main())
