#!/usr/bin/env bash
# Install, start, stop and inspect the unattended FMP ingestion agent.
#
# The agent is a launchd LaunchAgent, which is the point: launchd owns the
# process, so ingestion keeps running when the terminal, editor, or assistant
# session that started it goes away, and it restarts after a crash or a reboot.
#
# Usage:
#   bash scripts/manage_ingest_daemon.sh install   # copy plist + load + start
#   bash scripts/manage_ingest_daemon.sh status    # is it running, how far along
#   bash scripts/manage_ingest_daemon.sh stop      # stop and unload
#   bash scripts/manage_ingest_daemon.sh restart
#   bash scripts/manage_ingest_daemon.sh logs      # tail the daemon log

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.quant.fmp-ingest"
SRC="$REPO/config/launchd/$LABEL.plist"
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

case "${1:-status}" in
  install)
    mkdir -p "$HOME/Library/LaunchAgents"
    # The plist is a template: render the repo's actual location into it, so the
    # agent works wherever the repo lives. Note macOS refuses to run agents from
    # TCC-protected folders (~/Downloads, ~/Desktop, ~/Documents); keep the repo
    # outside those.
    sed "s|__REPO__|$REPO|g" "$SRC" > "$DEST"
    # bootout first so install is idempotent; ignore "not loaded".
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
    launchctl bootstrap "$DOMAIN" "$DEST" || {
        echo "bootstrap failed; falling back to legacy load"
        launchctl load -w "$DEST"
    }
    launchctl enable "$DOMAIN/$LABEL" 2>/dev/null
    echo "installed and started $LABEL"
    echo "it will keep running across terminal and session exits, and restart at login"
    ;;
  stop)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || launchctl unload -w "$DEST" 2>/dev/null
    echo "stopped $LABEL (in-flight ingester, if any, is left to finish)"
    ;;
  restart)
    launchctl kickstart -k "$DOMAIN/$LABEL" && echo "restarted $LABEL"
    ;;
  logs)
    tail -n "${2:-40}" "$REPO/logs/ingest_daemon.log"
    ;;
  status)
    echo "== launchd =="
    launchctl print "$DOMAIN/$LABEL" 2>/dev/null \
      | grep -E "state = |pid = |last exit code = " \
      || echo "  not loaded (run: bash scripts/manage_ingest_daemon.sh install)"
    echo
    echo "== heartbeat =="
    cat "$REPO/data/quality/ingest_heartbeat.txt" 2>/dev/null || echo "  none yet"
    echo
    echo "== ingester process =="
    pgrep -fl "ingest_fmp.py" || echo "  no ingester running"
    ;;
  *)
    echo "usage: $0 {install|stop|restart|status|logs [n]}" >&2
    exit 2
    ;;
esac
