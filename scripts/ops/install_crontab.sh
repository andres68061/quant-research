#!/usr/bin/env bash
# Render scripts/ops/crontab.txt for this repo's location and install it.
# Usage: bash scripts/ops/install_crontab.sh        (re-run after moving the repo)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sed "s|__REPO__|$REPO|g" "$REPO/scripts/ops/crontab.txt" | crontab -
echo "installed crontab for $REPO:"; crontab -l | grep -vE '^\s*(#|$)'
