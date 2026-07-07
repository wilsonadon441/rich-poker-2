#!/bin/bash
# Poker44 miner startup wrapper.
#
# The canonical way to run this miner is the .env-driven pm2 ecosystem:
#   pm2 start scripts/miner/ecosystem.config.cjs && pm2 save
# This wrapper simply delegates to it so older docs keep working.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

if ! command -v pm2 &> /dev/null; then
    echo "Error: PM2 is not installed"
    exit 1
fi
if [ ! -f .env ]; then
    echo "Error: $REPO/.env not found — configure wallet/port/repo settings first"
    exit 1
fi

pm2 start scripts/miner/ecosystem.config.cjs
pm2 save

PM2_NAME=$(grep -E '^POKER44_PM2_NAME=' .env | cut -d= -f2- | tr -d '"' || true)
echo "Miner started via ecosystem config (pm2 name: ${PM2_NAME:-see pm2 list})"
echo "View logs: pm2 logs ${PM2_NAME:-}"
