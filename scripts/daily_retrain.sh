#!/usr/bin/env bash
# Daily auto-retrain pipeline, driven entirely by the repo's .env.
#
# Flow: fetch newly released benchmark dates -> train this repo's variant on
# the refreshed corpus -> promote the candidate only if it beats the deployed
# artifact on the newest holdout -> restart the pm2 miner so the runtime
# manifest (artifact sha256, model version, training statement) is rebuilt.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
set -a
[ -f .env ] && . ./.env
set +a

PYTHON="$REPO/miner_env/bin/python"
export PYTHONPATH="$REPO"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_retrain.log"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_FILE"; }

exec 9>"$LOG_DIR/retrain.lock"
if ! flock -n 9; then
    log "Another retrain is already running; exiting."
    exit 0
fi

DEPLOY_PATH="${POKER44_MODEL_PATH:-$REPO/models/deploy.joblib}"
if [ "${FORCE_RETRAIN:-0}" != "1" ] && [ -f "$DEPLOY_PATH" ]; then
    AGE=$(( $(date +%s) - $(stat -c %Y "$DEPLOY_PATH") ))
    if [ "$AGE" -lt 21600 ]; then
        log "Deployed artifact is ${AGE}s old (<6h); skipping retrain."
        exit 0
    fi
fi

log "=== Daily retrain started (${POKER44_PM2_NAME:-unnamed}) ==="

log "Step 1/3: syncing benchmark data..."
if ! "$PYTHON" -m training.fetch_benchmark 2>&1 | tee -a "$LOG_FILE"; then
    log "WARNING: benchmark sync failed; training on existing local corpus."
fi

log "Step 2/3: training variant candidate..."
if ! "$PYTHON" -m training.train_variant 2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: training failed; keeping deployed model."
    exit 1
fi

log "Step 3/3: evaluating candidate vs deployed..."
"$PYTHON" -m training.promote_if_better 2>&1 | tee -a "$LOG_FILE"
PROMOTE_EXIT=${PIPESTATUS[0]}

if [ "$PROMOTE_EXIT" -eq 0 ]; then
    NEW_SHA=$(sha256sum "$DEPLOY_PATH" | cut -d' ' -f1)
    log "Promoted new artifact sha256=$NEW_SHA"
    PM2_BIN="$(command -v pm2 || echo /usr/local/bin/pm2)"
    if [ -n "${POKER44_PM2_NAME:-}" ] && "$PM2_BIN" describe "$POKER44_PM2_NAME" > /dev/null 2>&1; then
        log "Restarting pm2 process ${POKER44_PM2_NAME} to refresh the manifest..."
        # No --update-env: the app must keep the ecosystem-config env, where
        # empty .env values (e.g. POKER44_MODEL_REPO_COMMIT) are dropped so the
        # miner's git-HEAD fallback keeps the manifest compliant.
        "$PM2_BIN" restart "$POKER44_PM2_NAME" 2>&1 | tee -a "$LOG_FILE"
    else
        log "pm2 process ${POKER44_PM2_NAME:-<unset>} not found; start it with scripts/miner/ecosystem.config.cjs"
    fi
elif [ "$PROMOTE_EXIT" -eq 3 ]; then
    log "Candidate did not beat deployed model; nothing deployed."
else
    log "ERROR: promotion step failed with exit $PROMOTE_EXIT."
    exit 1
fi

log "=== Daily retrain complete ==="
