#!/usr/bin/env bash
# Incremental ingestion → clean → features on EC2 (self-hosted runner).
# Used by .github/workflows/data-daily.yml. See docs/DEPLOY_AWS_GITHUB.md Step 10.

set -euo pipefail

APP_DIR="${APP_DIR:-/opt/ai-finance/app}"
GIT_BRANCH="${GIT_BRANCH:-main}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
INGESTION_UNIVERSE="${INGESTION_UNIVERSE:-sp500}"
LOG_FILE="${LOG_FILE:-/tmp/data-daily.log}"
FEATURES_STATE_FILE="${FEATURES_STATE_FILE:-/tmp/features-daily.state}"
VENV="${APP_DIR}/.venv/bin/activate"

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "DATABASE_URL is required" >&2
  exit 1
fi

if [[ ! -d "$APP_DIR/.git" ]]; then
  echo "APP_DIR is not a git repo: $APP_DIR" >&2
  exit 1
fi

if [[ ! -f "$VENV" ]]; then
  echo "Python venv not found: $VENV" >&2
  exit 1
fi

exec > "$LOG_FILE" 2>&1
echo "==> Daily pipeline started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "==> INGESTION_UNIVERSE=$INGESTION_UNIVERSE"

echo "==> Syncing $APP_DIR to $GIT_REMOTE/$GIT_BRANCH"
cd "$APP_DIR"
git fetch "$GIT_REMOTE" "$GIT_BRANCH"
git reset --hard "$GIT_REMOTE/$GIT_BRANCH"

# shellcheck disable=SC1090
source "$VENV"
export PYTHONPATH=src
export DATABASE_URL
export INGESTION_UNIVERSE
export FEATURES_STATE_FILE
unset FEATURES_BACKFILL FEATURES_START_AT

echo "==> make ingestion"
make ingestion

echo "==> make clean"
make clean

echo "==> make features"
make features

echo "==> make universe-preflight"
make universe-preflight

echo "==> Daily pipeline OK at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
