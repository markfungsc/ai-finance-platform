#!/usr/bin/env bash
# Deploy slim API on EC2: sync repo, rebuild image, restart container, smoke /health.
# Used by .github/workflows/deploy.yml (self-hosted runner). See docs/DEPLOY_AWS_GITHUB.md Step 11.

set -euo pipefail

APP_DIR="${APP_DIR:-/opt/ai-finance/app}"
MODELS_DIR="${MODELS_DIR:-/opt/ai-finance/models}"
CONTAINER_NAME="${CONTAINER_NAME:-finance-api}"
IMAGE_TAG="${IMAGE_TAG:-ai-finance-api:slim}"
GIT_BRANCH="${GIT_BRANCH:-main}"
GIT_REMOTE="${GIT_REMOTE:-origin}"

MODEL_PATH="${MODEL_PATH:-/app/models/swing-trade/random_forest_pooled.pkl}"
FEATURE_COLUMNS_PATH="${FEATURE_COLUMNS_PATH:-/app/models/swing-trade/random_forest_pooled_feature_columns.pkl}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"
HEALTH_RETRIES="${HEALTH_RETRIES:-30}"
HEALTH_SLEEP_SECONDS="${HEALTH_SLEEP_SECONDS:-2}"

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "DATABASE_URL is required" >&2
  exit 1
fi

if [[ ! -d "$APP_DIR/.git" ]]; then
  echo "APP_DIR is not a git repo: $APP_DIR" >&2
  exit 1
fi

echo "==> Syncing $APP_DIR to $GIT_REMOTE/$GIT_BRANCH"
cd "$APP_DIR"
git fetch "$GIT_REMOTE" "$GIT_BRANCH"
git reset --hard "$GIT_REMOTE/$GIT_BRANCH"

echo "==> Building Docker image $IMAGE_TAG"
docker build -f infra/Dockerfile.api.slim -t "$IMAGE_TAG" .

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "==> Stopping existing container $CONTAINER_NAME"
  docker stop "$CONTAINER_NAME" >/dev/null || true
  docker rm "$CONTAINER_NAME" >/dev/null || true
fi

echo "==> Starting container $CONTAINER_NAME"
docker run -d --name "$CONTAINER_NAME" --restart unless-stopped \
  -p 8000:8000 \
  -e DATABASE_URL="$DATABASE_URL" \
  -e MODEL_PATH="$MODEL_PATH" \
  -e FEATURE_COLUMNS_PATH="$FEATURE_COLUMNS_PATH" \
  -v "$MODELS_DIR:/app/models:ro" \
  "$IMAGE_TAG"

echo "==> Waiting for $HEALTH_URL"
for ((i = 1; i <= HEALTH_RETRIES; i++)); do
  if curl -sf "$HEALTH_URL" >/dev/null; then
    echo "==> Deploy OK (attempt $i/$HEALTH_RETRIES)"
    curl -sf "$HEALTH_URL"
    echo
    exit 0
  fi
  sleep "$HEALTH_SLEEP_SECONDS"
done

echo "Health check failed after $HEALTH_RETRIES attempts" >&2
docker logs --tail 50 "$CONTAINER_NAME" >&2 || true
exit 1
