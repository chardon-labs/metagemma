#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=scripts/remote_config.sh
source "$SCRIPT_DIR/remote_config.sh"

LOCAL_ARTIFACT_ROOT="$REPO_DIR/data"
REMOTE_TRACE_DIR="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["trace_dir"])' "$REPO_DIR/project_settings.json")"
REMOTE_OUTPUT_DIR="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["output_dir"])' "$REPO_DIR/project_settings.json")"

SSH_ARGS=(
  -i "$SSH_KEY"
  -o IdentitiesOnly=yes
  -p "$REMOTE_PORT"
)

RSYNC_SSH="ssh -i '$SSH_KEY' -o IdentitiesOnly=yes -p '$REMOTE_PORT'"

sync_artifact_dir() {
  local remote_relative_dir="$1"
  local local_dir="$LOCAL_ARTIFACT_ROOT/$remote_relative_dir"
  local remote_dir="${REMOTE_DIR%/}/$remote_relative_dir"

  mkdir -p "$local_dir"
  ssh "${SSH_ARGS[@]}" "$REMOTE_HOST" "test -d '$remote_dir'"
  rsync -az --progress \
    -e "$RSYNC_SSH" \
    "$REMOTE_HOST:$remote_dir/" \
    "$local_dir/"
}

sync_artifact_dir "$REMOTE_TRACE_DIR"
sync_artifact_dir "$REMOTE_OUTPUT_DIR"

echo "Synced remote artifacts into $LOCAL_ARTIFACT_ROOT"
