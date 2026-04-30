#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=scripts/remote_config.sh
source "$SCRIPT_DIR/remote_config.sh"

DELETE_ARGS=()

usage() {
  cat <<'USAGE'
Usage: ./scripts/sync_remote.sh [--delete] [--dry-run]

Sync this repository to the remote /workspace/ directory, excluding local
metadata, dependency directories, caches, generated outputs, and scripts.

Options:
  --delete   Delete remote files that do not exist locally.
  --dry-run  Show what would be synced without changing the remote.

Environment overrides are defined in scripts/remote_config.sh.
USAGE
}

EXCLUDES=(
  # Repository/local-agent metadata.
  '.git/'
  '.codex'
  '.DS_Store'

  # Local secrets and environment files.
  '.env'
  '.env.*'

  # Dependency installs and virtual environments.
  '.venv/'
  'node_modules/'

  # Python and tool caches.
  '__pycache__/'
  '*.py[cod]'
  '.pytest_cache/'
  '.ruff_cache/'
  '.mypy_cache/'
  '.ty/'
  '.cache/'
  '.ipynb_checkpoints/'

  # Build outputs.
  'dist/'
  'build/'
  '*.egg-info/'

  # Generated data, outputs, and vendored repos/artifacts.
  'data/'
  'outputs/'
  'repos/'
  'wandb/'
  'lightning_logs/'
  'checkpoints/'
  'runs/'

  # This local helper directory should not be mirrored to the remote workspace.
  'scripts/'
)

RSYNC_ARGS=(-az --progress)
for exclude in "${EXCLUDES[@]}"; do
  RSYNC_ARGS+=(--exclude="$exclude")
done

while (($#)); do
  case "$1" in
  --delete)
    DELETE_ARGS=(--delete)
    ;;
  --dry-run)
    RSYNC_ARGS+=(--dry-run)
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown argument: $1" >&2
    usage >&2
    exit 2
    ;;
  esac
  shift
done

require_remote_connection_config

SSH_ARGS=(
  -i "$SSH_KEY"
  -o IdentitiesOnly=yes
  -p "$REMOTE_PORT"
)

ssh "${SSH_ARGS[@]}" "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"
rsync "${RSYNC_ARGS[@]}" "${DELETE_ARGS[@]}" \
  -e "ssh -i '$SSH_KEY' -o IdentitiesOnly=yes -p '$REMOTE_PORT'" \
  "$REPO_DIR/" \
  "$REMOTE_HOST:$REMOTE_DIR"
