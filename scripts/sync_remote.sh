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

Sync this repository to the remote /workspace/ directory, excluding .git, .venv, and scripts.

Options:
  --delete   Delete remote files that do not exist locally.
  --dry-run  Show what would be synced without changing the remote.

Environment overrides are defined in scripts/remote_config.sh.
USAGE
}

RSYNC_ARGS=(-az --progress --exclude='.git/' --exclude='.venv/' --exclude='scripts/' --exclude='__pycache__/')

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
