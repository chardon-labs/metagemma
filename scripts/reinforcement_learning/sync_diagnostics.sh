#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# shellcheck source=scripts/remote/remote_config.sh
source "$REPO_DIR/scripts/remote/remote_config.sh"
require_remote_connection_config

LOCAL_DIAGNOSTICS_ROOT="$REPO_DIR/data/reinforcement_learning/diagnostics"
REMOTE_CANDIDATE_ROOTS=(
  "${REMOTE_DIR%/}/reinforcement_learning/outputs"
  "${REMOTE_DIR%/}/outputs"
)

SSH_ARGS=(
  -i "$SSH_KEY"
  -o IdentitiesOnly=yes
  -p "$REMOTE_PORT"
)

RSYNC_SSH="ssh -i '$SSH_KEY' -o IdentitiesOnly=yes -p '$REMOTE_PORT'"

mkdir -p "$LOCAL_DIAGNOSTICS_ROOT"

synced_any=0
for remote_output_root in "${REMOTE_CANDIDATE_ROOTS[@]}"; do
  if ! ssh "${SSH_ARGS[@]}" "$REMOTE_HOST" "test -d '$remote_output_root'"; then
    continue
  fi

  echo "Syncing diagnostics from $remote_output_root"
  rsync -az --progress \
    --include='*/' \
    --include='logs/***' \
    --exclude='*' \
    -e "$RSYNC_SSH" \
    "$REMOTE_HOST:$remote_output_root/" \
    "$LOCAL_DIAGNOSTICS_ROOT/"
  synced_any=1
done

if [[ "$synced_any" -eq 0 ]]; then
  echo "No remote diagnostics output roots found." >&2
  exit 1
fi

echo "Synced remote diagnostics into $LOCAL_DIAGNOSTICS_ROOT"
