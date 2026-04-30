#!/usr/bin/env bash

REMOTE_CONFIG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_INSTANCE_CONFIG="${REMOTE_INSTANCE_CONFIG:-$REMOTE_CONFIG_DIR/remote_instance.sh}"

if [[ -f "$REMOTE_INSTANCE_CONFIG" ]]; then
  # shellcheck source=/dev/null
  source "$REMOTE_INSTANCE_CONFIG"
fi

REMOTE_HOST="${REMOTE_HOST:-${VASTAI_REMOTE_HOST:-}}"
REMOTE_PORT="${REMOTE_PORT:-${VASTAI_REMOTE_PORT:-}}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_vastai}"
LOCAL_TUNNEL_PORT="${LOCAL_TUNNEL_PORT:-8010}"
REMOTE_TUNNEL_HOST="${REMOTE_TUNNEL_HOST:-localhost}"
REMOTE_TUNNEL_PORT="${REMOTE_TUNNEL_PORT:-8010}"

require_remote_connection_config() {
  if [[ -n "${REMOTE_HOST:-}" && -n "${REMOTE_PORT:-}" ]]; then
    return 0
  fi

  cat >&2 <<EOF
Missing Vast.ai SSH target.
Run ./scripts/update_remote_instance.sh to write $REMOTE_INSTANCE_CONFIG.
EOF
  return 1
}
