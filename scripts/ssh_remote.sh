#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=scripts/remote_config.sh
source "$SCRIPT_DIR/remote_config.sh"
require_remote_connection_config

ssh -i "$SSH_KEY" \
  -o IdentitiesOnly=yes \
  -p "$REMOTE_PORT" \
  "$REMOTE_HOST" \
  -L "$LOCAL_TUNNEL_PORT:$REMOTE_TUNNEL_HOST:$REMOTE_TUNNEL_PORT"
