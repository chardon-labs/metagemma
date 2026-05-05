#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_CONFIG="$SCRIPT_DIR/remote_instance.sh"
TMP_CONFIG="$(mktemp "${INSTANCE_CONFIG}.tmp.XXXXXX")"

cleanup() {
  rm -f "$TMP_CONFIG"
}
trap cleanup EXIT

INSTANCES_JSON="$(vastai show instances --raw)"

VASTAI_INSTANCES_JSON="$INSTANCES_JSON" \
VASTAI_INSTANCE_CONFIG="$INSTANCE_CONFIG" \
python3 "$SCRIPT_DIR/update_remote_instance.py" "$TMP_CONFIG"

mv "$TMP_CONFIG" "$INSTANCE_CONFIG"
chmod 600 "$INSTANCE_CONFIG"

# shellcheck source=/dev/null
source "$INSTANCE_CONFIG"

echo "Wrote $INSTANCE_CONFIG"
echo "REMOTE_HOST=$VASTAI_REMOTE_HOST"
echo "REMOTE_PORT=$VASTAI_REMOTE_PORT"
