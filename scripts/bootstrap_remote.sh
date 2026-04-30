#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=scripts/remote_config.sh
source "$SCRIPT_DIR/remote_config.sh"
require_remote_connection_config

SSH_ARGS=(
  -i "$SSH_KEY"
  -o IdentitiesOnly=yes
  -p "$REMOTE_PORT"
)

printf -v REMOTE_DIR_Q "%q" "$REMOTE_DIR"

ssh "${SSH_ARGS[@]}" "$REMOTE_HOST" "REMOTE_DIR=$REMOTE_DIR_Q bash -s" <<'REMOTE_BOOTSTRAP'
set -euo pipefail

apt_get() {
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo apt-get "$@"
  else
    echo "apt-get needs root or sudo on this machine." >&2
    exit 1
  fi
}

export DEBIAN_FRONTEND=noninteractive

echo "Installing apt packages..."
apt_get update
apt_get install -y bubblewrap btop curl ca-certificates unzip

export BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
export PATH="$BUN_INSTALL/bin:$PATH"

if command -v bun >/dev/null 2>&1; then
  echo "bun already installed: $(bun --version)"
else
  echo "Installing bun..."
  curl -fsSL https://bun.sh/install | bash
  export PATH="$BUN_INSTALL/bin:$PATH"
  echo "bun installed: $(bun --version)"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed or not on PATH. This script assumes the CUDA container already provides uv." >&2
  exit 1
fi

if [[ ! -d "$REMOTE_DIR" ]]; then
  echo "Remote repo directory does not exist: $REMOTE_DIR" >&2
  echo "Sync the repo first with ./scripts/sync_remote.sh, then rerun bootstrap." >&2
  exit 1
fi

echo "Syncing uv environments under $REMOTE_DIR..."
mapfile -t PYPROJECTS < <(
  find "$REMOTE_DIR" \
    \( -path '*/.git' -o -path '*/.venv' -o -path '*/__pycache__' \) -prune \
    -o -name pyproject.toml -print | sort
)

if [[ "${#PYPROJECTS[@]}" -eq 0 ]]; then
  echo "No pyproject.toml files found under $REMOTE_DIR" >&2
  exit 1
fi

for PYPROJECT in "${PYPROJECTS[@]}"; do
  PROJECT_DIR="$(dirname "$PYPROJECT")"
  echo
  echo "uv sync: $PROJECT_DIR"
  (
    cd "$PROJECT_DIR"
    uv sync
  )
done

echo
echo "Bootstrap complete."
REMOTE_BOOTSTRAP
