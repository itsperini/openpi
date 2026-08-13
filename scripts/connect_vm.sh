#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${1:-$repo_root/.env.vm}"

if [[ ! -f "$env_file" ]]; then
  echo "Missing $env_file. Copy .env.vm.example to .env.vm and configure the GPU VM." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

for variable in OPENPI_VM_SSH_HOST OPENPI_VM_SSH_USER OPENPI_VM_IDENTITY_FILE; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing $variable in $env_file." >&2
    exit 1
  fi
done

local_host="${OPENPI_VM_LOCAL_HOST:-127.0.0.1}"
local_port="${OPENPI_VM_LOCAL_PORT:-8000}"
remote_port="${OPENPI_VM_REMOTE_PORT:-8000}"
identity_file="${OPENPI_VM_IDENTITY_FILE/#\~/$HOME}"

if [[ ! -f "$identity_file" ]]; then
  echo "SSH identity file not found: $identity_file" >&2
  exit 1
fi

echo "Forwarding ws://$local_host:$local_port to $OPENPI_VM_SSH_HOST:$remote_port."
echo "Keep this terminal open while the Mac client is running."

exec ssh \
  -i "$identity_file" \
  -N \
  -L "$local_host:$local_port:127.0.0.1:$remote_port" \
  -o IdentitiesOnly=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  "$OPENPI_VM_SSH_USER@$OPENPI_VM_SSH_HOST"
