#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="$repo_root/.env"
overlay_file="${1:-}"

if [[ ! -f "$env_file" ]]; then
  echo "Missing $env_file. Copy .env.example to .env and add your Modal credentials." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

if [[ -n "$overlay_file" ]]; then
  if [[ "$overlay_file" != /* ]]; then
    overlay_file="$repo_root/$overlay_file"
  fi
  if [[ ! -f "$overlay_file" ]]; then
    echo "Missing deployment overlay: $overlay_file" >&2
    exit 1
  fi

  set -a
  # shellcheck disable=SC1090
  source "$overlay_file"
  set +a
fi

# Empty values would override credentials already stored in ~/.modal.toml.
for variable in MODAL_TOKEN_ID MODAL_TOKEN_SECRET; do
  if [[ -z "${!variable:-}" ]]; then
    unset "$variable"
  fi
done

exec modal deploy "$repo_root/scripts/modal_openpi.py"
