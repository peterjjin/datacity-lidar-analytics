#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTOWARE_CORE_ROOT="${AUTOWARE_CORE_ROOT:-$HOME/autoware_core_workspace/src/autoware_core}"
PATCH_FILE="$REPO_ROOT/autoware_custom/patches/autoware_core_lanelet2_map_visualizer.patch"

if [[ ! -d "$AUTOWARE_CORE_ROOT/.git" ]]; then
  echo "missing autoware_core git checkout: $AUTOWARE_CORE_ROOT" >&2
  exit 2
fi

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "missing patch file: $PATCH_FILE" >&2
  exit 2
fi

git -C "$AUTOWARE_CORE_ROOT" apply "$PATCH_FILE"
echo "applied $PATCH_FILE to $AUTOWARE_CORE_ROOT"
