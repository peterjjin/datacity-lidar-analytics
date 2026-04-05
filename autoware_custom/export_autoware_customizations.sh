#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTOWARE_CORE_ROOT="${AUTOWARE_CORE_ROOT:-$HOME/autoware_core_workspace/src/autoware_core}"
PATCH_FILE="$REPO_ROOT/autoware_custom/patches/autoware_core_lanelet2_map_visualizer.patch"
TARGET_PATH="map/autoware_lanelet2_map_visualizer/src/lanelet2_map_visualization_node.cpp"

if [[ ! -d "$AUTOWARE_CORE_ROOT/.git" ]]; then
  echo "missing autoware_core git checkout: $AUTOWARE_CORE_ROOT" >&2
  exit 2
fi

mkdir -p "$(dirname "$PATCH_FILE")"
git -C "$AUTOWARE_CORE_ROOT" diff -- "$TARGET_PATH" > "$PATCH_FILE"

if [[ ! -s "$PATCH_FILE" ]]; then
  echo "no local diff found for $TARGET_PATH" >&2
  exit 1
fi

echo "wrote $PATCH_FILE"
