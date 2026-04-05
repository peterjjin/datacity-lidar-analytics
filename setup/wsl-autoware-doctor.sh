#!/usr/bin/env bash
set -u

AUTOWARE_FLAVOR="${AUTOWARE_FLAVOR:-core}"
AUTOWARE_WS="${AUTOWARE_WS:-$HOME/autoware_core_workspace}"
ROS_DISTRO="${ROS_DISTRO:-humble}"

usage() {
  cat <<'EOF'
Usage:
  bash setup/wsl-autoware-doctor.sh [--workspace PATH] [--flavor core|main] [--ros-distro NAME]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      AUTOWARE_WS="$2"
      shift 2
      ;;
    --flavor)
      AUTOWARE_FLAVOR="$2"
      shift 2
      ;;
    --ros-distro)
      ROS_DISTRO="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

PASS_COUNT=0
WARN_COUNT=0

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  echo "[pass] $*"
}

warn() {
  WARN_COUNT=$((WARN_COUNT + 1))
  echo "[warn] $*"
}

check_cmd() {
  local cmd="$1"
  if command -v "$cmd" >/dev/null 2>&1; then
    pass "command available: $cmd"
  else
    warn "command missing: $cmd"
  fi
}

echo "WSL doctor for Autoware"
echo "workspace: $AUTOWARE_WS"
echo "flavor: $AUTOWARE_FLAVOR"
echo "ros_distro: $ROS_DISTRO"
echo

check_cmd git
check_cmd rosdep
check_cmd colcon
check_cmd vcs

if [[ -d "/opt/ros/$ROS_DISTRO" ]]; then
  pass "ROS 2 present at /opt/ros/$ROS_DISTRO"
else
  warn "ROS 2 not found at /opt/ros/$ROS_DISTRO"
fi

if [[ -d "$AUTOWARE_WS/src" ]]; then
  pass "workspace src directory exists"
else
  warn "workspace src directory missing"
fi

case "$AUTOWARE_FLAVOR" in
  core)
    if [[ -d "$AUTOWARE_WS/src/autoware_core/.git" ]]; then
      pass "autoware_core repository present"
    else
      warn "autoware_core repository missing"
    fi
    ;;
  main)
    if [[ -d "$AUTOWARE_WS/src/autoware/.git" ]]; then
      pass "autoware repository present"
    else
      warn "autoware repository missing"
    fi
    ;;
  *)
    warn "unknown flavor: $AUTOWARE_FLAVOR"
    ;;
esac

if [[ -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  pass "rosdep initialized"
else
  warn "rosdep not initialized"
fi

if [[ -d "$AUTOWARE_WS/build" || -d "$AUTOWARE_WS/install" ]]; then
  pass "workspace has build or install artifacts"
else
  warn "workspace has not been built yet"
fi

echo
echo "Summary: $PASS_COUNT pass, $WARN_COUNT warn"
if [[ "$WARN_COUNT" -gt 0 ]]; then
  exit 1
fi
