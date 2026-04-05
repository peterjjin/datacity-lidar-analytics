#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AUTOWARE_FLAVOR="${AUTOWARE_FLAVOR:-core}"
AUTOWARE_WS="${AUTOWARE_WS:-$HOME/autoware_core_workspace}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
SKIP_APT=0

usage() {
  cat <<'EOF'
Usage:
  bash setup/bootstrap-wsl-autoware.sh [--skip-apt] [--flavor core|main] [--workspace PATH]

What it does:
  - prepares a separate ROS 2 Humble workspace for Autoware testing
  - installs core ROS build tools if needed
  - clones either autoware_core or full autoware into a dedicated WSL workspace
  - runs rosdep dependency installation

Options:
  --skip-apt
      Skip apt-based package installation. Use only if ROS 2 Humble tools are already present.
  --flavor core|main
      core: clone autoware_core into ~/autoware_core_workspace
      main: clone autoware into ~/autoware_workspace
  --workspace PATH
      Override the target workspace path.

Environment variables:
  AUTOWARE_FLAVOR
  AUTOWARE_WS
  ROS_DISTRO
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-apt)
      SKIP_APT=1
      shift
      ;;
    --flavor)
      AUTOWARE_FLAVOR="$2"
      shift 2
      ;;
    --workspace)
      AUTOWARE_WS="$2"
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

case "$AUTOWARE_FLAVOR" in
  core)
    AUTOWARE_REPO_URL="${AUTOWARE_REPO_URL:-https://github.com/autowarefoundation/autoware_core.git}"
    AUTOWARE_REPO_DIR="$AUTOWARE_WS/src/autoware_core"
    DEFAULT_WS="$HOME/autoware_core_workspace"
    ;;
  main)
    AUTOWARE_REPO_URL="${AUTOWARE_REPO_URL:-https://github.com/autowarefoundation/autoware.git}"
    AUTOWARE_REPO_DIR="$AUTOWARE_WS/src/autoware"
    DEFAULT_WS="$HOME/autoware_workspace"
    ;;
  *)
    echo "Invalid --flavor: $AUTOWARE_FLAVOR (expected core or main)" >&2
    exit 2
    ;;
esac

if [[ "${AUTOWARE_WS:-}" == "" ]]; then
  AUTOWARE_WS="$DEFAULT_WS"
fi

step() {
  echo
  echo "==> $*"
}

ensure_ros_apt_repo() {
  step "Ensuring ROS 2 apt repository and key"
  curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
    | gpg --dearmor \
    | sudo tee /usr/share/keyrings/ros-archive-keyring.gpg >/dev/null
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
}

if [[ "$SKIP_APT" -eq 0 ]]; then
  ensure_ros_apt_repo

  step "Installing ROS 2 and Autoware prerequisites"
  sudo apt-get -y update
  sudo apt-get -y install \
    git \
    curl \
    wget \
    lsb-release \
    gnupg2 \
    build-essential \
    python3-argcomplete

  step "Installing ROS 2 build tools"
  sudo apt-get -y install \
    python3-colcon-common-extensions \
    python3-rosdep2 \
    python3-vcstool

  if [[ ! -d "/opt/ros/$ROS_DISTRO" ]]; then
    step "Installing ROS 2 $ROS_DISTRO desktop"
    sudo apt-get -y install "ros-$ROS_DISTRO-desktop"
  else
    step "ROS 2 $ROS_DISTRO already present"
  fi

  if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    step "Initializing rosdep"
    sudo rosdep init
  else
    step "rosdep already initialized"
  fi
else
  step "Skipping apt-based prerequisite install"
fi

step "Updating rosdep database"
rosdep update

step "Creating Autoware workspace at $AUTOWARE_WS"
mkdir -p "$AUTOWARE_WS/src"

if [[ ! -d "$AUTOWARE_REPO_DIR/.git" ]]; then
  step "Cloning $AUTOWARE_REPO_URL"
  git clone "$AUTOWARE_REPO_URL" "$AUTOWARE_REPO_DIR"
else
  step "Autoware repository already present at $AUTOWARE_REPO_DIR"
fi

step "Installing workspace dependencies with rosdep"
# ROS setup scripts assume some shell vars may be unset, so source them with nounset disabled.
set +u
source "/opt/ros/$ROS_DISTRO/setup.bash"
set -u
cd "$AUTOWARE_WS"
rosdep install -y --from-paths src --ignore-src --rosdistro "$ROS_DISTRO"

step "Done"
echo "Workspace: $AUTOWARE_WS"
echo "Repo: $AUTOWARE_REPO_DIR"
echo
echo "Next steps:"
echo "  conda deactivate  # repeat until no Conda env is active"
echo "  source /opt/ros/$ROS_DISTRO/setup.bash"
echo "  cd \"$AUTOWARE_WS\""
echo "  colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"
echo "  bash \"$WORKSPACE_ROOT/setup/wsl-autoware-doctor.sh\" --workspace \"$AUTOWARE_WS\" --flavor \"$AUTOWARE_FLAVOR\""
