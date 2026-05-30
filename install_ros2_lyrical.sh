#!/usr/bin/env bash
set -euo pipefail

variant="${1:-ros-base}"

case "$variant" in
  ros-base|base)
    ros_pkg="ros-lyrical-ros-base"
    ;;
  desktop)
    ros_pkg="ros-lyrical-desktop"
    ;;
  *)
    echo "Usage: $0 [ros-base|desktop]" >&2
    exit 2
    ;;
esac

. /etc/os-release
codename="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"

if [[ "${VERSION_ID:-}" != "26.04" || "$codename" != "resolute" ]]; then
  echo "This script is intended for Ubuntu 26.04 (resolute)." >&2
  echo "Detected: VERSION_ID=${VERSION_ID:-unknown}, codename=${codename:-unknown}" >&2
  exit 1
fi

echo "[INFO] Installing ROS 2 Lyrical package: $ros_pkg"

sudo apt update
sudo apt install -y locales software-properties-common curl ca-certificates
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
sudo add-apt-repository -y universe

ros_apt_source_version="1.2.0"

deb="/tmp/ros2-apt-source.deb"
url="https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ros_apt_source_version}/ros2-apt-source_${ros_apt_source_version}.${codename}_all.deb"

echo "[INFO] Downloading ROS apt source: $url"
curl -fL -o "$deb" "$url"
sudo dpkg -i "$deb"

sudo apt update
sudo apt install -y "$ros_pkg" ros-dev-tools

setup_line="source /opt/ros/lyrical/setup.bash"
if ! grep -qxF "$setup_line" "$HOME/.bashrc"; then
  echo "$setup_line" >> "$HOME/.bashrc"
fi

echo "[INFO] ROS 2 Lyrical installed."
echo "[INFO] Run: source /opt/ros/lyrical/setup.bash"
echo "[INFO] Then test: ros2 --help"
