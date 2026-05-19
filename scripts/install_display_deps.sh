#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Run this with sudo:" >&2
  echo "  sudo $0" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y \
  libx11-dev \
  libxext-dev \
  libxfixes-dev \
  libxrandr-dev \
  libxi-dev \
  libxcursor-dev \
  libxinerama-dev \
  libgl1-mesa-dev \
  libglx-dev \
  libegl1-mesa-dev \
  libgles2-mesa-dev \
  mesa-common-dev \
  libglew-dev \
  libglfw3-dev \
  libdrm-dev \
  libgbm-dev \
  nvidia-settings

echo "Display development dependencies installed."
echo "You should now have X11/OpenGL/EGL headers for the CUDA display backend work."
