#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 1
fi

if [[ "${1:-}" == "--reset" ]]; then
  sudo nvidia-smi -i "$GPU" -rgc
  sudo nvidia-smi -i "$GPU" -rmc
  exit 0
fi

read -r graphics_clock memory_clock < <(
  nvidia-smi -i "$GPU" --query-gpu=clocks.max.graphics,clocks.max.memory --format=csv,noheader,nounits |
    tr -d ',' |
    awk '{print $1, $2}'
)

sudo nvidia-smi -i "$GPU" -pm 1
sudo nvidia-smi -i "$GPU" -lgc "$graphics_clock"
sudo nvidia-smi -i "$GPU" -lmc "$memory_clock"

echo "Locked GPU $GPU clocks: graphics=${graphics_clock}MHz memory=${memory_clock}MHz"
