#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_ID="$(id -u)"

export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/${USER_ID}}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"
export PYTHONNOUSERSITE=1
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TRANSFORMIRROR_TRT_ENGINE_ROOT="${TRANSFORMIRROR_TRT_ENGINE_ROOT:-$ROOT_DIR/trt_engines}"
export PATH="$ROOT_DIR/.venv/bin:/usr/local/cuda/bin:$PATH"

gsettings set org.gnome.desktop.session idle-delay 0 >/dev/null 2>&1 || true
gsettings set org.gnome.desktop.screensaver lock-enabled false >/dev/null 2>&1 || true
gsettings set org.gnome.desktop.screensaver idle-activation-enabled false >/dev/null 2>&1 || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' >/dev/null 2>&1 || true
xset s off s noblank -dpms >/dev/null 2>&1 || true
xset dpms force on >/dev/null 2>&1 || true

(
  while true; do
    xset s off s noblank -dpms >/dev/null 2>&1 || true
    xset dpms force on >/dev/null 2>&1 || true
    sleep 30
  done
) &

cd "$ROOT_DIR"

CONFIG_PATH="${TRANSFORMIRROR_CONFIG:-$ROOT_DIR/live_config.json}"

eval "$("$ROOT_DIR/.venv/bin/python" - "$CONFIG_PATH" <<'PY'
import json
import shlex
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = {}
if config_path.exists():
    with config_path.open() as f:
        config = json.load(f)

defaults = {
    "width": 1280,
    "height": 640,
    "camera_device": "/dev/video0",
    "camera_backend": "v4l2",
    "camera_fps": 30,
    "prompt": "a cinematic mirror portrait, detailed face, luminous color, sharp focus",
    "seed": 0,
    "strength": 0.7,
    "blend": 1.0,
    "steps": 2,
    "http_port": 8080,
    "osc_port": 9000,
}
for key, value in defaults.items():
    value = config.get(key, value)
    var = "CFG_" + key.upper()
    print(f"{var}={shlex.quote(str(value))}")
PY
)"

WIDTH="${TRANSFORMIRROR_WIDTH:-$CFG_WIDTH}"
HEIGHT="${TRANSFORMIRROR_HEIGHT:-$CFG_HEIGHT}"
ENGINE_DIR="$ROOT_DIR/trt_engines/${WIDTH}x${HEIGHT}"
APP_BINARY="$ROOT_DIR/cpp/build_${WIDTH}x${HEIGHT}/transformirror_fast_app"

if [[ ! -x "$APP_BINARY" ]]; then
  TRANSFORMIRROR_PROMPT="$CFG_PROMPT" \
  TRANSFORMIRROR_SEED="$CFG_SEED" \
  TRANSFORMIRROR_STRENGTH="$CFG_STRENGTH" \
  TRANSFORMIRROR_STEPS="$CFG_STEPS" \
  TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}" \
    "$ROOT_DIR/scripts/build_resolution_app.sh" "$WIDTH" "$HEIGHT"
fi

exec "$APP_BINARY" \
  --engine-dir "$ENGINE_DIR" \
  --asset-dir "$ENGINE_DIR/assets" \
  --web-root "$ROOT_DIR/web" \
  --conditioning-backend worker \
  --python "$ROOT_DIR/.venv/bin/python" \
  --camera-device "$CFG_CAMERA_DEVICE" \
  --capture-backend "$CFG_CAMERA_BACKEND" \
  --display-backend "${TRANSFORMIRROR_DISPLAY_BACKEND:-gl}" \
  --gl-sync "${TRANSFORMIRROR_GL_SYNC:-vsync}" \
  --capture-width "${TRANSFORMIRROR_CAPTURE_WIDTH:-1920}" \
  --capture-height "${TRANSFORMIRROR_CAPTURE_HEIGHT:-1080}" \
  --camera-fps "$CFG_CAMERA_FPS" \
  --http-port "$CFG_HTTP_PORT" \
  --osc-port "$CFG_OSC_PORT" \
  --initial-prompt "$CFG_PROMPT" \
  --initial-seed "$CFG_SEED" \
  --initial-strength "$CFG_STRENGTH" \
  --initial-steps "$CFG_STEPS" \
  --initial-blend "$CFG_BLEND" \
  "$@"
