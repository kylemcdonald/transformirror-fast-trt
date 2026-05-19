#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WIDTH="${1:?width required}"
HEIGHT="${2:?height required}"

cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" && -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
if [[ ! -x "$TRTEXEC" ]]; then
  echo "trtexec not found at $TRTEXEC. Set TRTEXEC=/path/to/trtexec." >&2
  exit 1
fi

ONNX_DIR="$ROOT/onnx/${WIDTH}x${HEIGHT}"
ENGINE_DIR="$ROOT/trt_engines/${WIDTH}x${HEIGHT}"
BUILD_DIR="$ROOT/cpp/build_${WIDTH}x${HEIGHT}"

mkdir -p "$ONNX_DIR" "$ENGINE_DIR"

if [[ ! -f "$ONNX_DIR/taesdxl_encode.onnx" ||
      ! -f "$ONNX_DIR/taesdxl_decode.onnx" ||
      ! -f "$ONNX_DIR/sdxl_turbo_unet.onnx" ]]; then
  "$PYTHON_BIN" export_onnx_components.py --width "$WIDTH" --height "$HEIGHT" --component all --out-dir "$ONNX_DIR"
else
  echo "keeping existing ONNX exports in $ONNX_DIR"
fi

build_plan() {
  local name="$1"
  local extra="${2:-}"
  local onnx="$ONNX_DIR/${name}.onnx"
  local plan="$ENGINE_DIR/${name}.plan"
  if [[ -f "$plan" ]]; then
    echo "keeping existing $plan"
    return
  fi
  "$TRTEXEC" --onnx="$onnx" --fp16 --saveEngine="$plan" $extra
}

build_plan taesdxl_encode
build_plan taesdxl_decode
build_plan sdxl_turbo_unet "--useCudaGraph"

ASSET_PROMPT="${TRANSFORMIRROR_PROMPT:-a cinematic mirror portrait, detailed face, luminous color, sharp focus}"
ASSET_SEED="${TRANSFORMIRROR_SEED:-0}"
ASSET_STRENGTH="${TRANSFORMIRROR_STRENGTH:-0.7}"
ASSET_STEPS="${TRANSFORMIRROR_STEPS:-2}"
"$PYTHON_BIN" export_cpp_assets.py \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --prompt "$ASSET_PROMPT" \
  --seed "$ASSET_SEED" \
  --strength "$ASSET_STRENGTH" \
  --steps "$ASSET_STEPS" \
  --out-dir "$ENGINE_DIR/assets"

cmake -S "$ROOT/cpp" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRANSFORMIRROR_WIDTH="$WIDTH" \
  -DTRANSFORMIRROR_HEIGHT="$HEIGHT"
cmake --build "$BUILD_DIR" -j"$(nproc)" --target transformirror_fast_app

echo "$BUILD_DIR/transformirror_fast_app"
