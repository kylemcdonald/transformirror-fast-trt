#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WIDTH="${1:?width required}"
HEIGHT="${2:?height required}"

cd "$ROOT"
source "$ROOT/.venv/bin/activate"

ONNX_DIR="$ROOT/onnx/${WIDTH}x${HEIGHT}"
ENGINE_DIR="$ROOT/trt_engines/${WIDTH}x${HEIGHT}"

python export_onnx_components.py --width "$WIDTH" --height "$HEIGHT" --component all --out-dir "$ONNX_DIR"
python build_trt_engines.py --onnx-dir "$ONNX_DIR" --engine-dir "$ENGINE_DIR"
python export_cpp_assets.py --width "$WIDTH" --height "$HEIGHT" --out-dir "$ENGINE_DIR/assets"
