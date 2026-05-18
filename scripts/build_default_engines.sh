#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

if [[ ! -x "$TRTEXEC" ]]; then
  echo "trtexec not found at $TRTEXEC. Set TRTEXEC=/path/to/trtexec." >&2
  exit 1
fi

python export_onnx_components.py --component all --out-dir onnx

"$TRTEXEC" --onnx=onnx/taesdxl_encode.onnx \
  --fp16 --saveEngine=onnx/taesdxl_encode.plan

"$TRTEXEC" --onnx=onnx/taesdxl_decode.onnx \
  --fp16 --saveEngine=onnx/taesdxl_decode.plan

"$TRTEXEC" --onnx=onnx/sdxl_turbo_unet.onnx \
  --fp16 --saveEngine=onnx/sdxl_turbo_unet.plan --useCudaGraph

python export_cpp_assets.py --out-dir cpp_assets

cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j"$(nproc)"
