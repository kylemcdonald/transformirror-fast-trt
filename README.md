# Transformirror Fast TRT

Realtime SDXL Turbo img2img for webcam input using TensorRT, CUDA Graphs, and a
native C++ runtime.

The default build targets 1024x1024 SDXL Turbo img2img with TAESDXL,
`strength=0.7`, `steps=2`, and `guidance_scale=0`. With that configuration
Diffusers resolves to one effective denoise step at timestep `499`.

## Current 5090 Performance

Measured on an RTX 5090 with TensorRT 10.12, CUDA 12.8, and a Logitech BRIO:

| Path | Mean |
| --- | ---: |
| Diffusers + TAESDXL, no stable-fast | 69.9 ms |
| stable-fast Diffusers pipeline | 56.0 ms |
| Manual stable-fast CUDA Graph | 41.8 ms |
| Python TensorRT runtime graph | 31.4 ms |
| C++ TensorRT/CUDA Graph core | 31.3 ms |
| C++ app, webcam upload + display readback | about 31-32 ms diffusion time |
| C++ app, direct V4L2 capture, no display/readback | model 29.7 ms, loop 32.2 ms |
| C++ app, direct V4L2 capture, FFplay display | model 31.5 ms, display write 1.0 ms, loop 33.4 ms |
| C++ app, direct V4L2 capture, CUDA/GL display | model 29.9-30.2 ms, display 0.07-0.14 ms, loop 32.1-36.0 ms |

In the webcam app, the BRIO at 1920x1080 MJPEG/30 is camera-limited around
30 FPS. The model path itself is right at the 30 FPS boundary for 1024x1024.

Batching was tested with static TensorRT engines:

| Batch | Core batch mean | Core per frame | Upload+download batch mean | Upload+download per frame |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 31.33 ms | 31.33 ms | 32.05 ms | 32.05 ms |
| 2 | 60.39 ms | 30.19 ms | 61.13 ms | 30.57 ms |
| 4 | 117.11 ms | 29.28 ms | 117.90 ms | 29.47 ms |

Batch 4 only improves throughput about 8% while adding batch-fill latency. For
single-webcam use, the live app stays on the lower-latency single-frame path.

## What This App Does

`transformirror_fast_app`:

* captures a centered square crop from a webcam with direct V4L2 mmap or FFmpeg
* scales the crop to the TensorRT engine resolution
* uploads RGB frames to CUDA
* runs TAESDXL encode, SDXL Turbo UNet, scheduler math, TAESDXL decode, and blend
* displays fullscreen with CUDA/OpenGL interop or FFplay
* can skip output readback entirely in no-display mode
* serves a browser control frontend over HTTP
* receives OSC control messages over UDP
* hot-loads prompt/seed/strength/step conditioning assets without restarting
* defaults to lowest-latency capture: always process the newest camera frame

The frame loop is C++/CUDA/TensorRT. Prompt/seed/strength/step edits are handled
out of band by a persistent Python conditioning worker because text encoding is
not in the realtime loop. The old one-shot helper remains available with
`--conditioning-backend script`.

## Install

Ubuntu packages:

```bash
sudo apt-get update
sudo apt-get install -y \
  git python3-venv python3-dev build-essential cmake ninja-build \
  ffmpeg v4l-utils pkg-config libjpeg-dev
```

For the CUDA/OpenGL display backend:

```bash
sudo ./scripts/install_display_deps.sh
```

You also need a recent NVIDIA driver, CUDA Toolkit, and TensorRT with `trtexec`.
On this machine, `trtexec` is at:

```bash
/usr/src/tensorrt/bin/trtexec
```

Create the Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

Build the default ONNX exports, TensorRT engines, conditioning assets, and C++
binaries:

```bash
source .venv/bin/activate
./scripts/build_default_engines.sh
```

TensorRT plans are GPU and TensorRT-version specific. Build them on the machine
that will run the app. For a 4090, run the same build script on the 4090.

Optional batching benchmark:

```bash
source .venv/bin/activate
python export_onnx_components.py --component all --batch-size 4 --out-dir /tmp/sdxl_trt_batch_b4
/usr/src/tensorrt/bin/trtexec --onnx=/tmp/sdxl_trt_batch_b4/taesdxl_encode.onnx --fp16 --saveEngine=/tmp/sdxl_trt_batch_b4/taesdxl_encode.plan
/usr/src/tensorrt/bin/trtexec --onnx=/tmp/sdxl_trt_batch_b4/taesdxl_decode.onnx --fp16 --saveEngine=/tmp/sdxl_trt_batch_b4/taesdxl_decode.plan
/usr/src/tensorrt/bin/trtexec --onnx=/tmp/sdxl_trt_batch_b4/sdxl_turbo_unet.onnx --fp16 --saveEngine=/tmp/sdxl_trt_batch_b4/sdxl_turbo_unet.plan --useCudaGraph
python benchmark_trt_batch.py --engine-dir /tmp/sdxl_trt_batch_b4 --batch-size 4 --include-upload --include-download
```

## Run

Headless smoke test:

```bash
./cpp/build/transformirror_fast_app --no-display --max-frames 120
```

Fullscreen webcam app:

```bash
./cpp/build/transformirror_fast_app \
  --camera-device /dev/video0 \
  --capture-backend v4l2 \
  --display-backend gl \
  --capture-width 1920 \
  --capture-height 1080 \
  --camera-fps 30 \
  --conditioning-backend worker \
  --http-port 8080 \
  --osc-port 9000
```

Low-jitter options:

```bash
./cpp/build/transformirror_fast_app \
  --capture-backend v4l2 \
  --display-backend gl \
  --realtime \
  --main-core 2 \
  --capture-core 3 \
  --http-core 4 \
  --osc-core 4 \
  --reload-core 5
```

`--realtime` requests `SCHED_FIFO` priority 10 and `mlockall`. These require
system privileges or raised resource limits, so the app logs a warning and keeps
running if the OS denies them. Core pinning works without elevated privileges.

For a no-display benchmark that avoids output readback:

```bash
./cpp/build/transformirror_fast_app --capture-backend v4l2 --display-backend none --max-frames 300
```

To reduce GPU clock-related jitter:

```bash
./scripts/lock_gpu_clocks.sh
./scripts/lock_gpu_clocks.sh --reset
```

The clock script uses `sudo nvidia-smi`; run it from a shell where sudo is
available.

Open the control UI:

```text
http://localhost:8080/
http://<hostname>.local:8080/
```

HTTP and OSC bind to all IPv4 and IPv6 interfaces. On Linux with Avahi/mDNS
enabled, the same HTTP UI and OSC UDP port are reachable at
`<hostname>.local`.

## HTTP API

State:

```bash
curl http://localhost:8080/api/state
```

Update:

```bash
curl -X POST http://localhost:8080/api/state \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a neon mirror portrait","seed":42,"strength":0.7,"steps":2,"blend":0.5,"use_latest_frame":true}'
```

Prompt, seed, and strength trigger asynchronous conditioning regeneration.
With the default persistent worker, prompt changes are typically tens of
milliseconds after the worker has warmed. The current fastest TensorRT graph is
one pass and clamps `steps` to `2`. Blend and passthrough update immediately.

`use_latest_frame` controls capture behavior:

* `true` - default, drain camera input continuously and process the newest frame
* `false` - FIFO mode, do not drop frames

## OSC

OSC UDP port defaults to `9000`.

Supported addresses:

```text
/prompt        string
/seed          int
/strength      float
/steps         int
/blend         float
/passthrough   int/bool
/use_latest_frame int/bool
/frame_mode    string, "latest" or "fifo"
/width         int
/height        int
```

Namespaced versions also work, for example `/transformirror/blend`.

## Latency Notes

Implemented low-level latency controls:

* reusable CUDA events instead of per-frame event allocation
* no-display mode skips device-to-host output readback
* CUDA/OpenGL display avoids device-to-host output readback in the visible path
* direct V4L2 mmap capture for MJPEG webcams
* explicit capture/model/display/loop timings in `/api/state`
* optional CPU affinity and `SCHED_FIFO` thread priority
* optional `mlockall` process memory locking
* GPU clock lock/reset helper script

The default display backend is `gl`, implemented with GLX, an OpenGL pixel
buffer object, and CUDA graphics interop. Use `--display-backend ffplay` as a
fallback if X11/OpenGL is unavailable.

## Resolution Notes

The current fast path is fixed-shape TensorRT. The app exposes width/height in
the API for compatibility with earlier Transformirror control surfaces, but the
default build only ships commands for 1024x1024 engines. Changing resolution
properly means exporting and building a matching set of TensorRT plans.

## Architectural Research Notes

Ideas considered:

* TensorRT FP16 engines gave the largest real speedup in this repo.
* Naive `trtexec --fp8` on the exported SDXL Turbo UNet did not help on this
  setup; the FP8 engine was slightly slower than FP16.
* NVIDIA's calibrated 8-bit diffusion quantization is still the most promising
  next low-level path. Their TensorRT writeup reports up to 1.95x speedups for
  SDXL with calibrated INT8/FP8 PTQ while preserving quality.
* DeepCache-style feature reuse is less applicable here because this img2img
  configuration already performs only one effective denoise step.
* StreamDiffusion ideas like batching, residual CFG, and stochastic similarity
  filtering are valuable for multi-step/CFG pipelines. This app uses
  `guidance_scale=0` and one denoise step, so RCFG and denoise batching do not
  buy much.
* The largest remaining architectural jump is model replacement/distillation.
  SDXS reports one-step image-conditioned models, with SDXS-1024 around 30 FPS
  in the paper. Publicly available SDXS assets are currently centered on 512px,
  so this repo stays with SDXL Turbo until a suitable 1024 image-conditioned
  replacement is available.
* lyraDiff and related engines point toward fused GroupNorm/NHWC, fused GEMM,
  Flash Attention, and INT8/FP8/INT4 pipelines as the next class of work.

References:

* NVIDIA TensorRT 8-bit diffusion quantization:
  https://developer.nvidia.com/blog/?p=78835
* NVIDIA Model Optimizer diffusion examples:
  https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/diffusers/README.md
* StreamDiffusion:
  https://huggingface.co/papers/2312.12491
* DeepCache:
  https://openaccess.thecvf.com/content/CVPR2024/html/Ma_DeepCache_Accelerating_Diffusion_Models_for_Free_CVPR_2024_paper.html
* SDXS:
  https://arxiv.org/abs/2403.16627
* lyraDiff:
  https://github.com/TMElyralab/lyraDiff

## Files

* `cpp/transformirror_fast_app.cu` - webcam app
* `cpp/trt_sdxl_runner.cu` - benchmark runner
* `benchmark_trt_batch.py` - static batch TensorRT benchmark helper
* `export_onnx_components.py` - ONNX export for VAE/UNet
* `export_cpp_assets.py` - prompt/noise/scheduler asset export
* `scripts/build_default_engines.sh` - default 1024x1024 build
* `scripts/install_display_deps.sh` - Ubuntu display backend dependencies
* `scripts/lock_gpu_clocks.sh` - optional NVIDIA clock lock/reset helper
