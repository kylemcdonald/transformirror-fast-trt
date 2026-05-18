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

In the webcam app, the BRIO at 1920x1080 MJPEG/30 is camera-limited around
30 FPS. The model path itself is right at the 30 FPS boundary for 1024x1024.

## What This App Does

`transformirror_fast_app`:

* captures a centered square crop from a webcam with FFmpeg/V4L2
* scales the crop to the TensorRT engine resolution
* uploads RGB frames to CUDA
* runs TAESDXL encode, SDXL Turbo UNet, scheduler math, TAESDXL decode, and blend
* displays fullscreen with FFplay
* serves a browser control frontend over HTTP
* receives OSC control messages over UDP
* hot-loads prompt/seed/strength/step conditioning assets without restarting

The frame loop is C++/CUDA/TensorRT. Prompt/seed/strength/step edits are handled
out of band by a Python helper because text encoding is not in the realtime loop.

## Install

Ubuntu packages:

```bash
sudo apt-get update
sudo apt-get install -y \
  git python3-venv python3-dev build-essential cmake ninja-build \
  ffmpeg v4l-utils pkg-config
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

## Run

Headless smoke test:

```bash
./cpp/build/transformirror_fast_app --no-display --max-frames 120
```

Fullscreen webcam app:

```bash
./cpp/build/transformirror_fast_app \
  --camera-device /dev/video0 \
  --capture-width 1920 \
  --capture-height 1080 \
  --camera-fps 30 \
  --http-port 8080 \
  --osc-port 9000
```

Open the control UI:

```text
http://localhost:8080/
```

## HTTP API

State:

```bash
curl http://localhost:8080/api/state
```

Update:

```bash
curl -X POST http://localhost:8080/api/state \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a neon mirror portrait","seed":42,"strength":0.7,"steps":2,"blend":0.5}'
```

Prompt, seed, strength, and steps trigger asynchronous conditioning regeneration.
Blend and passthrough update immediately.

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
/width         int
/height        int
```

Namespaced versions also work, for example `/transformirror/blend`.

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
* `export_onnx_components.py` - ONNX export for VAE/UNet
* `export_cpp_assets.py` - prompt/noise/scheduler asset export
* `scripts/build_default_engines.sh` - default 1024x1024 build
