#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--assets-dir", type=Path, default=Path("cpp_assets"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--include-upload", action="store_true")
    parser.add_argument("--include-download", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


class Engine:
    def __init__(self, path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to load TensorRT engine: {path}")
        self.context = self.engine.create_execution_context()

    def shape(self, name):
        return tuple(int(dim) for dim in self.engine.get_tensor_shape(name))

    def bind(self, name, tensor):
        if not self.context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"failed to bind tensor address: {name}")

    def execute(self, stream):
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT execution failed")


def summarize(times):
    ordered = sorted(times)
    return {
        "count": len(times),
        "mean_ms": statistics.fmean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "p90_ms": ordered[int(0.9 * (len(ordered) - 1))],
        "p99_ms": ordered[int(0.99 * (len(ordered) - 1))],
    }


class BatchRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16

        self.encode_engine = Engine(args.engine_dir / "taesdxl_encode.plan")
        self.unet_engine = Engine(args.engine_dir / "sdxl_turbo_unet.plan")
        self.decode_engine = Engine(args.engine_dir / "taesdxl_decode.plan")

        image_shape = self.encode_engine.shape("image")
        latent_shape = self.encode_engine.shape("latents")
        decoded_shape = self.decode_engine.shape("image")
        if image_shape[0] != args.batch_size:
            raise ValueError(f"engine batch {image_shape[0]} does not match --batch-size {args.batch_size}")

        self.static_input = torch.empty(image_shape, device=self.device, dtype=self.dtype)
        self.encoded_latents = torch.empty(latent_shape, device=self.device, dtype=self.dtype)
        self.latents = torch.empty_like(self.encoded_latents)
        self.noised_latents = torch.empty_like(self.encoded_latents)
        self.unet_input = torch.empty_like(self.encoded_latents)
        self.noise_pred = torch.empty_like(self.encoded_latents)
        self.latents_f32 = torch.empty(latent_shape, device=self.device, dtype=torch.float32)
        self.noise_pred_f32 = torch.empty(latent_shape, device=self.device, dtype=torch.float32)
        self.decode_input = torch.empty_like(self.encoded_latents)
        self.decoded = torch.empty(decoded_shape, device=self.device, dtype=self.dtype)
        self.output = torch.empty_like(self.decoded)

        generator = torch.Generator(device=self.device).manual_seed(0)
        self.noise = torch.randn(latent_shape, generator=generator, device=self.device, dtype=self.dtype)
        self.timestep = torch.full(self.unet_engine.shape("timestep"), 499.0, device=self.device, dtype=torch.float32)
        self.prompt_embeds = torch.randn(
            self.unet_engine.shape("encoder_hidden_states"),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        self.text_embeds = torch.randn(
            self.unet_engine.shape("text_embeds"),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        self.time_ids = torch.zeros(self.unet_engine.shape("time_ids"), device=self.device, dtype=self.dtype)

        params = np.fromfile(args.assets_dir / "params.f32", dtype=np.float32)
        if params.size < 4:
            raise ValueError(f"expected at least 4 params in {args.assets_dir / 'params.f32'}")
        self.sigma = torch.tensor(params[0], device=self.device, dtype=self.dtype)
        self.sigma_f32 = torch.tensor(params[0], device=self.device, dtype=torch.float32)
        self.inv_sigma_scale = torch.tensor(params[1], device=self.device, dtype=self.dtype)
        self.scaling_factor = torch.tensor(params[2], device=self.device, dtype=self.dtype)
        self.inv_scaling_factor = torch.tensor(params[3], device=self.device, dtype=torch.float32)

        self.encode_engine.bind("image", self.static_input)
        self.encode_engine.bind("latents", self.encoded_latents)
        self.unet_engine.bind("sample", self.unet_input)
        self.unet_engine.bind("timestep", self.timestep)
        self.unet_engine.bind("encoder_hidden_states", self.prompt_embeds)
        self.unet_engine.bind("text_embeds", self.text_embeds)
        self.unet_engine.bind("time_ids", self.time_ids)
        self.unet_engine.bind("noise_pred", self.noise_pred)
        self.decode_engine.bind("latents", self.decode_input)
        self.decode_engine.bind("image", self.decoded)

        self.host_input = None
        self.host_output = None
        if args.include_upload:
            self.host_input = torch.empty(image_shape, device="cpu", dtype=self.dtype, pin_memory=True)
            self.host_input.uniform_(-1.0, 1.0)
        if args.include_download:
            self.host_output = torch.empty(decoded_shape, device="cpu", dtype=self.dtype, pin_memory=True)

        self.graph = None

    @torch.inference_mode()
    def forward(self):
        stream = torch.cuda.current_stream().cuda_stream
        self.encode_engine.execute(stream)
        torch.mul(self.encoded_latents, self.scaling_factor, out=self.latents)
        torch.mul(self.noise, self.sigma, out=self.noised_latents)
        torch.add(self.latents, self.noised_latents, out=self.latents)
        torch.mul(self.latents, self.inv_sigma_scale, out=self.unet_input)
        self.unet_engine.execute(stream)
        torch.mul(self.latents, 1.0, out=self.latents_f32)
        torch.mul(self.noise_pred, self.sigma_f32, out=self.noise_pred_f32)
        torch.sub(self.latents_f32, self.noise_pred_f32, out=self.latents_f32)
        torch.mul(self.latents_f32, self.inv_scaling_factor, out=self.decode_input)
        self.decode_engine.execute(stream)
        torch.mul(self.decoded, 0.5, out=self.output)
        self.output.add_(0.5)
        self.output.clamp_(0, 1)
        return self.output

    def capture(self):
        self.static_input.uniform_(-1.0, 1.0)
        for _ in range(5):
            self.forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.forward()
        self.graph = graph
        torch.cuda.synchronize()

    def replay(self):
        if self.host_input is not None:
            self.static_input.copy_(self.host_input, non_blocking=True)
        self.graph.replay()
        if self.host_output is not None:
            self.host_output.copy_(self.output, non_blocking=True)


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    runner = BatchRunner(args)
    runner.capture()

    times = []
    for i in range(args.warmup + args.runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        runner.replay()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if i >= args.warmup:
            times.append(elapsed_ms)

    stats = summarize(times)
    result = {
        "batch_size": args.batch_size,
        "engine_dir": str(args.engine_dir),
        "include_upload": args.include_upload,
        "include_download": args.include_download,
        "batch_ms": stats,
        "per_frame_ms": {key: value / args.batch_size for key, value in stats.items() if key != "count"},
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"batch={args.batch_size} mean={stats['mean_ms']:.3f} ms "
            f"median={stats['median_ms']:.3f} ms p90={stats['p90_ms']:.3f} ms "
            f"per_frame_mean={stats['mean_ms'] / args.batch_size:.3f} ms"
        )


if __name__ == "__main__":
    main()
