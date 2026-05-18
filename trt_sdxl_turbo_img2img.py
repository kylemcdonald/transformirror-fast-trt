#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny
from diffusers.utils.logging import disable_progress_bar
from PIL import Image, ImageDraw

from fastest_sdxl_turbo_img2img import make_test_image, run_original, tensor_to_pil, to_input_tensor


PROMPT = "a cinematic mirror portrait, detailed face, luminous color, sharp focus"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", type=Path, default=Path("onnx"))
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--include-upload", action="store_true")
    parser.add_argument("--include-download", action="store_true")
    parser.add_argument("--save-comparison", type=Path)
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
        self.tensors = {}

    def bind(self, name, tensor):
        self.tensors[name] = tensor
        if not self.context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"failed to bind tensor address: {name}")

    def execute(self, stream):
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT execution failed")


def build_setup_pipeline(device):
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
    )
    pipe.vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesdxl",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device=device, dtype=torch.float16)
    return pipe


class TRTSDXLTurboImg2Img:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.dtype = torch.float16
        self.static_input = torch.empty(
            (1, 3, args.height, args.width), device=device, dtype=self.dtype
        )

        setup_pipe = build_setup_pipeline(device)
        with torch.inference_mode():
            encoded = setup_pipe.encode_prompt(
                prompt=args.prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        self.prompt_embeds, _, pooled_prompt_embeds, _ = encoded

        setup_pipe.scheduler.set_timesteps(args.steps, device=device)
        self.timesteps, self.effective_steps = setup_pipe.get_timesteps(
            args.steps, args.strength, device
        )
        if len(self.timesteps) != 1:
            raise ValueError(f"this TRT fast path expects one effective denoise step, got {len(self.timesteps)}")

        sigma = setup_pipe.scheduler.sigmas[setup_pipe.scheduler.begin_index]
        next_sigma = float(setup_pipe.scheduler.sigmas[setup_pipe.scheduler.begin_index + 1])
        if next_sigma != 0.0:
            raise ValueError("this TRT fast path expects the final Euler sigma to be zero")

        self.sigma = sigma.to(device=device, dtype=self.dtype)
        self.sigma_f32 = sigma.to(device=device, dtype=torch.float32)
        sigma_scale = torch.sqrt(self.sigma * self.sigma + 1.0)
        self.inv_sigma_scale = torch.reciprocal(sigma_scale)
        self.scaling_factor = torch.tensor(
            setup_pipe.vae.config.scaling_factor, device=device, dtype=self.dtype
        )
        self.inv_scaling_factor = torch.reciprocal(self.scaling_factor)
        self.timestep = self.timesteps[:1].to(device=device, dtype=torch.float32)

        projection_dim = (
            int(pooled_prompt_embeds.shape[-1])
            if setup_pipe.text_encoder_2 is None
            else setup_pipe.text_encoder_2.config.projection_dim
        )
        add_time_ids, _ = setup_pipe._get_add_time_ids(
            (args.height, args.width),
            (0, 0),
            (args.height, args.width),
            6.0,
            2.5,
            (args.height, args.width),
            (0, 0),
            (args.height, args.width),
            dtype=self.prompt_embeds.dtype,
            text_encoder_projection_dim=projection_dim,
        )
        self.time_ids = add_time_ids.to(device)
        self.text_embeds = pooled_prompt_embeds.to(device)

        generator = torch.Generator(device=device).manual_seed(args.seed)
        latent_shape = (1, 4, args.height // setup_pipe.vae_scale_factor, args.width // setup_pipe.vae_scale_factor)
        self.noise = torch.randn(latent_shape, generator=generator, device=device, dtype=self.dtype)

        del setup_pipe
        torch.cuda.empty_cache()

        engine_dir = args.engine_dir
        self.encode_engine = Engine(engine_dir / "taesdxl_encode.plan")
        self.unet_engine = Engine(engine_dir / "sdxl_turbo_unet.plan")
        self.decode_engine = Engine(engine_dir / "taesdxl_decode.plan")

        self.encoded_latents = torch.empty(latent_shape, device=device, dtype=self.dtype)
        self.latents = torch.empty_like(self.encoded_latents)
        self.noised_latents = torch.empty_like(self.encoded_latents)
        self.unet_input = torch.empty_like(self.encoded_latents)
        self.noise_pred = torch.empty_like(self.encoded_latents)
        self.latents_f32 = torch.empty(latent_shape, device=device, dtype=torch.float32)
        self.noise_pred_f32 = torch.empty(latent_shape, device=device, dtype=torch.float32)
        self.decode_input = torch.empty_like(self.encoded_latents)
        self.decoded = torch.empty((1, 3, args.height, args.width), device=device, dtype=self.dtype)
        self.output = torch.empty_like(self.decoded)

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
        torch.mul(self.latents_f32, self.inv_scaling_factor.float(), out=self.decode_input)
        self.decode_engine.execute(stream)
        torch.mul(self.decoded, 0.5, out=self.output)
        self.output.add_(0.5)
        self.output.clamp_(0, 1)
        return self.output

    def capture(self, image):
        self.static_input.copy_(image)
        for _ in range(5):
            self.forward()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.forward()
        self.graph = graph
        torch.cuda.synchronize()

    def replay(self, image=None):
        if image is not None:
            self.static_input.copy_(image)
        self.graph.replay()
        return self.output


def summarize(times):
    ordered = sorted(times)
    return {
        "count": len(times),
        "mean_ms": statistics.fmean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "p90_ms": ordered[int(0.9 * (len(ordered) - 1))],
    }


def save_comparison(path, original, fast, width, height):
    comparison = Image.new("RGB", (width * 2, height), "black")
    comparison.paste(original.convert("RGB"), (0, 0))
    comparison.paste(fast.convert("RGB"), (width, 0))
    draw = ImageDraw.Draw(comparison)
    draw.rectangle((0, 0, width, 42), fill=(0, 0, 0))
    draw.rectangle((width, 0, width * 2, 42), fill=(0, 0, 0))
    draw.text((16, 12), "Original Diffusers pipeline", fill=(255, 255, 255))
    draw.text((width + 16, 12), "TensorRT fast pipeline", fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(path)


def main():
    args = parse_args()
    disable_progress_bar()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")

    image_np = make_test_image(args.height, args.width)
    original = None
    if args.save_comparison:
        original_pipe = build_setup_pipeline(device)
        original = run_original(original_pipe, image_np, args.prompt, args)
        del original_pipe
        torch.cuda.empty_cache()

    input_tensor = to_input_tensor(image_np, device)
    runner = TRTSDXLTurboImg2Img(args, device)
    runner.capture(input_tensor)

    cpu_frames = [make_test_image(args.height, args.width) for _ in range(args.runs + args.warmup)]
    times = []
    last = None
    for i in range(args.warmup + args.runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            frame = to_input_tensor(cpu_frames[i], device) if args.include_upload else input_tensor
            last = runner.replay(frame)
            if args.include_download:
                _ = last[0].permute(1, 2, 0).float().cpu().numpy()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0
        label = "warmup" if i < args.warmup else "run"
        idx = i + 1 if i < args.warmup else i - args.warmup + 1
        total = args.warmup if i < args.warmup else args.runs
        print(f"{label} {idx}/{total}: {elapsed:.2f} ms", flush=True)
        if i >= args.warmup:
            times.append(elapsed)

    result = summarize(times)
    result.update(
        {
            "engine_dir": str(args.engine_dir),
            "include_upload": args.include_upload,
            "include_download": args.include_download,
            "effective_denoise_steps": int(runner.effective_steps),
            "timesteps": [float(t.detach().cpu()) for t in runner.timesteps],
        }
    )

    if args.save_comparison:
        fast = tensor_to_pil(last)
        save_comparison(args.save_comparison, original, fast, args.width, args.height)
        result["comparison"] = str(args.save_comparison)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"summary: mean={result['mean_ms']:.2f} ms, "
            f"median={result['median_ms']:.2f} ms, min={result['min_ms']:.2f} ms, "
            f"p90={result['p90_ms']:.2f} ms, max={result['max_ms']:.2f} ms"
        )


if __name__ == "__main__":
    main()
