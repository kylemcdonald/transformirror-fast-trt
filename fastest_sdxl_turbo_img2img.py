#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny
from diffusers.utils.logging import disable_progress_bar
from PIL import Image, ImageDraw

from benchmark_sdxl_turbo_img2img import compile_with_stable_fast


PROMPT = "a cinematic mirror portrait, detailed face, luminous color, sharp focus"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("manual", "graph"), default="graph")
    parser.add_argument("--general-scheduler", action="store_true")
    parser.add_argument("--include-upload", action="store_true")
    parser.add_argument("--include-download", action="store_true")
    parser.add_argument("--no-preserve-parameters", action="store_true")
    parser.add_argument("--save-comparison", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def make_test_image(height, width):
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[..., 0] = 0.15 + 0.75 * x
    image[..., 1] = 0.20 + 0.55 * y
    image[..., 2] = 0.55 + 0.25 * np.sin(8 * np.pi * (x + y))
    image = np.clip(image, 0, 1)

    pil = Image.fromarray((image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil, "RGBA")
    draw.ellipse((300, 180, 724, 760), fill=(230, 190, 165, 190))
    draw.ellipse((395, 360, 465, 430), fill=(20, 30, 40, 220))
    draw.ellipse((560, 360, 630, 430), fill=(20, 30, 40, 220))
    draw.arc((430, 455, 595, 610), 10, 170, fill=(120, 40, 60, 220), width=12)
    draw.rectangle((140, 100, 884, 924), outline=(250, 250, 255, 160), width=18)
    draw.line((512, 0, 512, height), fill=(255, 255, 255, 90), width=5)
    return np.asarray(pil).astype(np.float32) / 255.0


def to_input_tensor(image, device):
    return (
        torch.from_numpy(image)
        .to(device=device, dtype=torch.float16)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .mul_(2.0)
        .sub_(1.0)
    )


def tensor_to_pil(image):
    array = (
        image[0]
        .detach()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .float()
        .cpu()
        .numpy()
    )
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def build_pipeline(device, no_preserve_parameters=False):
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

    compile_args = SimpleNamespace(
        cuda_graph=False,
        triton=False,
        xformers=False,
        fused_geglu=False,
        sfast_no_jit=False,
        sfast_no_jit_freeze=False,
        sfast_no_cnn_optimization=False,
        sfast_no_lowp_gemm=False,
        sfast_no_memory_format=False,
        no_preserve_parameters=no_preserve_parameters,
    )
    pipe, _ = compile_with_stable_fast(pipe, compile_args, device)
    return pipe


def build_original_pipeline(device):
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


class ManualSDXLTurboImg2Img:
    def __init__(self, pipe, prompt, height, width, steps, strength, seed, device):
        self.pipe = pipe
        self.height = height
        self.width = width
        self.device = device
        self.dtype = torch.float16

        with torch.inference_mode():
            encoded = pipe.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        self.prompt_embeds, _, pooled_prompt_embeds, _ = encoded
        self.add_text_embeds = pooled_prompt_embeds

        pipe.scheduler.set_timesteps(steps, device=device)
        self.timesteps, self.effective_steps = pipe.get_timesteps(steps, strength, device)
        self.latent_timestep = self.timesteps[:1].repeat(1)
        self.use_single_step_scheduler = (
            len(self.timesteps) == 1
            and float(pipe.scheduler.sigmas[pipe.scheduler.begin_index + 1]) == 0.0
        )
        self.sigma = pipe.scheduler.sigmas[pipe.scheduler.begin_index].to(
            device=device, dtype=self.dtype
        )
        self.sigma_f32 = pipe.scheduler.sigmas[pipe.scheduler.begin_index].to(
            device=device, dtype=torch.float32
        )
        self.sigma_scale = torch.sqrt(self.sigma * self.sigma + 1.0)

        projection_dim = (
            int(pooled_prompt_embeds.shape[-1])
            if pipe.text_encoder_2 is None
            else pipe.text_encoder_2.config.projection_dim
        )
        add_time_ids, _ = pipe._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            6.0,
            2.5,
            (height, width),
            (0, 0),
            (height, width),
            dtype=self.prompt_embeds.dtype,
            text_encoder_projection_dim=projection_dim,
        )
        self.add_time_ids = add_time_ids.to(device).repeat(1, 1)
        self.added_cond_kwargs = {
            "text_embeds": self.add_text_embeds.to(device),
            "time_ids": self.add_time_ids,
        }

        generator = torch.Generator(device=device).manual_seed(seed)
        self.noise = torch.randn(
            (1, 4, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor),
            generator=generator,
            device=device,
            dtype=self.dtype,
        )
        self.static_input = torch.empty((1, 3, height, width), device=device, dtype=self.dtype)
        self.static_output = None
        self.graph = None

    def forward(self, image, general_scheduler=False):
        pipe = self.pipe
        latents = pipe.vae.encode(image).latents
        latents = latents.to(self.dtype) * pipe.vae.config.scaling_factor
        if self.use_single_step_scheduler and not general_scheduler:
            latents = latents + self.noise * self.sigma
            latent_model_input = latents / self.sigma_scale
            noise_pred = pipe.unet(
                latent_model_input,
                self.timesteps[0],
                encoder_hidden_states=self.prompt_embeds,
                added_cond_kwargs=self.added_cond_kwargs,
                return_dict=False,
            )[0]
            latents = (latents.float() - self.sigma_f32 * noise_pred.float()).to(self.dtype)
        else:
            pipe.scheduler._step_index = None
            latents = pipe.scheduler.add_noise(latents, self.noise, self.latent_timestep)
            for t in self.timesteps:
                latent_model_input = pipe.scheduler.scale_model_input(latents, t)
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=self.prompt_embeds,
                    added_cond_kwargs=self.added_cond_kwargs,
                    return_dict=False,
                )[0]
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        return (decoded * 0.5 + 0.5).clamp(0, 1)

    def capture(self, image, general_scheduler=False):
        self.static_input.copy_(image)
        for _ in range(3):
            self.static_output = self.forward(self.static_input, general_scheduler=general_scheduler)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.static_output = self.forward(self.static_input, general_scheduler=general_scheduler)
        self.graph = graph
        torch.cuda.synchronize()

    def replay(self, image=None):
        if image is not None:
            self.static_input.copy_(image)
        self.graph.replay()
        return self.static_output


def run_original(pipe, image_np, prompt, args):
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with torch.inference_mode():
        return pipe(
            prompt=prompt,
            image=[image_np],
            num_inference_steps=args.steps,
            strength=args.strength,
            guidance_scale=0.0,
            generator=generator,
            output_type="pil",
        ).images[0]


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


def main():
    args = parse_args()
    disable_progress_bar()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")

    image_np = make_test_image(args.height, args.width)
    input_tensor = to_input_tensor(image_np, device)

    original_comparison = None
    if args.save_comparison:
        original_pipe = build_original_pipeline(device)
        original_comparison = run_original(original_pipe, image_np, args.prompt, args)
        del original_pipe
        torch.cuda.empty_cache()

    pipe = build_pipeline(device, no_preserve_parameters=args.no_preserve_parameters)
    runner = ManualSDXLTurboImg2Img(
        pipe, args.prompt, args.height, args.width, args.steps, args.strength, args.seed, device
    )

    if args.mode == "graph":
        runner.capture(input_tensor, general_scheduler=args.general_scheduler)

    cpu_frames = [make_test_image(args.height, args.width) for _ in range(args.runs + args.warmup)]
    times = []
    last = None
    for i in range(args.warmup + args.runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            if args.include_upload:
                frame = to_input_tensor(cpu_frames[i], device)
            else:
                frame = input_tensor
            if args.mode == "graph":
                last = runner.replay(frame)
            else:
                last = runner.forward(frame, general_scheduler=args.general_scheduler)
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
            "mode": args.mode,
            "include_upload": args.include_upload,
            "include_download": args.include_download,
            "general_scheduler": args.general_scheduler,
            "single_step_scheduler": runner.use_single_step_scheduler and not args.general_scheduler,
            "sfast_preserve_parameters": not args.no_preserve_parameters,
            "effective_denoise_steps": int(runner.effective_steps),
            "timesteps": [float(t.detach().cpu()) for t in runner.timesteps],
        }
    )

    if args.save_comparison:
        fast = tensor_to_pil(last)
        comparison = Image.new("RGB", (args.width * 2, args.height), "black")
        comparison.paste(original_comparison.convert("RGB"), (0, 0))
        comparison.paste(fast.convert("RGB"), (args.width, 0))
        draw = ImageDraw.Draw(comparison)
        draw.rectangle((0, 0, args.width, 42), fill=(0, 0, 0))
        draw.rectangle((args.width, 0, args.width * 2, 42), fill=(0, 0, 0))
        draw.text((16, 12), "Original Diffusers pipeline", fill=(255, 255, 255))
        draw.text((args.width + 16, 12), f"Fast {args.mode} pipeline", fill=(255, 255, 255))
        args.save_comparison.parent.mkdir(parents=True, exist_ok=True)
        comparison.save(args.save_comparison)
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
