#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny
from diffusers.utils.logging import disable_progress_bar


DEFAULT_PROMPT = "a vivid mirrored portrait, sharp detail, luminous color"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SDXL Turbo img2img with TAESDXL and optional stable-fast."
    )
    parser.add_argument("--model", default="stabilityai/sdxl-turbo")
    parser.add_argument("--vae", default="madebyollin/taesdxl")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input", choices=("random", "zeros"), default="random")
    parser.add_argument("--output-type", choices=("np", "pil", "latent"), default="np")
    parser.add_argument("--save-output", type=Path)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--no-stable-fast", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--triton", action="store_true")
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument("--fused-geglu", action="store_true")
    parser.add_argument("--sfast-no-jit", action="store_true")
    parser.add_argument("--sfast-no-jit-freeze", action="store_true")
    parser.add_argument("--sfast-no-cnn-optimization", action="store_true")
    parser.add_argument("--sfast-no-lowp-gemm", action="store_true")
    parser.add_argument("--sfast-no-memory-format", action="store_true")
    parser.add_argument("--no-cached-prompt", action="store_true")
    parser.add_argument("--compile-before-to", action="store_true")
    parser.add_argument("--no-preserve-parameters", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def is_rtx_5090(device):
    try:
        return "5090" in torch.cuda.get_device_name(device)
    except RuntimeError:
        return False


def make_frame(height, width, mode, seed):
    if mode == "zeros":
        return np.zeros((height, width, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    return rng.random((height, width, 3), dtype=np.float32)


def build_pipeline(args):
    local_files_only = not args.allow_downloads
    pipe = AutoPipelineForImage2Image.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=local_files_only,
    )
    pipe.vae = AutoencoderTiny.from_pretrained(
        args.vae,
        torch_dtype=torch.float16,
        local_files_only=local_files_only,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def compile_with_stable_fast(pipe, args, device):
    from sfast.compilers.diffusion_pipeline_compiler import (
        CompilationConfig,
        compile as compile_pipeline,
    )

    config = CompilationConfig.Default()
    config.enable_cuda_graph = args.cuda_graph
    config.enable_triton = args.triton
    config.enable_xformers = args.xformers
    config.enable_fused_linear_geglu = args.fused_geglu
    if args.sfast_no_jit:
        config.enable_jit = False
    if args.sfast_no_jit_freeze:
        config.enable_jit_freeze = False
    if args.sfast_no_cnn_optimization:
        config.enable_cnn_optimization = False
    if args.sfast_no_lowp_gemm:
        config.prefer_lowp_gemm = False
    if args.sfast_no_memory_format:
        config.memory_format = None
    if args.no_preserve_parameters:
        config.preserve_parameters = False

    if is_rtx_5090(device):
        config.enable_xformers = False
        config.enable_fused_linear_geglu = False

    print(f"stable-fast config: {config}", flush=True)
    return compile_pipeline(pipe, config=config), config


def prepare_prompt_kwargs(pipe, prompt, device):
    with torch.inference_mode():
        encoded = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    prompt_embeds, _, pooled_prompt_embeds, _ = encoded
    return {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }


def timed_call(pipe, image, prompt_kwargs, generator, args):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = pipe(
            image=[image],
            num_inference_steps=args.steps,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type=args.output_type,
            **prompt_kwargs,
        ).images
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0, output


def summarize(times):
    sorted_times = sorted(times)
    return {
        "count": len(times),
        "mean_ms": statistics.fmean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "p90_ms": sorted_times[int(0.9 * (len(sorted_times) - 1))],
    }


def maybe_save(output, path):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image = output[0]
    if hasattr(image, "save"):
        image.save(path)
        return
    if isinstance(image, np.ndarray):
        from PIL import Image

        array = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(array).save(path)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    disable_progress_bar()

    device = torch.device(args.device)
    print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}", flush=True)
    print(f"gpu: {torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"benchmark: {args.width}x{args.height}, strength={args.strength}, "
        f"steps={args.steps}, output_type={args.output_type}",
        flush=True,
    )

    pipe = build_pipeline(args)

    sfast_config = None
    if args.compile_before_to and not args.no_stable_fast:
        pipe, sfast_config = compile_with_stable_fast(pipe, args, device)

    pipe.to(device=device, dtype=torch.float16)

    if not args.compile_before_to and not args.no_stable_fast:
        pipe, sfast_config = compile_with_stable_fast(pipe, args, device)

    image = make_frame(args.height, args.width, args.input, args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    if args.no_cached_prompt:
        prompt_kwargs = {"prompt": args.prompt}
    else:
        prompt_kwargs = prepare_prompt_kwargs(pipe, args.prompt, device)

    last_output = None
    for i in range(args.warmup):
        elapsed, last_output = timed_call(pipe, image, prompt_kwargs, generator, args)
        print(f"warmup {i + 1}/{args.warmup}: {elapsed:.1f} ms", flush=True)

    times = []
    for i in range(args.runs):
        elapsed, last_output = timed_call(pipe, image, prompt_kwargs, generator, args)
        times.append(elapsed)
        print(f"run {i + 1:02d}/{args.runs}: {elapsed:.1f} ms", flush=True)

    result = summarize(times)
    result.update(
        {
            "model": args.model,
            "vae": args.vae,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "strength": args.strength,
            "guidance_scale": args.guidance_scale,
            "stable_fast": not args.no_stable_fast,
            "stable_fast_config": str(sfast_config) if sfast_config else None,
            "cached_prompt": not args.no_cached_prompt,
            "input": args.input,
            "output_type": args.output_type,
        }
    )

    maybe_save(last_output, args.save_output)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "summary: "
            f"mean={result['mean_ms']:.1f} ms, "
            f"median={result['median_ms']:.1f} ms, "
            f"min={result['min_ms']:.1f} ms, "
            f"p90={result['p90_ms']:.1f} ms, "
            f"max={result['max_ms']:.1f} ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
