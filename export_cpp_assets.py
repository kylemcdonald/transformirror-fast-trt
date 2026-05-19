#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny
from diffusers.utils.logging import disable_progress_bar

from fastest_sdxl_turbo_img2img import PROMPT, make_test_image, to_input_tensor

MAX_DENOISE_STEPS = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("cpp_assets"))
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def save_tensor(path, tensor):
    array = tensor.detach().contiguous().cpu().numpy()
    path.write_bytes(array.tobytes())


def main():
    args = parse_args()
    disable_progress_bar()
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")

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

    with torch.inference_mode():
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=args.prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

    pipe.scheduler.set_timesteps(args.steps, device=device)
    timesteps, effective_steps = pipe.get_timesteps(args.steps, args.strength, device)
    if len(timesteps) < 1:
        raise ValueError(f"this fast path expects at least one effective denoise step, got {len(timesteps)}")
    if len(timesteps) > MAX_DENOISE_STEPS:
        raise ValueError(f"this fast path supports at most {MAX_DENOISE_STEPS} effective denoise steps, got {len(timesteps)}")
    sigmas = pipe.scheduler.sigmas[
        pipe.scheduler.begin_index : pipe.scheduler.begin_index + len(timesteps) + 1
    ].to(device=device, dtype=torch.float32)
    sigma_scales = torch.sqrt(sigmas[:-1] * sigmas[:-1] + 1.0)
    inv_sigma_scales = torch.reciprocal(sigma_scales)
    step_params = torch.stack((sigmas[:-1], sigmas[1:], inv_sigma_scales), dim=1).contiguous()

    projection_dim = (
        int(pooled_prompt_embeds.shape[-1])
        if pipe.text_encoder_2 is None
        else pipe.text_encoder_2.config.projection_dim
    )
    time_ids, _ = pipe._get_add_time_ids(
        (args.height, args.width),
        (0, 0),
        (args.height, args.width),
        6.0,
        2.5,
        (args.height, args.width),
        (0, 0),
        (args.height, args.width),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=projection_dim,
    )
    time_ids = time_ids.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(
        (1, 4, args.height // pipe.vae_scale_factor, args.width // pipe.vae_scale_factor),
        generator=generator,
        device=device,
        dtype=torch.float16,
    )

    input_image = to_input_tensor(make_test_image(args.height, args.width), device)
    timestep = timesteps[:1].to(device=device, dtype=torch.float32)
    timesteps_f32 = timesteps.to(device=device, dtype=torch.float32).contiguous()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_tensor(args.out_dir / "input_image.fp16", input_image)
    save_tensor(args.out_dir / "noise.fp16", noise)
    save_tensor(args.out_dir / "prompt_embeds.fp16", prompt_embeds)
    save_tensor(args.out_dir / "text_embeds.fp16", pooled_prompt_embeds)
    save_tensor(args.out_dir / "time_ids.fp16", time_ids)
    save_tensor(args.out_dir / "timestep.f32", timestep)
    timesteps_array = np.zeros((MAX_DENOISE_STEPS,), dtype=np.float32)
    timesteps_array[: len(timesteps)] = timesteps_f32.detach().cpu().numpy()
    step_params_array = np.zeros((MAX_DENOISE_STEPS, 3), dtype=np.float32)
    step_params_array[: len(timesteps), :] = step_params.detach().cpu().numpy()
    (args.out_dir / "timesteps.f32").write_bytes(timesteps_array.tobytes())
    (args.out_dir / "step_params.f32").write_bytes(step_params_array.tobytes())
    (args.out_dir / "step_count.i32").write_bytes(np.asarray([len(timesteps)], dtype=np.int32).tobytes())
    params = np.asarray(
        [
            float(sigmas[0].detach().cpu()),
            float(inv_sigma_scales[0].detach().cpu()),
            float(pipe.vae.config.scaling_factor),
            float(1.0 / pipe.vae.config.scaling_factor),
        ],
        dtype=np.float32,
    )
    (args.out_dir / "params.f32").write_bytes(params.tobytes())

    metadata = {
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "strength": args.strength,
        "prompt": args.prompt,
        "seed": args.seed,
        "effective_steps": int(effective_steps),
        "timesteps": [float(t.detach().cpu()) for t in timesteps],
        "sigmas": [float(s.detach().cpu()) for s in sigmas],
        "step_params": step_params.detach().cpu().numpy().tolist(),
        "scaling_factor": float(pipe.vae.config.scaling_factor),
        "shapes": {
            "input_image": list(input_image.shape),
            "noise": list(noise.shape),
            "prompt_embeds": list(prompt_embeds.shape),
            "text_embeds": list(pooled_prompt_embeds.shape),
            "time_ids": list(time_ids.shape),
            "timestep": list(timestep.shape),
        },
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
