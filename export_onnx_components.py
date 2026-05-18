#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny
from diffusers.utils.logging import disable_progress_bar

from fastest_sdxl_turbo_img2img import ManualSDXLTurboImg2Img, make_test_image, to_input_tensor


PROMPT = "a cinematic mirror portrait, detailed face, luminous color, sharp focus"


class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )[0]


class VAEEncodeWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        return self.vae.encode(image).latents


class VAEDecodeWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(latents, return_dict=False)[0]


class FullPipelineWrapper(torch.nn.Module):
    def __init__(self, vae, unet, scaling_factor, sigma, sigma_scale):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.register_buffer("scaling_factor", scaling_factor)
        self.register_buffer("inv_scaling_factor", torch.reciprocal(scaling_factor))
        self.register_buffer("sigma", sigma)
        self.register_buffer("sigma_f32", sigma.float())
        self.register_buffer("inv_sigma_scale", torch.reciprocal(sigma_scale))

    def forward(self, image, noise, timestep, encoder_hidden_states, text_embeds, time_ids):
        latents = self.vae.encode(image).latents * self.scaling_factor
        latents = latents + noise * self.sigma
        latent_model_input = latents * self.inv_sigma_scale
        noise_pred = self.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )[0]
        latents = (latents.float() - self.sigma_f32 * noise_pred.float()).to(torch.float16)
        decoded = self.vae.decode(latents * self.inv_scaling_factor, return_dict=False)[0]
        return (decoded * 0.5 + 0.5).clamp(0, 1)


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
    pipe.unet.eval()
    pipe.vae.eval()
    return pipe


def export(module, inputs, path, input_names, output_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            module,
            inputs,
            str(path),
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
            external_data=True,
            dynamo=False,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("onnx"))
    parser.add_argument(
        "--component",
        choices=("unet", "vae-encode", "vae-decode", "full", "all"),
        default="all",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    disable_progress_bar()
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    pipe = build_original_pipeline(device)
    image = to_input_tensor(make_test_image(1024, 1024), device)
    runner = ManualSDXLTurboImg2Img(pipe, PROMPT, 1024, 1024, 2, 0.7, 0, device)

    latents = torch.empty((1, 4, 128, 128), device=device, dtype=torch.float16)
    timestep = runner.timesteps[:1].to(device=device, dtype=torch.float32)
    sigma = runner.sigma.reshape(())
    sigma_scale = runner.sigma_scale.reshape(())

    if args.component in ("vae-encode", "all"):
        export(
            VAEEncodeWrapper(pipe.vae),
            (image,),
            args.out_dir / "taesdxl_encode.onnx",
            ["image"],
            ["latents"],
        )

    if args.component in ("vae-decode", "all"):
        export(
            VAEDecodeWrapper(pipe.vae),
            (latents,),
            args.out_dir / "taesdxl_decode.onnx",
            ["latents"],
            ["image"],
        )

    if args.component in ("unet", "all"):
        export(
            UNetWrapper(pipe.unet),
            (
                latents,
                timestep,
                runner.prompt_embeds,
                runner.added_cond_kwargs["text_embeds"],
                runner.added_cond_kwargs["time_ids"],
            ),
            args.out_dir / "sdxl_turbo_unet.onnx",
            ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
            ["noise_pred"],
        )

    if args.component == "full":
        export(
            FullPipelineWrapper(pipe.vae, pipe.unet, torch.tensor(pipe.vae.config.scaling_factor, device=device, dtype=torch.float16), sigma, sigma_scale),
            (
                image,
                runner.noise,
                timestep,
                runner.prompt_embeds,
                runner.added_cond_kwargs["text_embeds"],
                runner.added_cond_kwargs["time_ids"],
            ),
            args.out_dir / "sdxl_turbo_full_img2img.onnx",
            ["image", "noise", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
            ["image_out"],
        )


if __name__ == "__main__":
    main()
