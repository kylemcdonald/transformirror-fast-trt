#!/usr/bin/env python3
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny
from diffusers.utils.logging import disable_progress_bar

from fastest_sdxl_turbo_img2img import PROMPT

MAX_DENOISE_STEPS = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("cpp_assets"))
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--idle-exit-sec", type=float, default=0.0)
    return parser.parse_args()


def write_bytes_atomic(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_tensor(path, tensor):
    array = tensor.detach().contiguous().cpu().numpy()
    write_bytes_atomic(path, array.tobytes())


class ConditioningGenerator:
    def __init__(self, width, height):
        disable_progress_bar()
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        self.width = width
        self.height = height
        self.device = torch.device("cuda:0")
        started = time.perf_counter()
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True,
        )
        self.pipe.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.text_encoder.to(device=self.device, dtype=torch.float16)
        self.pipe.text_encoder_2.to(device=self.device, dtype=torch.float16)
        torch.cuda.synchronize()
        self.load_ms = (time.perf_counter() - started) * 1000.0
        self.last_prompt = None
        self.last_prompt_embeds = None
        self.last_pooled_prompt_embeds = None

        try:
            least_priority, _ = torch.cuda.get_stream_priority_range()
            self.encode_stream = torch.cuda.Stream(priority=least_priority)
        except Exception:
            self.encode_stream = None

        # Pay first-use kernel/setup costs before the first real prompt edit.
        self.generate(
            {
                "prompt": PROMPT,
                "seed": 0,
                "strength": 0.7,
                "steps": 2,
                "width": self.width,
                "height": self.height,
            },
            None,
        )

    def encode_prompt(self, prompt):
        if prompt == self.last_prompt:
            return self.last_prompt_embeds, self.last_pooled_prompt_embeds, 0.0, True

        started = time.perf_counter()
        with torch.inference_mode():
            if self.encode_stream is None:
                prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                    prompt=prompt,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                torch.cuda.synchronize()
            else:
                with torch.cuda.stream(self.encode_stream):
                    prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                        prompt=prompt,
                        device=self.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                self.encode_stream.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.last_prompt = prompt
        self.last_prompt_embeds = prompt_embeds
        self.last_pooled_prompt_embeds = pooled_prompt_embeds
        return prompt_embeds, pooled_prompt_embeds, elapsed_ms, False

    def generate(self, request, out_dir):
        started = time.perf_counter()
        prompt = str(request.get("prompt", PROMPT))
        seed = int(request.get("seed", 0))
        strength = float(request.get("strength", 0.7))
        steps = int(request.get("steps", 2))
        width = int(request.get("width", self.width))
        height = int(request.get("height", self.height))
        if width != self.width or height != self.height:
            raise ValueError(f"worker was started for {self.width}x{self.height}, got {width}x{height}")

        prompt_embeds, pooled_prompt_embeds, encode_ms, cache_hit = self.encode_prompt(prompt)

        scheduler_started = time.perf_counter()
        self.pipe.scheduler.set_timesteps(steps, device=self.device)
        timesteps, effective_steps = self.pipe.get_timesteps(steps, strength, self.device)
        if len(timesteps) < 1:
            raise ValueError(f"this fast path expects at least one effective denoise step, got {len(timesteps)}")
        if len(timesteps) > MAX_DENOISE_STEPS:
            raise ValueError(f"this fast path supports at most {MAX_DENOISE_STEPS} effective denoise steps, got {len(timesteps)}")
        sigmas = self.pipe.scheduler.sigmas[
            self.pipe.scheduler.begin_index : self.pipe.scheduler.begin_index + len(timesteps) + 1
        ].to(device=self.device, dtype=torch.float32)
        sigma_scales = torch.sqrt(sigmas[:-1] * sigmas[:-1] + 1.0)
        inv_sigma_scales = torch.reciprocal(sigma_scales)
        step_params = torch.stack((sigmas[:-1], sigmas[1:], inv_sigma_scales), dim=1).contiguous()

        projection_dim = (
            int(pooled_prompt_embeds.shape[-1])
            if self.pipe.text_encoder_2 is None
            else self.pipe.text_encoder_2.config.projection_dim
        )
        time_ids, _ = self.pipe._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            6.0,
            2.5,
            (height, width),
            (0, 0),
            (height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=projection_dim,
        )
        time_ids = time_ids.to(self.device)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(
            (1, 4, height // self.pipe.vae_scale_factor, width // self.pipe.vae_scale_factor),
            generator=generator,
            device=self.device,
            dtype=torch.float16,
        )
        timestep = timesteps[:1].to(device=self.device, dtype=torch.float32)
        timesteps_f32 = timesteps.to(device=self.device, dtype=torch.float32).contiguous()
        torch.cuda.synchronize()
        scheduler_ms = (time.perf_counter() - scheduler_started) * 1000.0

        save_ms = 0.0
        if out_dir is not None:
            save_started = time.perf_counter()
            out_dir.mkdir(parents=True, exist_ok=True)
            save_tensor(out_dir / "noise.fp16", noise)
            save_tensor(out_dir / "prompt_embeds.fp16", prompt_embeds)
            save_tensor(out_dir / "text_embeds.fp16", pooled_prompt_embeds)
            save_tensor(out_dir / "time_ids.fp16", time_ids)
            save_tensor(out_dir / "timestep.f32", timestep)
            timesteps_array = np.zeros((MAX_DENOISE_STEPS,), dtype=np.float32)
            timesteps_array[: len(timesteps)] = timesteps_f32.detach().cpu().numpy()
            step_params_array = np.zeros((MAX_DENOISE_STEPS, 3), dtype=np.float32)
            step_params_array[: len(timesteps), :] = step_params.detach().cpu().numpy()
            write_bytes_atomic(out_dir / "timesteps.f32", timesteps_array.tobytes())
            write_bytes_atomic(out_dir / "step_params.f32", step_params_array.tobytes())
            write_bytes_atomic(out_dir / "step_count.i32", np.asarray([len(timesteps)], dtype=np.int32).tobytes())
            params = np.asarray(
                [
                    float(sigmas[0].detach().cpu()),
                    float(inv_sigma_scales[0].detach().cpu()),
                    float(self.pipe.vae.config.scaling_factor),
                    float(1.0 / self.pipe.vae.config.scaling_factor),
                ],
                dtype=np.float32,
            )
            write_bytes_atomic(out_dir / "params.f32", params.tobytes())
            save_ms = (time.perf_counter() - save_started) * 1000.0

        metadata = {
            "width": width,
            "height": height,
            "steps": steps,
            "strength": strength,
            "prompt": prompt,
            "seed": seed,
            "effective_steps": int(effective_steps),
            "timesteps": [float(t.detach().cpu()) for t in timesteps],
            "sigmas": [float(s.detach().cpu()) for s in sigmas],
            "step_params": step_params.detach().cpu().numpy().tolist(),
            "scaling_factor": float(self.pipe.vae.config.scaling_factor),
            "cache_hit": cache_hit,
            "load_ms": self.load_ms,
            "encode_ms": encode_ms,
            "scheduler_ms": scheduler_ms,
            "save_ms": save_ms,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "shapes": {
                "noise": list(noise.shape),
                "prompt_embeds": list(prompt_embeds.shape),
                "text_embeds": list(pooled_prompt_embeds.shape),
                "time_ids": list(time_ids.shape),
                "timestep": list(timestep.shape),
            },
        }
        if out_dir is not None:
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
        return metadata


def parent_is_alive(parent_pid):
    if parent_pid <= 0:
        return True
    try:
        os.kill(parent_pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def socket_path_matches(path, owner_stat):
    try:
        current = path.stat()
    except FileNotFoundError:
        return False
    return current.st_dev == owner_stat.st_dev and current.st_ino == owner_stat.st_ino


def recv_json_line(conn):
    chunks = []
    while True:
        data = conn.recv(65536)
        if not data:
            break
        chunks.append(data)
        if b"\n" in data:
            break
    raw = b"".join(chunks).split(b"\n", 1)[0]
    if not raw:
        raise ValueError("empty request")
    return json.loads(raw.decode("utf-8"))


def main():
    args = parse_args()
    generator = ConditioningGenerator(args.width, args.height)
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    try:
        args.socket.unlink()
    except FileNotFoundError:
        pass

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.settimeout(0.5)
    server.bind(str(args.socket))
    server.listen(8)
    socket_owner = args.socket.stat()
    print(
        json.dumps(
            {
                "status": "ready",
                "socket": str(args.socket),
                "load_ms": generator.load_ms,
                "pid": os.getpid(),
            }
        ),
        file=sys.stderr,
        flush=True,
    )

    last_request = time.monotonic()
    try:
        while parent_is_alive(args.parent_pid):
            if args.idle_exit_sec > 0 and time.monotonic() - last_request > args.idle_exit_sec:
                break
            if not socket_path_matches(args.socket, socket_owner):
                break
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            with conn:
                last_request = time.monotonic()
                try:
                    request = recv_json_line(conn)
                    metadata = generator.generate(request, args.out_dir)
                    response = {"ok": True, **metadata}
                except Exception as exc:
                    response = {"ok": False, "error": str(exc)}
                conn.sendall((json.dumps(response, separators=(",", ":")) + "\n").encode("utf-8"))
    finally:
        server.close()
        try:
            args.socket.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
