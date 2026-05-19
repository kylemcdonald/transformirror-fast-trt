import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fastest_sdxl_turbo_img2img import to_input_tensor
from trt_sdxl_turbo_img2img import PROMPT, TRTSDXLTurboImg2Img


class DiffusionProcessor:
    def __init__(self, warmup=None, local_files_only=True, gpu_id=0, **kwargs):
        self.local_files_only = local_files_only
        self.device = torch.device(f"cuda:{gpu_id}")
        self.width, self.height = self._parse_warmup(warmup)
        self.engine_root = Path(kwargs.get("engine_root") or os.environ.get("TRANSFORMIRROR_TRT_ENGINE_ROOT", "trt_engines"))
        self.runner = None
        self.runner_key = None

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)

        prompt = kwargs.get("prompt", PROMPT)
        steps = int(kwargs.get("steps", 2))
        strength = float(kwargs.get("strength", 0.7))
        seed = int(kwargs.get("seed", 0))
        blank = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self._build_runner(prompt, steps, strength, seed, blank)

    def _parse_warmup(self, warmup):
        if warmup:
            match = re.match(r"1x(?P<h>\d+)x(?P<w>\d+)x3", str(warmup))
            if match:
                return int(match.group("w")), int(match.group("h"))
        return 1280, 640

    def _engine_dir(self):
        engine_dir = self.engine_root / f"{self.width}x{self.height}"
        missing = [
            path.name
            for path in (
                engine_dir / "taesdxl_encode.plan",
                engine_dir / "sdxl_turbo_unet.plan",
                engine_dir / "taesdxl_decode.plan",
            )
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"missing TensorRT engines for {self.width}x{self.height} in {engine_dir}: {', '.join(missing)}"
            )
        return engine_dir

    def _build_runner(self, prompt, steps, strength, seed, image):
        key = (str(prompt), int(steps), float(strength), int(seed))
        if self.runner is not None and key == self.runner_key:
            return

        started = time.perf_counter()
        if self.runner is not None:
            del self.runner
            self.runner = None
            torch.cuda.empty_cache()

        args = SimpleNamespace(
            engine_dir=self._engine_dir(),
            width=self.width,
            height=self.height,
            steps=int(steps),
            strength=float(strength),
            prompt=str(prompt),
            seed=int(seed),
        )
        input_tensor = to_input_tensor(self._normalize_image(image), self.device)
        self.runner = TRTSDXLTurboImg2Img(args, self.device)
        self.runner.capture(input_tensor)
        self.runner_key = key
        elapsed = time.perf_counter() - started
        print(
            f"TensorRT runner ready for {self.width}x{self.height} "
            f"prompt/seed/strength/steps in {elapsed:.2f}s",
            flush=True,
        )

    def _normalize_image(self, image):
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32, copy=False)
        if image.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"expected {self.width}x{self.height} input, got {image.shape[1]}x{image.shape[0]}"
            )
        return np.ascontiguousarray(image)

    def run(self, images, prompt, num_inference_steps, strength, seed):
        if not images:
            return []
        image = self._normalize_image(images[0])
        self._build_runner(prompt, num_inference_steps, strength, seed, image)
        input_tensor = to_input_tensor(image, self.device)
        with torch.inference_mode():
            output = self.runner.replay(input_tensor)
            result = output[0].detach().permute(1, 2, 0).float().cpu().numpy()
        return [np.ascontiguousarray(result)]
