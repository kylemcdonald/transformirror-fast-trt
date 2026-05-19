#!/usr/bin/env python3
import argparse
from pathlib import Path

import tensorrt as trt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_engine(onnx_path, engine_path, workspace_bytes, verbose):
    if engine_path.exists():
        print(f"keeping existing {engine_path}", flush=True)
        return
    severity = trt.Logger.VERBOSE if verbose else trt.Logger.INFO
    logger = trt.Logger(severity)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    print(f"parsing {onnx_path}", flush=True)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_bytes))
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"building {engine_path}", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {engine_path}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    print(f"wrote {engine_path} ({engine_path.stat().st_size / (1024 * 1024):.1f} MiB)", flush=True)


def main():
    args = parse_args()
    workspace_bytes = args.workspace_gb * 1024 * 1024 * 1024
    for name in ("taesdxl_encode", "taesdxl_decode", "sdxl_turbo_unet"):
        build_engine(
            args.onnx_dir / f"{name}.onnx",
            args.engine_dir / f"{name}.plan",
            workspace_bytes,
            args.verbose,
        )


if __name__ == "__main__":
    main()
