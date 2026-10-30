"""Evaluation, metrics, manifests, and progress logging."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from codec.images import image_paths, read_image, to_uint8
from codec.runtime import compress_image, configure_runtime, decompress_image


def configure_eval(device: str = "cuda", threads: int = 1, tf32: bool = False) -> dict:
    configure_runtime(device, threads)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = not bool(tf32)
    return {
        "device": device,
        "threads": int(threads),
        "tf32": bool(tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }


def stream_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def evaluate_image(model, image, qp: int, *, legacy: bool = False) -> dict:
    data, encode_ms = compress_image(model, image, qp, legacy=legacy)
    reconstruction, decode_ms = decompress_image(model, data, allow_legacy=legacy)
    pixels_u8 = to_uint8(reconstruction)
    quantized = pixels_u8.float() / 255
    mse = (quantized - image).square().mean().item()
    psnr = -10 * math.log10(max(mse, 1e-12))
    pixels = image.shape[-2] * image.shape[-1]
    return {
        "bytes": len(data),
        "pixels": pixels,
        "bpp": 8 * len(data) / pixels,
        "mse": mse,
        "psnr": psnr,
        "encode_ms": encode_ms,
        "decode_ms": decode_ms,
        "stream_hash": stream_sha256(data),
        "recon_hash": hashlib.sha256(pixels_u8.cpu().contiguous().numpy().tobytes()).hexdigest(),
        "decode_ok": True,
        "image_height": int(image.shape[-2]),
        "image_width": int(image.shape[-1]),
    }


def decode_fixed(model, y_hat, qp: int) -> torch.Tensor:
    with torch.no_grad():
        return model.decode(y_hat, qp=qp)


def output_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def delta_metrics(psnr: float, bpp: float, ref_psnr: float, ref_bpp: float, slope: float | None) -> dict:
    d_psnr = float(psnr - ref_psnr)
    d_bpp_pct = 100.0 * (float(bpp) / float(ref_bpp) - 1.0)
    if slope is None or slope == 0:
        penalty_pct = None
    else:
        penalty_pct = 100.0 * ((1.0 + d_bpp_pct / 100.0) * 2.0 ** (-d_psnr / float(slope)) - 1.0)
    return {"d_psnr": d_psnr, "d_bpp_pct": d_bpp_pct, "penalty_pct": penalty_pct}


def slopes_from_curve(points: list[dict]) -> dict[int, float]:
    ordered = sorted(points, key=lambda row: int(row["qp"]))
    if len(ordered) < 2:
        return {}
    xs = [math.log2(float(row["bpp"])) for row in ordered]
    ys = [float(row["psnr"]) for row in ordered]
    slopes = {}
    for index, row in enumerate(ordered):
        if index == 0:
            dx, dy = xs[1] - xs[0], ys[1] - ys[0]
        elif index == len(ordered) - 1:
            dx, dy = xs[-1] - xs[-2], ys[-1] - ys[-2]
        else:
            dx, dy = xs[index + 1] - xs[index - 1], ys[index + 1] - ys[index - 1]
        slopes[int(row["qp"])] = dy / dx if dx else 0.0
    return slopes


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_state() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def package_versions() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
    }


def hardware_info() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "name": properties.name,
        "capability": f"{properties.major}.{properties.minor}",
        "total_memory_mb": round(properties.total_memory / (1024 ** 2), 1),
    }


def build_manifest(*, model_path: str | Path, config: dict, command: str) -> dict:
    model_path = Path(model_path)
    return {
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path / "model.pt"),
        "entropy_control_sha256": sha256_file(model_path / "entropy_control_int.pt"),
        "config_sha256": sha256_file(model_path / "config.json"),
        "quant_config": config,
        "quant_config_sha256": sha256_json(config),
        "git": git_state(),
        "packages": package_versions(),
        "hardware": hardware_info(),
        "command": command,
    }


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


__all__ = [
    "append_jsonl", "build_manifest", "configure_eval", "decode_fixed", "delta_metrics",
    "evaluate_image", "git_state", "hardware_info", "image_paths", "output_delta",
    "package_versions", "read_image", "sha256_file", "sha256_json", "slopes_from_curve",
    "stream_sha256", "write_json",
]
