"""Measure actual model size, RSS, VRAM, and GPU energy per format/component."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quantization-pluse-mse" / "src"))

import psutil
import pynvml
import torch

from codec.checkpoint import load_model
from codec.images import read_image
from codec.runtime import compress_image, configure_runtime, decompress_image
from pulsequant.core import (
    COMPONENTS, MAIN_SWEEP, apply_quant, component_size, conv_modules, weight_bits,
)

FIELDS = [
    "format", "component", "qp", "image", "repeat_seconds", "repeats",
    "model_fp32_bytes", "component_fp32_bytes", "component_packed_bytes",
    "model_packed_bytes", "size_reduction_pct",
    "rss_clean_mb", "rss_loaded_mb", "rss_after_run_mb", "rss_after_clean_mb",
    "vram_clean_mb", "vram_loaded_mb", "vram_peak_alloc_mb",
    "vram_peak_reserved_mb", "vram_after_mb",
    "energy_j", "energy_raw_j", "energy_j_per_image", "energy_j_per_mp",
    "idle_energy_j", "idle_power_w", "elapsed_s",
    "energy_kind", "power_w", "sm_clock_mhz", "mem_clock_mhz", "temp_c",
    "status", "error",
]


def rss_mb():
    return psutil.Process().memory_info().rss / 1024**2


def vram_mb():
    return {
        "alloc": torch.cuda.memory_allocated() / 1024**2,
        "reserved": torch.cuda.memory_reserved() / 1024**2,
    }


def gpu_state(handle):
    return {
        "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
        "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
        "mem_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
        "temp_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
    }


def energy_mj(handle):
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)


def packed_component_bytes(model, component, fmt):
    total = 0
    for name, module in conv_modules(model, component):
        weight = module.weight.detach().float()
        numel = weight.numel()
        out_channels = weight.shape[0]
        if fmt in {"fp16", "bf16"}:
            total += numel * 2
        elif fmt in {"fp8_e4m3", "fp8_e5m2", "int8"}:
            total += numel + out_channels * 4
        elif fmt in {"int4", "fp4_e2m1"}:
            total += (numel + 1) // 2 + out_channels * 4
        else:
            total += numel * 4
    return total


def model_bytes(model):
    return (
        sum(p.numel() * p.element_size() for p in model.parameters())
        + sum(b.numel() * b.element_size() for b in model.buffers())
    )


def run_repeats(model, image, qp, seconds):
    count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        data, _ = compress_image(model, image, qp)
        decompress_image(model, data)
        count += 1
    return count, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/pulse-mse")
    parser.add_argument("--formats", nargs="+", default=[
        "fp32", "fp16", "bf16", "int8", "int4",
        "fp8_e4m3", "fp8_e5m2", "fp4_e2m1",
    ])
    parser.add_argument("--components", nargs="+", default=["all"])
    parser.add_argument("--qps", type=int, nargs="+", default=[0, 3, 7])
    parser.add_argument("--image", default="datasets/kodak/kodim01.png")
    parser.add_argument("--repeat-seconds", type=float, default=10.0)
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default="quantization-pluse-mse/results/memory_energy.csv")
    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    components = MAIN_SWEEP if "all" in args.components else args.components
    configure_runtime(args.device, args.threads)
    pynvml.nvmlInit()
    nvml = pynvml.nvmlDeviceGetHandleByIndex(0)

    rss_clean = rss_mb()
    torch.cuda.reset_peak_memory_stats()
    vram_clean = vram_mb()
    model = load_model(args.model, args.device)
    rss_loaded = rss_mb()
    vram_loaded = vram_mb()
    base_bytes = model_bytes(model)
    image = read_image(args.image, args.device)
    rows = []
    runs = 0

    for fmt in args.formats:
        for component in components:
            for qp in args.qps:
                if args.max_runs is not None and runs >= args.max_runs:
                    break
                runs += 1
                row = {
                    "format": fmt, "component": component, "qp": qp,
                    "image": str(args.image), "repeat_seconds": args.repeat_seconds,
                    "repeats": 0, "model_fp32_bytes": base_bytes,
                    "rss_clean_mb": rss_clean, "rss_loaded_mb": rss_loaded,
                    "vram_clean_mb": vram_clean["alloc"],
                    "vram_loaded_mb": vram_loaded["alloc"],
                    "energy_kind": "real" if fmt == "fp32" else "fake_quant",
                    "status": "ok", "error": "",
                }
                handle = None
                try:
                    if fmt == "fp32":
                        handle = apply_quant(model, component, control="identity")
                    else:
                        handle = apply_quant(model, component, fmt=fmt)
                    size = component_size(model, component, fmt)
                    packed = packed_component_bytes(model, component, fmt)
                    row.update({
                        "component_fp32_bytes": size["fp32_bytes"],
                        "component_packed_bytes": packed,
                        "model_packed_bytes": base_bytes - size["fp32_bytes"] + packed,
                        "size_reduction_pct": 100.0 * (1.0 - (base_bytes - size["fp32_bytes"] + packed) / base_bytes),
                        "rss_after_run_mb": rss_mb(),
                        "vram_peak_alloc_mb": vram_mb()["alloc"],
                    })
                    if not args.no_energy:
                        e0 = energy_mj(nvml)
                        idle0 = time.perf_counter()
                        time.sleep(args.idle_seconds)
                        idle_energy = (energy_mj(nvml) - e0) / 1000.0
                        e1 = energy_mj(nvml)
                        count, elapsed = run_repeats(model, image, qp, args.repeat_seconds)
                        e2 = energy_mj(nvml)
                        state = gpu_state(nvml)
                        raw = (e2 - e1) / 1000.0
                        idle_power = idle_energy / max(args.idle_seconds, 1e-6)
                        energy = raw - idle_power * elapsed
                        pixels = image.shape[-2] * image.shape[-1]
                        row.update({
                            "repeats": count,
                            "energy_j": energy,
                            "energy_raw_j": raw,
                            "energy_j_per_image": energy / max(count, 1),
                            "energy_j_per_mp": energy / max(count * pixels / 1e6, 1e-12),
                            "idle_energy_j": idle_energy,
                            "idle_power_w": idle_power,
                            "elapsed_s": elapsed,
                            **state,
                        })
                        if energy < 0:
                            row["status"] = "noise"
                            row["error"] = "idle-subtracted energy is negative"
                    row["vram_peak_alloc_mb"] = torch.cuda.max_memory_allocated() / 1024**2
                    row["vram_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024**2
                except Exception as error:
                    row.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
                finally:
                    if handle is not None:
                        handle.restore()
                    gc.collect()
                    torch.cuda.empty_cache()
                    row["rss_after_clean_mb"] = rss_mb()
                    row["vram_after_mb"] = vram_mb()["alloc"]
                rows.append(row)
                print(json.dumps({k: row.get(k) for k in (
                    "format", "component", "qp", "component_packed_bytes",
                    "model_packed_bytes", "energy_j", "status")}))
            if args.max_runs is not None and runs >= args.max_runs:
                break
        if args.max_runs is not None and runs >= args.max_runs:
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    pynvml.nvmlShutdown()
    print(json.dumps({"rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
