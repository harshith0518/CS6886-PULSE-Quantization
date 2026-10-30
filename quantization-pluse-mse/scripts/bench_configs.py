"""Latency, energy and memory for the whole-model configurations.

Measures the base model, every fake-quantized configuration, and an attempt at genuine
half-precision execution. Fake quantization keeps FP32 tensors and FP32 kernels, so it should
measure the same as the base model; genuine FP16/BF16 changes the kernels and should not.

    python quantization-pluse-mse/scripts/bench_configs.py --dataset kodak --qps 0 3 7
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quantization-pluse-mse" / "src"))

import pynvml
import torch

from codec.checkpoint import load_model
from codec.images import image_paths, read_image
from codec.runtime import compress_image, configure_runtime, decompress_image
from pulsequant.core import MAIN_SWEEP, apply_assignment

FIELDS = [
    "run_index", "config", "kind", "dataset", "qp", "images", "repeats",
    "encode_ms", "decode_ms", "roundtrip_ms", "ms_per_mp",
    "energy_j", "energy_j_per_image", "energy_mj_per_mp",
    "avg_power_w", "peak_vram_mb", "sm_clock_mhz", "temp_c",
    "status", "error",
]


def read_assignment(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("assignment", payload)


def resolve(name: str, decision_dir: Path, ilp_dir: Path) -> tuple[dict, str]:
    if name == "fp32":
        return {}, "fake_quant"
    if name.endswith("_native"):
        return {}, "native"
    if name.endswith("_autocast"):
        return {}, "autocast"
    if name in {"safe", "greedy"}:
        return read_assignment(decision_dir / f"decision_assignment_{name}.json"), "fake_quant"
    if name == "ilp":
        return read_assignment(ilp_dir / "assignment_ilp.json"), "fake_quant"
    if name in {"fp16", "bf16", "int8", "int4", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1"}:
        return {component: name for component in MAIN_SWEEP}, "fake_quant"
    raise SystemExit(f"unknown config {name!r}")


def make_context(kind: str, name: str):
    if kind != "autocast":
        return contextlib.nullcontext()
    dtype = torch.float16 if name.startswith("fp16") else torch.bfloat16
    return torch.autocast("cuda", dtype=dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/pulse-mse")
    parser.add_argument("--dataset", default="kodak")
    parser.add_argument("--configs", nargs="+", default=[
        "fp32", "fp16", "bf16", "int8", "safe", "greedy", "ilp", "fp16_native", "bf16_native",
    ])
    parser.add_argument("--qps", type=int, nargs="+", default=[0, 3, 7])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--decision-dir", default="quantization-pluse-mse/results/decision")
    parser.add_argument("--ilp-dir", default="quantization-pluse-mse/results/ilp")
    parser.add_argument("--output", default="quantization-pluse-mse/results/metrics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    configure_runtime(args.device, args.threads)
    pynvml.nvmlInit()
    nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
    energy_supported = True
    try:
        pynvml.nvmlDeviceGetTotalEnergyConsumption(nvml)
    except pynvml.NVMLError:
        energy_supported = False

    paths = image_paths(Path(f"datasets/{args.dataset}"))
    if args.max_images:
        paths = paths[: args.max_images]
    model = load_model(args.model, args.device)
    rows = []
    print(json.dumps({"dataset": args.dataset, "images": len(paths), "qps": args.qps,
                      "repeats": args.repeats, "energy_supported": energy_supported}))

    for name in args.configs:
        assignment, kind = resolve(name, Path(args.decision_dir), Path(args.ilp_dir))
        for qp in args.qps:
            row = {"run_index": len(rows), "config": name, "kind": kind, "dataset": args.dataset, "qp": qp,
                   "images": len(paths), "repeats": args.repeats, "status": "ok", "error": ""}
            handle = None
            try:
                if kind == "native":
                    dtype = torch.float16 if name.startswith("fp16") else torch.bfloat16
                    model = model.to(dtype)
                elif assignment:
                    handle = apply_assignment(model, assignment, control="none")

                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                ctx = make_context(kind, name)

                def do_encode(img):
                    with ctx:
                        return compress_image(model, img, qp, legacy=False)

                def do_decode(payload):
                    with ctx:
                        return decompress_image(model, payload, allow_legacy=False)

                encode_ms, decode_ms = [], []
                pixels_total = 0
                loaded = []
                for path in paths:
                    image = read_image(path, args.device)
                    loaded.append(image)
                    pixels_total += int(image.shape[-2] * image.shape[-1])
                    for _ in range(args.warmup):
                        data, _ = do_encode(image)
                        do_decode(data)
                    for _ in range(args.repeats):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        data, _ = do_encode(image)
                        torch.cuda.synchronize()
                        middle = time.perf_counter()
                        do_decode(data)
                        torch.cuda.synchronize()
                        end = time.perf_counter()
                        encode_ms.append((middle - start) * 1000.0)
                        decode_ms.append((end - middle) * 1000.0)

                mean_pixels = pixels_total / max(len(loaded), 1)
                total_ops = max(1, len(paths) * args.repeats)
                energy_j, power_max = 0.0, 0.0
                if energy_supported:
                    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(nvml)
                    power_max = pynvml.nvmlDeviceGetPowerUsage(nvml) / 1000.0
                    for image in loaded:
                        for _ in range(args.repeats):
                            data, _ = do_encode(image)
                            do_decode(data)
                    torch.cuda.synchronize()
                    e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(nvml)
                    energy_j = (e1 - e0) / 1000.0
                else:
                    power_max = pynvml.nvmlDeviceGetPowerUsage(nvml) / 1000.0

                roundtrip = statistics.median(encode_ms) + statistics.median(decode_ms)
                row.update({
                    "encode_ms": round(statistics.median(encode_ms), 4),
                    "decode_ms": round(statistics.median(decode_ms), 4),
                    "roundtrip_ms": round(roundtrip, 4),
                    "ms_per_mp": round(roundtrip / max(mean_pixels / 1e6, 1e-9), 4),
                    "energy_j": round(energy_j, 4) if energy_supported else "",
                    "energy_j_per_image": round(energy_j / total_ops, 6) if energy_supported else "",
                    "energy_mj_per_mp": round(1000.0 * energy_j / max(total_ops * mean_pixels / 1e6, 1e-9), 4)
                                        if energy_supported else "",
                    "avg_power_w": round(power_max, 3),
                    "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
                    "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(nvml, pynvml.NVML_CLOCK_SM),
                    "temp_c": pynvml.nvmlDeviceGetTemperature(nvml, pynvml.NVML_TEMPERATURE_GPU),
                })
            except Exception as error:
                row.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
            finally:
                if handle is not None:
                    handle.restore()
                if kind == "native":
                    model = load_model(args.model, args.device)
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in
                              ("config", "qp", "roundtrip_ms", "energy_j_per_image",
                               "peak_vram_mb", "status")}))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"bench_{args.dataset}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    pynvml.nvmlShutdown()
    print(json.dumps({"rows": len(rows), "output": str(out / f'bench_{args.dataset}.csv')}))


if __name__ == "__main__":
    main()
