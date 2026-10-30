"""Paired latency and energy comparison, robust to thermal drift.

The straightforward benchmark is unsafe on this laptop: repeating the *same* FP32 configuration
at the start and the end of a long run moved round-trip time from 248 ms to 308 ms (+25%) and
energy from 3.5 J to 4.3 J per image. Any comparison across sequentially measured configurations
therefore measures the drift, not the model.

This script interleaves configurations per image and round, so drift affects every candidate
equally, and reports paired deltas against FP32 with a bootstrap confidence interval.

    python quantization-pluse-mse/scripts/bench_interleaved.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quantization-pluse-mse" / "src"))
sys.path.insert(0, str(ROOT / "quantization-pluse-mse" / "scripts"))

import pynvml
import torch

from codec.checkpoint import load_model
from codec.images import image_paths, read_image
from codec.runtime import compress_image, configure_runtime, decompress_image
from bench_configs import make_context, resolve
from pulsequant.core import apply_assignment

FIELDS = ["config", "dataset", "qp", "image", "round", "encode_ms", "decode_ms",
          "roundtrip_ms", "energy_j", "status", "error"]


def bootstrap_ci(values: list[float], samples: int = 2000, alpha: float = 0.05):
    if len(values) < 3:
        return None, None
    random.seed(0)
    means = []
    for _ in range(samples):
        means.append(statistics.mean(random.choices(values, k=len(values))))
    means.sort()
    return means[int(alpha / 2 * samples)], means[int((1 - alpha / 2) * samples)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/pulse-mse")
    parser.add_argument("--dataset", default="kodak")
    parser.add_argument("--configs", nargs="+", default=["fp32", "ilp", "fp16_autocast", "bf16_autocast"])
    parser.add_argument("--qps", type=int, nargs="+", default=[0, 3, 7])
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--decision-dir", default="quantization-pluse-mse/results/decision")
    parser.add_argument("--ilp-dir", default="quantization-pluse-mse/results/ilp")
    parser.add_argument("--output", default="quantization-pluse-mse/results/metrics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    configure_runtime(args.device, args.threads)
    pynvml.nvmlInit()
    nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
    paths = image_paths(Path(f"datasets/{args.dataset}"))[: args.images]
    images = [(p.name, read_image(p, args.device)) for p in paths]
    base_model = load_model(args.model, args.device)

    resolved = {name: resolve(name, Path(args.decision_dir), Path(args.ilp_dir)) for name in args.configs}
    rows = []
    print(json.dumps({"dataset": args.dataset, "images": len(images), "rounds": args.rounds,
                      "configs": args.configs, "qps": args.qps}))

    for qp in args.qps:
        for round_index in range(args.rounds):
            for image_index, (image_name, image) in enumerate(images):
                order = list(args.configs)
                random.seed(hash((qp, round_index, image_index)) & 0xFFFF)
                random.shuffle(order)
                for name in order:
                    assignment, kind = resolved[name]
                    handle = None
                    row = {"config": name, "dataset": args.dataset, "qp": qp,
                           "image": image_name, "round": round_index, "status": "ok", "error": ""}
                    try:
                        if assignment:
                            handle = apply_assignment(base_model, assignment, control="none")
                        ctx = make_context(kind, name)
                        with ctx:
                            compress_image(base_model, image, qp, legacy=False)
                            decompress_image(base_model, compress_image(base_model, image, qp)[0])
                        torch.cuda.synchronize()
                        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(nvml)
                        with ctx:
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            data, _ = compress_image(base_model, image, qp, legacy=False)
                            torch.cuda.synchronize()
                            middle = time.perf_counter()
                            decompress_image(base_model, data)
                            torch.cuda.synchronize()
                            end = time.perf_counter()
                        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(nvml)
                        row.update({
                            "encode_ms": round((middle - start) * 1000.0, 3),
                            "decode_ms": round((end - middle) * 1000.0, 3),
                            "roundtrip_ms": round((end - start) * 1000.0, 3),
                            "energy_j": round((e1 - e0) / 1000.0, 4),
                        })
                    except Exception as error:
                        row.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
                    finally:
                        if handle is not None:
                            handle.restore()
                    rows.append(row)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"interleaved_{args.dataset}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    by_key = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        by_key[(row["config"], row["qp"], row["image"], row["round"])] = row
    summary = []
    for config in args.configs:
        if config == "fp32":
            continue
        for qp in args.qps:
            rt_delta, e_delta = [], []
            for (cfg, q, image, rnd), row in by_key.items():
                if cfg != config or q != qp:
                    continue
                base = by_key.get(("fp32", q, image, rnd))
                if base is None:
                    continue
                base_rt = float(base["roundtrip_ms"])
                base_e = float(base["energy_j"])
                if base_rt > 0 and base_e > 0:
                    rt_delta.append((float(row["roundtrip_ms"]) / base_rt - 1.0) * 100.0)
                    e_delta.append((float(row["energy_j"]) / base_e - 1.0) * 100.0)
            if not rt_delta:
                continue
            rt_low, rt_high = bootstrap_ci(rt_delta)
            e_low, e_high = bootstrap_ci(e_delta)
            summary.append({
                "config": config, "qp": qp, "pairs": len(rt_delta),
                "rt_delta_pct_median": round(statistics.median(rt_delta), 3),
                "rt_delta_ci_low": round(rt_low, 3) if rt_low is not None else "",
                "rt_delta_ci_high": round(rt_high, 3) if rt_high is not None else "",
                "energy_delta_pct_median": round(statistics.median(e_delta), 3),
                "energy_delta_ci_low": round(e_low, 3) if e_low is not None else "",
                "energy_delta_ci_high": round(e_high, 3) if e_high is not None else "",
            })
    with (out / f"interleaved_paired_{args.dataset}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]) if summary else ["config"])
        writer.writeheader()
        writer.writerows(summary)
    pynvml.nvmlShutdown()
    print(json.dumps({"rows": len(rows), "paired": len(summary),
                      "output": str(out / f'interleaved_{args.dataset}.csv')}))
    for entry in summary:
        print(json.dumps(entry))


if __name__ == "__main__":
    main()
