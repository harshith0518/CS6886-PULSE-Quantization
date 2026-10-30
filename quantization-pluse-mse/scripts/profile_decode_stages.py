"""Decompose the decode path, and test whether the entropy bucket is real work.

The stage table says decode is entropy-dominated, but `model.decompress` bundles the entropy-model
network (hyper decoder, entropy-model steps), the C++ rANS, and any per-call table setup. This
separates those and tests three competing explanations:

  1. residual overhead: Python glue, copies, bitstream unpacking
  2. a fixed per-call setup cost, which would make encode and decode show the same ~80 ms
  3. genuine per-symbol coding work, which would scale with the symbol count

    python quantization-pluse-mse/scripts/profile_decode_stages.py
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import io
import json
import pstats
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

import MLCodec_extensions_cpp
from codec import bitstream
from codec.checkpoint import load_model
from codec.images import pad_image, read_image
from codec.models.entropy.y_cdf import resolve_y_cdf
from codec.runtime import compress_image, configure_runtime, decompress_image

FIELDS = ["device", "threads", "wxh", "mp", "symbols", "stage", "ms", "ns_per_symbol"]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def stamp(fn, repeats):
    fn()
    samples = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        out = fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/pulse-mse")
    parser.add_argument("--image", default="datasets/kodak/kodim01.png")
    parser.add_argument("--qp", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="quantization-pluse-mse/results/metrics")
    args = parser.parse_args()

    print(json.dumps({
        "extension_file": MLCodec_extensions_cpp.__file__,
        "extension_attrs": [a for a in dir(MLCodec_extensions_cpp) if not a.startswith("_")][:12],
    }))

    rows = []
    fits = {}
    sizes = [(256, 256), (384, 384), (512, 512)]
    for device, threads in (("cpu", 1), ("cuda", 1)):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        configure_runtime(device, threads)
        model = load_model(args.model, device).eval()
        full = read_image(args.image, device)
        measured = []

        for height, width in sizes:
            image = full[..., :height, :width].contiguous()
            with torch.inference_mode():
                data = compress_image(model, image, qp=args.qp)[0]
                metadata, payload = bitstream.unpack(data)
                resolve_y_cdf(model.entropy_model, metadata["y_cdf_mode"])
                latent = model.decompress(
                    payload, metadata["z_shape"], metadata["y_shape"], qp=metadata["qp"],
                    skip_index_cutoff=None if metadata["skip_index_cutoff"] < 0 else metadata["skip_index_cutoff"],
                    y_cdf_mode=metadata["y_cdf_mode"])
                recon = model.decode(latent, qp=metadata["qp"])
                h, w = metadata["height"], metadata["width"]
                mp = h * w / 1e6
                symbols = int(metadata["y_shape"][1] * metadata["y_shape"][2] * metadata["y_shape"][3])

                stages = {}
                stages["decode_total"], _ = stamp(lambda: decompress_image(model, data), args.repeats)
                stages["unpack"], _ = stamp(lambda: bitstream.unpack(data), args.repeats)
                stages["y_cdf"], _ = stamp(
                    lambda: resolve_y_cdf(model.entropy_model, metadata["y_cdf_mode"]), args.repeats)
                stages["decompress"], _ = stamp(lambda: model.decompress(
                    payload, metadata["z_shape"], metadata["y_shape"], qp=metadata["qp"],
                    skip_index_cutoff=None if metadata["skip_index_cutoff"] < 0 else metadata["skip_index_cutoff"],
                    y_cdf_mode=metadata["y_cdf_mode"]), args.repeats)
                stages["synthesis"], _ = stamp(lambda: model.decode(latent, qp=metadata["qp"]), args.repeats)
                stages["crop"], _ = stamp(lambda: recon[..., :h, :w], args.repeats)

                cold_start = time.perf_counter()
                sync()
                decompress_image(model, data)
                sync()
                cold = (time.perf_counter() - cold_start) * 1000.0

            for name, value in stages.items():
                rows.append({
                    "device": device, "threads": threads, "wxh": f"{w}x{h}", "mp": round(mp, 4),
                    "symbols": symbols, "stage": name, "ms": round(value, 2),
                    "ns_per_symbol": round(value * 1e6 / symbols, 2) if symbols else "",
                })
            rows.append({
                "device": device, "threads": threads, "wxh": f"{w}x{h}", "mp": round(mp, 4),
                "symbols": symbols, "stage": "cold_first_call", "ms": round(cold, 2),
                "ns_per_symbol": round(cold * 1e6 / symbols, 2),
            })
            accounted = stages["unpack"] + stages["decompress"] + stages["synthesis"] + stages["crop"]
            measured.append((symbols, mp, stages["decompress"], stages["synthesis"], stages["decode_total"]))
            print(json.dumps({
                "device": device, "wxh": f"{w}x{h}", "mp": round(mp, 4), "symbols": symbols,
                "total": round(stages["decode_total"], 2),
                "decompress": round(stages["decompress"], 2),
                "synthesis": round(stages["synthesis"], 2),
                "unpack": round(stages["unpack"], 2),
                "accounted": round(accounted, 2),
                "unaccounted": round(stages["decode_total"] - accounted, 2),
                "ns_per_symbol": round(stages["decompress"] * 1e6 / symbols, 2),
                "cold_first_call": round(cold, 2),
            }))

        if len(measured) >= 2:
            xs = [m[0] for m in measured]
            ys = [m[2] for m in measured]
            n = len(xs)
            mean_x, mean_y = sum(xs) / n, sum(ys) / n
            slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum((x - mean_x) ** 2 for x in xs)
            intercept = mean_y - slope * mean_x
            fits[device] = {"fixed_ms": round(intercept, 2),
                            "ns_per_extra_symbol": round(slope * 1e6, 2),
                            "r2_points": n,
                            "points": [(s, round(d, 1)) for s, _, d, _, _ in measured]}
            print(json.dumps({"fit": device, **fits[device]}))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    configure_runtime("cpu", 1)
    model = load_model(args.model, "cpu").eval()
    with torch.inference_mode():
        image = read_image(args.image, "cpu")[..., :512, :512].contiguous()
        data = compress_image(model, image, qp=args.qp)[0]
        profile = cProfile.Profile()
        profile.enable()
        for _ in range(3):
            decompress_image(model, data)
        profile.disable()
        buffer = io.StringIO()
        pstats.Stats(profile, stream=buffer).sort_stats("cumulative").print_stats(18)
        print(buffer.getvalue())

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "profile_decode_stages.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(out / "profile_decode_stages.csv")}))


if __name__ == "__main__":
    main()
