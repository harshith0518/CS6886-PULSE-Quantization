"""Where the time actually goes: stage-level latency for encode and decode.

Splits each direction into the neural network and the entropy coder, and repeats the measurement
across devices and thread counts, because the operational point matters more than the model.

    python quantization-pluse-mse/scripts/profile_latency.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from codec.checkpoint import load_model
from codec.images import image_paths, pad_image, read_image
from codec.models.entropy.y_cdf import resolve_y_cdf
from codec.runtime import compress_image, configure_runtime, decompress_image

FIELDS = ["device", "threads", "image", "qp", "encode_total_ms", "encode_nn_ms",
          "encode_entropy_ms", "decode_total_ms", "decode_entropy_ms", "decode_nn_ms",
          "pixels_mp", "ms_per_mp_encode", "ms_per_mp_decode"]


def timed(fn, repeats):
    fn()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/pulse-mse")
    parser.add_argument("--dataset", default="kodak")
    parser.add_argument("--images", type=int, default=3)
    parser.add_argument("--qp", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="quantization-pluse-mse/results/metrics")
    args = parser.parse_args()

    paths = image_paths(Path(f"datasets/{args.dataset}"))[: args.images]
    rows = []

    for device, threads in (("cpu", 1), ("cpu", 4), ("cuda", 1), ("cuda", 4)):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        configure_runtime(device, threads)
        model = load_model(args.model, device).eval()
        for path in paths:
            image = read_image(path, device)
            padded = pad_image(image)
            with torch.inference_mode():
                result = model.compress(padded, qp=args.qp, skip_index_cutoff=2,
                                        y_cdf_mode="part_specific")
                payload = bytes(result["stream"])
                resolve_y_cdf(model.entropy_model, result.get("y_cdf_mode", "gaussian"))
                latent = model.decompress(
                    payload, result["z_shape"], result["y_shape"], qp=args.qp,
                    skip_index_cutoff=result["skip_index_cutoff"],
                    y_cdf_mode=result["y_cdf_mode"],
                )
                data = compress_image(model, image, qp=args.qp)[0]

                encode_total, _ = timed(lambda: compress_image(model, image, qp=args.qp), args.repeats)
                encode_nn, _ = timed(lambda: model.encode(padded, qp=args.qp), args.repeats)
                decode_total, _ = timed(lambda: decompress_image(model, data), args.repeats)
                decode_entropy, _ = timed(lambda: model.decompress(
                    payload, result["z_shape"], result["y_shape"], qp=args.qp,
                    skip_index_cutoff=result["skip_index_cutoff"],
                    y_cdf_mode=result["y_cdf_mode"]), args.repeats)
                decode_nn, _ = timed(lambda: model.decode(latent, qp=args.qp), args.repeats)

            mp = image.shape[-2] * image.shape[-1] / 1e6
            rows.append({
                "device": device, "threads": threads, "image": path.name, "qp": args.qp,
                "encode_total_ms": round(encode_total, 2),
                "encode_nn_ms": round(encode_nn, 2),
                "encode_entropy_ms": round(encode_total - encode_nn, 2),
                "decode_total_ms": round(decode_total, 2),
                "decode_entropy_ms": round(decode_entropy, 2),
                "decode_nn_ms": round(decode_nn, 2),
                "pixels_mp": round(mp, 4),
                "ms_per_mp_encode": round(encode_total / mp, 2),
                "ms_per_mp_decode": round(decode_total / mp, 2),
            })
            print(json.dumps({k: rows[-1][k] for k in
                              ("device", "threads", "encode_total_ms", "encode_nn_ms",
                               "decode_total_ms", "decode_entropy_ms", "decode_nn_ms")}))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"profile_{args.dataset}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(out / f'profile_{args.dataset}.csv')}))


if __name__ == "__main__":
    main()
