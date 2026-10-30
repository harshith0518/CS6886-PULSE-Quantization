"""Measure the cost of one QAT training step on this laptop.

Training the released model from scratch is 500,000 steps at batch 8, crop 512 (see
`train_mse.py --check-config`). That is out of reach here. This measures what a *fine-tuning*
step costs, which is what a QAT trial would need: forward, rate+distortion loss, backward.

    python quantization-pluse-mse/scripts/bench_qat_step.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from codec.checkpoint import load_model
from codec.images import read_image
from codec.training.losses import RDObjective

SETTINGS = {"mse": 1.0, "lambda_min": 8.0, "lambda_max": 80.0, "mse_alpha": 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/pulse-mse")
    parser.add_argument("--image", default="datasets/kodak/kodim01.png")
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--qp", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="quantization-pluse-mse/results/metrics/qat_step.json")
    args = parser.parse_args()

    model = load_model(args.model, args.device).train()
    criterion = RDObjective(SETTINGS, qp_num=8).to(args.device)
    image = read_image(args.image, args.device)
    _, _, height, width = image.shape
    top = max(0, (height - args.crop) // 2)
    left = max(0, (width - args.crop) // 2)
    patch = image[..., top:top + args.crop, left:left + args.crop]

    results = []
    for batch in args.batch:
        x = patch.repeat(batch, 1, 1, 1)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        times, peak = [], 0.0
        try:
            torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
            for step in range(args.warmup + args.steps):
                if args.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                output, y_hat, codec_res = model(x, return_codec_res=True, qp=args.qp)
                loss, metrics = criterion(output, x, codec_res, args.qp, {})
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if args.device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if step >= args.warmup:
                    times.append(elapsed)
            if args.device == "cuda":
                peak = torch.cuda.max_memory_allocated() / 1024**2
            median = statistics.median(times)
            row = {
                "batch": batch, "crop": args.crop, "qp": args.qp,
                "step_s": round(median, 4),
                "peak_vram_mb": round(peak, 1),
                "bpp": round(float(metrics["bpp"]), 4),
                "mse": round(float(metrics["mse"]), 6),
                "loss": round(float(metrics["loss"]), 4),
                "status": "ok",
            }
        except RuntimeError as error:
            row = {"batch": batch, "crop": args.crop, "qp": args.qp, "status": "error",
                   "error": f"{type(error).__name__}: {str(error)[:160]}"}
            if args.device == "cuda":
                torch.cuda.empty_cache()
        results.append(row)
        print(json.dumps(row))

    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        best = min(ok, key=lambda r: r["step_s"])
        per_step = best["step_s"]
        summary = {
            "measured_step_s": per_step,
            "batch": best["batch"],
            "crop": best["crop"],
            "peak_vram_mb": best["peak_vram_mb"],
            "steps_500": round(per_step * 500 / 60, 2),
            "steps_1000": round(per_step * 1000 / 60, 2),
            "steps_2000": round(per_step * 2000 / 60, 2),
            "units": "minutes for the stated number of optimizer steps",
        }
        print(json.dumps({"projection": summary}, indent=2))
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"runs": results, "projection": summary}, indent=2) + "\n",
                       encoding="utf-8")
        print(json.dumps({"output": str(out)}))


if __name__ == "__main__":
    main()
