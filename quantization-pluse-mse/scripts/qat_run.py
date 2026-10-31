"""Quantization-aware fine-tuning for one mixed-precision assignment.

Loads the released bundle, applies the assignment with straight-through quantized convolutions,
fine-tunes on a pool that excludes every evaluation set, saves a loadable bundle, and reports the
quality before and after.

    python quantization-pluse-mse/scripts/qat_run.py --tier quantization-pluse-mse/configs/qat_tiers.json --tier-name tier1_safe_int8 --steps 800
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quantization-pluse-mse" / "src"))

import torch

from codec.checkpoint import load_model
from codec.images import image_paths, pad_image, read_image, to_uint8
from codec.training.losses import RDObjective
from pulsequant.qat import save_bundle, set_trainable, unwrap, wrap_assignment

SETTINGS = {"mse": 1.0, "lambda_min": 8.0, "lambda_max": 80.0, "mse_alpha": 0.0}


def log(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def load_pool(directory: Path, device: str, limit: int) -> list[torch.Tensor]:
    paths = image_paths(directory)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no images in {directory}; run tools/fetch_qat_pool.py first")
    return [read_image(p, device) for p in paths]


def measure(model, images, crop, qps) -> dict:
    """Per-QP (psnr, bpp) on fixed centre crops at the training resolution.

    Avoids `codec.runtime.compress_image`, which is decorated with `torch.inference_mode()`:
    inference tensors get cached on the entropy model and then break the next autograd step.
    """
    was_training = model.training
    model.eval()
    out = {}
    with torch.no_grad():
        for full in images:
            _, _, height, width = full.shape
            top = max(0, (height - crop) // 2)
            left = max(0, (width - crop) // 2)
            image = full[..., top:top + crop, left:left + crop].contiguous()
            padded = pad_image(image)
            for qp in qps:
                result = model.compress(padded, qp=qp, skip_index_cutoff=2,
                                        y_cdf_mode="part_specific")
                stream = bytes(result["stream"])
                latent = model.decompress(
                    stream, result["z_shape"], result["y_shape"], qp=qp,
                    skip_index_cutoff=result["skip_index_cutoff"],
                    y_cdf_mode=result["y_cdf_mode"])
                reconstruction = model.decode(latent, qp=qp)[..., :crop, :crop]
                pixels = to_uint8(reconstruction).float() / 255
                mse = (pixels - image).square().mean().item()
                psnr = -10 * math.log10(max(mse, 1e-12))
                bpp = 8 * len(stream) / (image.shape[-2] * image.shape[-1])
                surrogate = float(model(image, return_codec_res=True, qp=qp)[2]["sq_loss"].mean())
                previous = out.get(qp, (0.0, 0.0, 0.0, 0))
                out[qp] = (previous[0] + psnr, previous[1] + bpp,
                           previous[2] + surrogate, previous[3] + 1)
    if was_training:
        model.train()
    return {qp: (total / count, rate / count, sur / count)
            for qp, (total, rate, sur, count) in out.items()}


def slopes_from(points: dict) -> dict:
    ordered = sorted(points)
    if len(ordered) < 2:
        return {}
    xs = [math.log2(points[q][1]) for q in ordered]
    ys = [points[q][0] for q in ordered]
    slopes = {}
    for index, qp in enumerate(ordered):
        if index == 0:
            dx, dy = xs[1] - xs[0], ys[1] - ys[0]
        elif index == len(ordered) - 1:
            dx, dy = xs[-1] - xs[-2], ys[-1] - ys[-2]
        else:
            dx, dy = xs[index + 1] - xs[index - 1], ys[index + 1] - ys[index - 1]
        slopes[qp] = dy / dx if dx else 3.0
    return slopes


def penalty_score(baseline: dict, current: dict, slopes: dict) -> float:
    """Worst-QP rate-equivalent penalty against the pre-QAT model, the metric we report."""
    worst = 0.0
    for qp, values in current.items():
        if qp not in baseline:
            continue
        psnr, bpp = values[0], values[1]
        base_psnr, base_bpp = baseline[qp][0], baseline[qp][1]
        d_psnr = psnr - base_psnr
        d_bpp = 100.0 * (bpp / base_bpp - 1.0) if base_bpp else 0.0
        slope = slopes.get(qp) or 3.0
        if not slope:
            continue
        penalty = 100.0 * ((1.0 + d_bpp / 100.0) * 2.0 ** (-d_psnr / slope) - 1.0)
        worst = max(worst, penalty)
    return worst


def random_crop(image: torch.Tensor, size: int, generator: torch.Generator) -> torch.Tensor:
    _, _, height, width = image.shape
    top = int(torch.randint(0, max(1, height - size + 1), (1,), generator=generator).item())
    left = int(torch.randint(0, max(1, width - size + 1), (1,), generator=generator).item())
    return image[..., top:top + size, left:left + size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/pulse-mse")
    parser.add_argument("--tier", default="quantization-pluse-mse/configs/qat_tiers.json")
    parser.add_argument("--tier-name", required=True)
    parser.add_argument("--pool", default="datasets/qat_pool/div2k_valid")
    parser.add_argument("--output", default="quantization-pluse-mse/results/qat")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--scope", default="backbone",
                        choices=["assigned", "backbone", "encoder", "all"])
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--objective", choices=["augmented_lagrangian", "rate_preserving", "rd"],
                        default="augmented_lagrangian",
                        help="rate_preserving keeps bitrate at its pre-QAT value and pushes distortion down")
    parser.add_argument("--mu-alpha", type=float, default=1.0,
                        help="dual step size for the augmented-Lagrangian rate multiplier")
    parser.add_argument("--mu-max", type=float, default=20.0)
    parser.add_argument("--rate-weight", type=float, default=1.0,
                        help="weight on the rate-deviation penalty for --objective rate_preserving")
    parser.add_argument("--rd-weight", type=float, default=4.0,
                        help="multiplier on the distortion term; higher protects PSNR")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="validate and keep the best checkpoint every N steps (0 disables)")
    parser.add_argument("--eval-images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="tiny run for wiring checks")
    args = parser.parse_args()

    if args.smoke:
        args.steps = min(args.steps, 3)
        args.max_images = min(args.max_images or 2, 2)
        args.log_every = 1

    tiers = json.loads(Path(args.tier).read_text(encoding="utf-8"))
    if args.tier_name not in tiers:
        raise SystemExit(f"{args.tier_name} not in {args.tier}")
    tier = tiers[args.tier_name]
    assignment = tier["assignment"]

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    model = load_model(args.model, args.device)
    records = wrap_assignment(model, assignment)
    trainable, frozen = set_trainable(model, args.scope, set(assignment))
    model.train()
    criterion = RDObjective(SETTINGS, qp_num=8).to(args.device)
    pool = load_pool(Path(args.pool), args.device, args.max_images)

    log({"tier": args.tier_name, "events": "start", "steps": args.steps,
         "crop": args.crop, "scope": args.scope, "lr": args.lr,
         "wrapped_modules": len(records), "trainable_params": trainable, "frozen_params": frozen,
         "pool_images": len(pool), "assignment": assignment})

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim <= 1 else decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": 1e-4},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr)

    validation_images = pool[: max(1, args.eval_images)]
    validation_qps = tuple(range(8))
    baseline_measure = measure(model, validation_images, args.crop, validation_qps)
    slopes = slopes_from(baseline_measure)
    baseline_psnr = sum(v[0] for v in baseline_measure.values()) / len(baseline_measure)
    baseline_bpp = sum(v[1] for v in baseline_measure.values()) / len(baseline_measure)
    best_score, best_step = 0.0, 0
    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    log({"tier": args.tier_name, "events": "baseline_validation",
         "psnr": round(baseline_psnr, 4), "bpp": round(baseline_bpp, 4),
         "slopes": {str(k): round(v, 3) for k, v in slopes.items()}})

    mu = 0.0
    started = time.perf_counter()
    running = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        chunk = []
        for _ in range(args.accumulate):
            image = random.choice(pool)
            crop = random_crop(image, args.crop, generator).to(args.device)
            if args.batch > 1:
                crop = crop.repeat(args.batch, 1, 1, 1)
            qp = int(torch.randint(0, 8, (1,), generator=generator).item())
            output, _, rates = model(crop, return_codec_res=True, qp=qp)
            # Explicit RD objective so the distortion/rate balance is controllable. The released
            # RDObjective weights rate heavily, which makes fine-tuning reduce bitrate rather than
            # recover quality; rd_weight biases it back toward PSNR.
            target = crop * 2 - 1
            mse_all = (output - target).square().mean(dim=(1, 2, 3))
            bpp = rates["sq_loss"].mean()
            if args.objective == "augmented_lagrangian":
                # Minimise distortion subject to holding the rate at its pre-QAT value. The dual
                # multiplier mu is what enforces the constraint; a fixed penalty cannot, because it
                # either lets the model drift or makes it dump distortion to hit the target.
                # mu is updated on the fixed validation set from the surrogate rate, which is the
                # only rate the training loss can see (stream bpp is a different quantity, roughly
                # a third of the surrogate, so comparing the two makes mu diverge).
                ref_psnr = baseline_measure[qp][0]
                ref_mse = 10.0 ** (-ref_psnr / 10.0)
                ref_sur = baseline_measure[qp][2]
                rate_ratio = bpp / ref_sur - 1.0
                loss = (args.rd_weight * (mse_all.mean() / ref_mse)
                        + mu * rate_ratio)
            elif args.objective == "rate_preserving":
                # Keep bitrate at its pre-QAT level and push distortion down. Optimising the raw RD
                # objective instead slides *along* the curve: 5,900 steps bought 29% rate for 1.54 dB,
                # a 52% penalty. Normalising both terms makes the target "same rate, less error".
                ref_psnr, ref_bpp = baseline_measure[qp]
                ref_mse = 10.0 ** (-ref_psnr / 10.0)
                rate_term = args.rate_weight * ((bpp / ref_bpp) - 1.0) ** 2
                loss = args.rd_weight * (mse_all.mean() / ref_mse) + rate_term
            else:
                lam = criterion.lambdas[qp]
                loss = (args.rd_weight * lam * mse_all).mean() + bpp
            metrics = {"loss": loss.detach(), "bpp": rates["sq_loss"].mean().detach(),
                       "mse": mse_all.mean().detach()}
            (loss / args.accumulate).backward()
            chunk.append((float(metrics["loss"]), float(metrics["bpp"]), float(metrics["mse"])))
        if args.clip:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.clip)
        optimizer.step()
        running.extend(chunk)
        if args.eval_every and (step % args.eval_every == 0 or step == args.steps):
            current = measure(model, validation_images, args.crop, validation_qps)
            if args.objective == "augmented_lagrangian":
                errors = [current[q][2] / baseline_measure[q][2] - 1.0
                          for q in validation_qps if baseline_measure[q][2] > 0]
                mu = max(-args.mu_max, min(args.mu_max,
                                           mu + args.mu_alpha * (sum(errors) / max(1, len(errors)))))
            score = penalty_score(baseline_measure, current, slopes)
            psnr = sum(v[0] for v in current.values()) / len(current)
            bpp = sum(v[1] for v in current.values()) / len(current)
            improved = score < best_score
            if improved:
                best_score, best_step = score, step
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            log({"tier": args.tier_name, "events": "validated", "step": step,
                 "psnr": round(psnr, 4), "bpp": round(bpp, 4),
                 "d_psnr_vs_start": round(psnr - baseline_psnr, 4),
                 "penalty_vs_start_pct": round(score, 3),
                 "mu": round(mu, 4) if args.objective == "augmented_lagrangian" else None,
                 "best": improved})
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            recent = running[-args.log_every:]
            log({"tier": args.tier_name, "events": "step", "step": step,
                 "loss": round(sum(r[0] for r in recent) / len(recent), 5),
                 "bpp": round(sum(r[1] for r in recent) / len(recent), 4),
                 "mse": round(sum(r[2] for r in recent) / len(recent), 6),
                 "elapsed_s": round(time.perf_counter() - started, 1)})

    model.load_state_dict(best_state)
    unwrap(records)
    out = Path(args.output)
    bundle = out / "bundles" / args.tier_name
    save_bundle(model, Path(args.model), bundle)
    elapsed = time.perf_counter() - started
    summary = {
        "tier": args.tier_name,
        "events": "finish",
        "bundle": str(bundle),
        "steps": args.steps,
        "train_minutes": round(elapsed / 60, 2),
        "final_loss": round(sum(r[0] for r in running[-args.log_every:]) / max(1, len(running[-args.log_every:])), 5),
        "final_bpp": round(sum(r[1] for r in running[-args.log_every:]) / max(1, len(running[-args.log_every:])), 4),
        "baseline_val_psnr": round(baseline_psnr, 4),
        "best_penalty_vs_start_pct": round(best_score, 3),
        "best_step": best_step,
        "kept_best_checkpoint": best_state is not None,
        "assignment": assignment,
    }
    log(summary)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tier_name}_train.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                      encoding="utf-8")


if __name__ == "__main__":
    main()
