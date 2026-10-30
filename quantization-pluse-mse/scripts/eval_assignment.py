"""Measure a mixed-precision assignment, or a uniform format, against the FP32 reference.

This is the whole-model experiment that the single-component sweep could not express: one
model carrying a different precision per component, evaluated on every image and QP.

    python quantization-pluse-mse/scripts/eval_assignment.py --dataset kodak \
        --configs fp32 fp16 bf16 int8 safe greedy

`fp32` must reproduce the reference stream and reconstruction hashes exactly; that is the
correctness gate for everything else in the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quantization-pluse-mse" / "src"))

import torch

from codec.checkpoint import load_model
from pulsequant.core import COMPONENTS, apply_assignment, assert_handle, component_size, weight_stats
from pulsequant.eval import (
    build_manifest,
    configure_eval,
    delta_metrics,
    evaluate_image,
    image_paths,
    read_image,
    sha256_json,
    write_json,
)

UNIFORM = ["fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "int8", "int4", "fp4_e2m1"]
MAIN_SWEEP = [name for name in COMPONENTS if name not in {"H4", "H5", "D9"}]


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_reference(reference_dir: Path) -> tuple[dict, dict]:
    rows = read_csv(reference_dir / "pass1.csv")
    if not rows:
        raise SystemExit(f"missing reference: {reference_dir / 'pass1.csv'}")
    reference = {(row["image"], int(row["qp"])): row for row in rows}
    slopes = {}
    curve_path = reference_dir / "curve.json"
    if curve_path.is_file():
        payload = json.loads(curve_path.read_text(encoding="utf-8-sig"))
        slopes = {int(k): float(v) for k, v in payload.get("slopes", {}).items()}
    return reference, slopes


def load_sizes(root: Path) -> tuple[dict, dict]:
    sizes, fp32 = {}, {}
    for fmt in UNIFORM:
        path = root / fmt / "results" / f"configs_{fmt}.csv"
        for row in read_csv(path):
            component = row["component"]
            if row.get("quant_bytes"):
                sizes[(component, fmt)] = float(row["quant_bytes"])
            if row.get("fp32_bytes"):
                fp32[component] = float(row["fp32_bytes"])
    return sizes, fp32


def resolve_config(name: str, decision_dir: Path, ilp_dir: Path) -> tuple[str, dict, str]:
    if name == "fp32":
        return "fp32", {}, "fp32"
    if name in UNIFORM:
        return name, {component: name for component in MAIN_SWEEP}, name
    if name in {"safe", "greedy"}:
        path = decision_dir / f"decision_assignment_{name}.json"
        if not path.is_file():
            raise SystemExit(f"missing {path}; run scripts/visualize_decision.py first")
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return name, payload["assignment"], payload.get("note", name)
    if name == "ilp":
        path = ilp_dir / "assignment_ilp.json"
        if not path.is_file():
            raise SystemExit(f"missing {path}; run scripts/solve_ilp.py first")
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return "ilp", payload["assignment"], payload.get("note", "ilp")
    path = Path(name)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        assignment = payload.get("assignment", payload)
        if not isinstance(assignment, dict) or any(isinstance(v, dict) for v in assignment.values()):
            raise SystemExit(
                f"{path} does not look like a single assignment; pass a file whose 'assignment' "
                "maps component names to precision strings, or extract one tier first")
        return path.stem, assignment, str(path)
    raise SystemExit(f"unknown config {name!r}")


def analytic_size(assignment: dict, sizes: dict, fp32: dict, model=None) -> dict:
    total_fp32 = sum(fp32.get(component, 0.0) for component in MAIN_SWEEP)
    total = 0.0
    for component in MAIN_SWEEP:
        fmt = assignment.get(component)
        if fmt is None:
            total += fp32.get(component, 0.0)
        else:
            value = sizes.get((component, fmt))
            if value is None and model is not None:
                value = float(component_size(model, component, fmt)["quant_bytes"])
                sizes[(component, fmt)] = value
            if value is None:
                raise SystemExit(f"no measured size for {component} {fmt}")
            total += value
    return {
        "fp32_mb": round(total_fp32 / 1e6, 3),
        "quant_mb": round(total / 1e6, 3),
        "saving_pct": round(100.0 * (1 - total / total_fp32), 2) if total_fp32 else 0.0,
        "effective_bits_per_param": round(8.0 * total / total_fp32, 3) if total_fp32 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/pulse-mse")
    parser.add_argument("--root", default="quantization-pluse-mse")
    parser.add_argument("--dataset", default="kodak")
    parser.add_argument("--configs", nargs="+", default=["fp32", "fp16", "bf16", "int8", "safe"])
    parser.add_argument("--qps", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--reference-dir", default="quantization-pluse-mse/results/_ledger/reference")
    parser.add_argument("--output", default="quantization-pluse-mse/results/mixed")
    parser.add_argument("--decision-dir", default="quantization-pluse-mse/results/decision")
    parser.add_argument("--ilp-dir", default="quantization-pluse-mse/results/ilp")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reference, slopes = load_reference(Path(args.reference_dir) / args.dataset)
    sizes, fp32 = load_sizes(root)

    paths = image_paths(Path(f"datasets/{args.dataset}"))
    if args.max_images:
        paths = paths[: args.max_images]
    print(json.dumps({"dataset": args.dataset, "images": len(paths), "qps": args.qps, "configs": args.configs}))

    model = load_model(args.model, args.device)
    flags = configure_eval(args.device, args.threads, tf32=False)

    sweep_rows, summary_rows, config_rows = [], [], []
    for name in args.configs:
        config_name, assignment, note = resolve_config(name, Path(args.decision_dir), Path(args.ilp_dir))
        config_id = sha256_json({"mode": "w_only", "assignment": assignment, "control": "none"})[:12]
        size = analytic_size(assignment, sizes, fp32, model)
        handle = None
        try:
            if assignment:
                handle = apply_assignment(model, assignment, control="none")
                if not handle.weights_changed:
                    raise SystemExit(f"assignment {config_name!r} did not change any weight")
            for path in paths:
                image = read_image(path, args.device)
                for qp in args.qps:
                    key = (path.name, qp)
                    ref = reference.get(key)
                    if ref is None:
                        raise KeyError(f"missing reference for {key}")
                    metrics = evaluate_image(model, image, qp, legacy=args.legacy)
                    deltas = delta_metrics(
                        metrics["psnr"], metrics["bpp"],
                        float(ref["psnr"]), float(ref["bpp"]),
                        slopes.get(qp),
                    )
                    sweep_rows.append({
                        "config": config_name, "config_id": config_id,
                        "dataset": args.dataset, "image": path.name, "qp": qp,
                        "bpp": round(metrics["bpp"], 6), "psnr": round(metrics["psnr"], 6),
                        "bytes": metrics["bytes"], "pixels": metrics["pixels"],
                        "d_psnr": round(deltas["d_psnr"], 6),
                        "d_bpp_pct": round(deltas["d_bpp_pct"], 6),
                        "penalty_pct": "" if deltas["penalty_pct"] is None else round(deltas["penalty_pct"], 6),
                        "stream_match": metrics["stream_hash"] == ref["stream_hash"],
                        "recon_match": metrics["recon_hash"] == ref.get("recon_hash", ""),
                        "status": "ok",
                    })
        finally:
            if handle is not None:
                handle.restore()
                assert_handle(handle)

        group = defaultdict(list)
        for row in sweep_rows:
            if row["config"] == config_name and row["dataset"] == args.dataset:
                group[row["qp"]].append(row)
        qp_penalties = []
        for qp, rows in sorted(group.items()):
            penalties = [row["penalty_pct"] for row in rows if row["penalty_pct"] != ""]
            d_psnr = [row["d_psnr"] for row in rows]
            d_bpp = [row["d_bpp_pct"] for row in rows]
            worst = min(rows, key=lambda row: row["d_psnr"])
            mean_penalty = sum(penalties) / len(penalties) if penalties else None
            if mean_penalty is not None:
                qp_penalties.append(mean_penalty)
            summary_rows.append({
                "config": config_name, "config_id": config_id, "dataset": args.dataset, "qp": qp,
                "rows": len(rows),
                "mean_bpp": round(sum(row["bpp"] for row in rows) / len(rows), 6),
                "mean_psnr": round(sum(row["psnr"] for row in rows) / len(rows), 6),
                "mean_d_psnr": round(sum(d_psnr) / len(d_psnr), 6),
                "worst_d_psnr": round(min(d_psnr), 6),
                "mean_d_bpp_pct": round(sum(d_bpp) / len(d_bpp), 6),
                "mean_penalty_pct": "" if mean_penalty is None else round(mean_penalty, 6),
                "max_penalty_pct": "" if not penalties else round(max(penalties), 6),
                "worst_image": worst["image"],
                "stream_match_rate": round(sum(row["stream_match"] for row in rows) / len(rows), 4),
                "recon_match_rate": round(sum(row["recon_match"] for row in rows) / len(rows), 4),
            })
        config_rows.append({
            "config": config_name, "config_id": config_id, "note": note,
            "components_quantized": len(assignment),
            **size,
            "worst_qp_mean_penalty_pct": "" if not qp_penalties else round(max(qp_penalties), 3),
            "mean_penalty_pct": "" if not qp_penalties else round(sum(qp_penalties) / len(qp_penalties), 3),
            "assignment": json.dumps(dict(sorted(assignment.items())), separators=(",", ":")),
        })
        print(json.dumps({k: v for k, v in config_rows[-1].items() if k != "assignment"}))

    write_csv(output / f"sweep_{args.dataset}.csv", sweep_rows)
    write_csv(output / f"summary_{args.dataset}.csv", summary_rows)
    write_csv(output / f"configs_{args.dataset}.csv", config_rows)

    gate = [row for row in summary_rows if row["config"] == "fp32"]
    if gate:
        clean = all(
            float(row["stream_match_rate"]) == 1.0 and float(row["recon_match_rate"]) == 1.0
            and abs(float(row["mean_d_psnr"])) < 1e-9
            for row in gate
        )
        write_json(output / f"gate_{args.dataset}.json", {
            "fp32_reproduces_reference": clean,
            "qps": len(gate),
            "note": "fp32 must be bit-identical to the reference stream and reconstruction",
        })
        if not clean:
            raise SystemExit("FP32 gate failed: the harness does not reproduce the reference")

    write_json(output / f"manifest_{args.dataset}.json", build_manifest(
        model_path=args.model,
        config={"dataset": args.dataset, "configs": args.configs, "qps": args.qps, "mode": "w_only"},
        command=" ".join(sys.argv),
    ) | {"eval_flags": flags})
    print(json.dumps({"output": str(output), "rows": len(sweep_rows), "configs": len(config_rows)}))


if __name__ == "__main__":
    main()
