"""Test whether single-component penalties add up to the measured whole-model penalty.

The ILP objective and every budget estimate so far assume additivity: the cost of a mixed
assignment is the sum of its components' single-component costs. This script measures that
assumption instead of assuming it.

    python quantization-pluse-mse/scripts/analyze_additivity.py --dataset kodak
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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


def number(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_coefficients(root: Path, dataset: str) -> dict:
    coefficients = {}
    for surface in sorted(root.glob("*/results/summary.csv")):
        fmt = surface.parent.parent.name
        for row in read_csv(surface):
            if row.get("dataset") != dataset or row.get("control") != "none":
                continue
            key = (row["component"], fmt, int(row["qp"]))
            coefficients[key] = {
                "penalty": number(row.get("mean_penalty_pct")),
                "d_psnr": number(row.get("mean_d_psnr")),
                "d_bpp": number(row.get("mean_d_bpp_pct")),
            }
    return coefficients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="quantization-pluse-mse")
    parser.add_argument("--mixed", default="quantization-pluse-mse/results/mixed")
    parser.add_argument("--dataset", nargs="+", default=["kodak", "clic_professional"])
    args = parser.parse_args()

    root, mixed = Path(args.root), Path(args.mixed)
    all_rows = []
    for dataset in args.dataset:
        summary = read_csv(mixed / f"summary_{dataset}.csv")
        configs = {row["config"]: row for row in read_csv(mixed / f"configs_{dataset}.csv")}
        if not summary:
            print(f"skip {dataset}: no summary")
            continue
        coefficients = load_coefficients(root, dataset)

        measured = {(row["config"], int(row["qp"])): row for row in summary}
        for config, meta in configs.items():
            assignment = json.loads(meta["assignment"]) if meta.get("assignment") else {}
            for qp in range(8):
                row = measured.get((config, qp))
                if row is None:
                    continue
                predicted_penalty = 0.0
                predicted_psnr = 0.0
                predicted_bpp = 0.0
                missing = 0
                for component, fmt in assignment.items():
                    coefficient = coefficients.get((component, fmt, qp))
                    if coefficient is None or coefficient["penalty"] is None:
                        missing += 1
                        continue
                    predicted_penalty += coefficient["penalty"]
                    predicted_psnr += coefficient["d_psnr"] or 0.0
                    predicted_bpp += coefficient["d_bpp"] or 0.0
                actual = number(row.get("mean_penalty_pct"))
                ratio = None
                if actual is not None and predicted_penalty > 1e-9:
                    ratio = actual / predicted_penalty
                all_rows.append({
                    "dataset": dataset,
                    "config": config,
                    "qp": qp,
                    "components": len(assignment),
                    "missing_coefficients": missing,
                    "measured_penalty_pct": "" if actual is None else round(actual, 4),
                    "predicted_penalty_pct": round(predicted_penalty, 4),
                    "ratio_measured_over_predicted": "" if ratio is None else round(ratio, 4),
                    "measured_d_psnr": round(number(row.get("mean_d_psnr")) or 0.0, 4),
                    "predicted_d_psnr": round(predicted_psnr, 4),
                    "measured_d_bpp_pct": round(number(row.get("mean_d_bpp_pct")) or 0.0, 4),
                    "predicted_d_bpp_pct": round(predicted_bpp, 4),
                    "size_mb": meta.get("quant_mb"),
                    "saving_pct": meta.get("saving_pct"),
                })

    write_csv(mixed / "additivity.csv", all_rows)

    print(f"{'dataset':18} {'config':8} {'qp':>2} {'meas%':>9} {'pred%':>9} {'ratio':>7}")
    for row in all_rows:
        if row["config"] in {"fp32", "fp16"}:
            continue
        print(f"{row['dataset']:18} {row['config']:8} {row['qp']:>2} "
              f"{row['measured_penalty_pct']:>9} {row['predicted_penalty_pct']:>9} "
              f"{row['ratio_measured_over_predicted']:>7}")


if __name__ == "__main__":
    main()
