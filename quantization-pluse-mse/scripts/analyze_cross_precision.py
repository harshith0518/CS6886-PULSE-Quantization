"""Compare component sensitivity across all completed precision folders."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quantization-pluse-mse" / "src"))

from pulsequant import plots

FORMATS = ["int8", "int4", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1"]


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="quantization-pluse-mse")
    parser.add_argument("--output", default="quantization-pluse-mse/results/cross_precision")
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    available = [fmt for fmt in FORMATS if (root / fmt / "results" / "summary.csv").is_file()]
    long_rows = []
    for fmt in available:
        summary = read_csv(root / fmt / "results" / "summary.csv")
        configs = {row["component"]: row for row in read_csv(root / fmt / "results" / f"configs_{fmt}.csv")}
        groups = defaultdict(list)
        for row in summary:
            if row["control"] == "identity":
                continue
            groups[row["component"]].append(row)
        for component, rows in sorted(groups.items()):
            penalty = [float(row["mean_penalty_pct"]) for row in rows if row["mean_penalty_pct"] not in ("", "None")]
            d_psnr = [float(row["mean_d_psnr"]) for row in rows]
            d_bpp = [float(row["mean_d_bpp_pct"]) for row in rows]
            config = configs.get(component, {})
            long_rows.append({
                "format": fmt,
                "component": component,
                "datasets": len({row["dataset"] for row in rows}),
                "qps": len({row["qp"] for row in rows}),
                "mean_penalty_pct": statistics.mean(penalty) if penalty else None,
                "max_penalty_pct": max(penalty) if penalty else None,
                "mean_d_psnr": statistics.mean(d_psnr),
                "worst_d_psnr": min(d_psnr),
                "mean_d_bpp_pct": statistics.mean(d_bpp),
                "size_reduction_pct": config.get("size_reduction_pct"),
                "effective_bits_per_param": config.get("effective_bits_per_param"),
            })
    write_csv(output / "component_precision_summary.csv", long_rows)

    components = sorted({row["component"] for row in long_rows})
    penalty_wide = []
    psnr_wide = []
    for component in components:
        prow = {"component": component}
        srow = {"component": component}
        for fmt in available:
            match = next((row for row in long_rows if row["component"] == component and row["format"] == fmt), None)
            prow[fmt] = match["mean_penalty_pct"] if match else None
            srow[fmt] = match["mean_d_psnr"] if match else None
        penalty_wide.append(prow)
        psnr_wide.append(srow)
    write_csv(output / "component_precision_penalty.csv", penalty_wide)
    write_csv(output / "component_precision_d_psnr.csv", psnr_wide)

    if available:
        plots.heatmap(
            output / "component_precision_penalty_heatmap.png",
            "Mean rate-equivalent penalty by component and precision",
            components,
            available,
            [[row[fmt] for fmt in available] for row in penalty_wide],
        )
        plots.heatmap(
            output / "component_precision_d_psnr_heatmap.png",
            "Mean d_PSNR by component and precision",
            components,
            available,
            [[row[fmt] for fmt in available] for row in psnr_wide],
        )

    lines = [
        "# Cross-precision component summary",
        "",
        f"Formats: {', '.join(available)}",
        "",
        "| Component | " + " | ".join(available) + " |",
        "|---|" + "|".join("---:" for _ in available) + "|",
    ]
    for row in penalty_wide:
        lines.append("| " + row["component"] + " | " + " | ".join(
            f"{row[fmt]:.3f}" if row[fmt] is not None else "-" for fmt in available
        ) + " |")
    lines.extend(["", "## Files", "", "- `component_precision_summary.csv`",
                  "- `component_precision_penalty.csv`", "- `component_precision_d_psnr.csv`",
                  "- `component_precision_penalty_heatmap.png`",
                  "- `component_precision_d_psnr_heatmap.png`"])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"formats={available} components={len(components)}")


if __name__ == "__main__":
    main()
