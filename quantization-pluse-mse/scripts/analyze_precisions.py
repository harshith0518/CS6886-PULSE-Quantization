"""Build the component x precision table, heatmaps, and recommendation report."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quantization-pluse-mse" / "src"))

from PIL import Image, ImageDraw, ImageFont
from pulsequant import plots

FORMATS = [
    "fp16", "bf16", "fp8_e4m3", "fp8_e5m2",
    "int8", "int4", "fp4_e2m1",
]
PRETTY = {
    "fp16": "FP16", "bf16": "BF16", "fp8_e4m3": "FP8-E4M3",
    "fp8_e5m2": "FP8-E5M2", "int8": "INT8", "int4": "INT4",
    "fp4_e2m1": "FP4",
}
AGGRESSION = {name: index for index, name in enumerate(FORMATS)}


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


def classify(value):
    if value is None:
        return "na"
    if value <= 1.0:
        return "green"
    if value <= 3.0:
        return "yellow"
    return "red"


def fmt(value):
    return "-" if value is None else f"{value:.3f}"


def font(size=14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def table_image(path: Path, title: str, components: list[str], formats: list[str], values: dict):
    cell_w, cell_h = 150, 30
    left, top = 110, 90
    width = left + cell_w * len(formats) + 30
    height = top + cell_h * len(components) + 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 24), title, fill=(20, 20, 20), font=font(20), anchor="mt")
    for col, fmt_name in enumerate(formats):
        x = left + col * cell_w + cell_w / 2
        draw.text((x, top - 20), PRETTY[fmt_name], fill=(20, 20, 20), font=font(13), anchor="mm")
    for row, component in enumerate(components):
        y = top + row * cell_h
        draw.text((left - 10, y + cell_h / 2), component, fill=(20, 20, 20), font=font(13), anchor="rm")
        for col, fmt_name in enumerate(formats):
            x = left + col * cell_w
            value = values[(component, fmt_name)]
            kind = classify(value)
            color = {
                "green": (210, 240, 210),
                "yellow": (255, 242, 190),
                "red": (250, 205, 205),
                "na": (238, 238, 238),
            }[kind]
            draw.rectangle((x, y, x + cell_w - 2, y + cell_h - 2), fill=color, outline="white")
            draw.text((x + cell_w / 2, y + cell_h / 2), fmt(value), fill=(20, 20, 20), font=font(12), anchor="mm")
    draw.text((20, height - 30), "green <= 1% | yellow 1-3% | red > 3% | worst-QP mean penalty",
              fill=(60, 60, 60), font=font(13), anchor="lm")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="quantization-pluse-mse")
    parser.add_argument("--output", default="quantization-pluse-mse/results/analysis")
    parser.add_argument("--exclude", nargs="*", default=[])
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    inventory = json.loads((root / "artifacts" / "component_inventory.json").read_text(encoding="utf-8"))
    components = sorted(inventory.get("main_sweep", []))
    available = [
        fmt for fmt in FORMATS
        if fmt not in args.exclude and (root / fmt / "results" / "summary.csv").is_file()
    ]

    long_rows = []
    values = {}
    for fmt_name in available:
        summary = read_csv(root / fmt_name / "results" / "summary.csv")
        configs = {row["component"]: row for row in read_csv(root / fmt_name / "results" / f"configs_{fmt_name}.csv")}
        groups = defaultdict(list)
        for row in summary:
            if row["control"] == "identity":
                continue
            groups[row["component"]].append(row)
        for component in components:
            rows = groups.get(component, [])
            if not rows:
                continue
            mean_penalty = [float(row["mean_penalty_pct"]) for row in rows if row["mean_penalty_pct"] not in ("", "None")]
            worst_qp = [float(row["mean_penalty_pct"]) for row in rows if row["mean_penalty_pct"] not in ("", "None")]
            mean_d_psnr = [float(row["mean_d_psnr"]) for row in rows]
            worst_d_psnr = [float(row["worst_d_psnr"]) for row in rows]
            mean_d_bpp = [float(row["mean_d_bpp_pct"]) for row in rows]
            config = configs.get(component, {})
            item = {
                "format": fmt_name,
                "component": component,
                "mean_penalty_pct": statistics.mean(mean_penalty) if mean_penalty else None,
                "worst_qp_penalty_pct": max(worst_qp) if worst_qp else None,
                "max_image_penalty_pct": max(float(row["max_penalty_pct"]) for row in rows),
                "mean_d_psnr": statistics.mean(mean_d_psnr),
                "worst_d_psnr": min(worst_d_psnr),
                "mean_d_bpp_pct": statistics.mean(mean_d_bpp),
                "size_reduction_pct": config.get("size_reduction_pct"),
                "effective_bits_per_param": config.get("effective_bits_per_param"),
                "class": classify(max(worst_qp) if worst_qp else None),
            }
            long_rows.append(item)
            values[(component, fmt_name)] = item["worst_qp_penalty_pct"]
    write_csv(output / "component_precision_metrics.csv", long_rows)

    penalty_wide = []
    psnr_wide = []
    size_wide = []
    for component in components:
        prow = {"component": component}
        srow = {"component": component}
        zrow = {"component": component}
        for fmt_name in available:
            item = next((row for row in long_rows if row["component"] == component and row["format"] == fmt_name), None)
            prow[fmt_name] = item["worst_qp_penalty_pct"] if item else None
            srow[fmt_name] = item["mean_d_psnr"] if item else None
            zrow[fmt_name] = item["size_reduction_pct"] if item else None
        penalty_wide.append(prow)
        psnr_wide.append(srow)
        size_wide.append(zrow)
    write_csv(output / "component_precision_penalty.csv", penalty_wide)
    write_csv(output / "component_precision_d_psnr.csv", psnr_wide)
    write_csv(output / "component_precision_size.csv", size_wide)

    table_image(
        output / "component_precision_table.png",
        "Worst-QP mean penalty by component and precision",
        components,
        available,
        values,
    )
    plots.heatmap(
        output / "component_precision_penalty_heatmap.png",
        "Worst-QP mean penalty by component and precision",
        components,
        [PRETTY[name] for name in available],
        [[values.get((component, fmt_name)) for fmt_name in available] for component in components],
    )
    plots.heatmap(
        output / "component_precision_d_psnr_heatmap.png",
        "Mean d_PSNR by component and precision",
        components,
        [PRETTY[name] for name in available],
        [[next((row["mean_d_psnr"] for row in long_rows if row["component"] == component and row["format"] == fmt_name), None)
          for fmt_name in available] for component in components],
    )

    recommendations = []
    for component in components:
        items = [row for row in long_rows if row["component"] == component]
        green = [row for row in items if row["class"] == "green"]
        safe = max(green, key=lambda row: AGGRESSION[row["format"]])["format"] if green else None
        int8 = next((row for row in items if row["format"] == "int8"), None)
        fp16 = next((row for row in items if row["format"] in {"fp16", "bf16"}), None)
        needs_qat = bool(int8 and int8["class"] in {"yellow", "red"})
        do_not = bool(fp16 and fp16["class"] != "green")
        recommendations.append({
            "component": component,
            "safe_to": PRETTY.get(safe, "none"),
            "needs_qat": "yes" if needs_qat else "no",
            "do_not_quantize": "yes" if do_not else "no",
            "int8_class": int8["class"] if int8 else "missing",
            "int8_penalty_pct": int8["worst_qp_penalty_pct"] if int8 else None,
            "fp16_class": fp16["class"] if fp16 else "missing",
            "params": inventory.get("components", {}).get(component, {}).get("params"),
            "kmac_per_px": inventory.get("components", {}).get(component, {}).get("kmac_per_px"),
        })
    write_csv(output / "recommendations.csv", recommendations)

    lines = [
        "# Component x precision analysis",
        "",
        f"Formats included: {', '.join(PRETTY[name] for name in available)}",
        "",
        "| Component | " + " | ".join(PRETTY[name] for name in available) + " | Safe to | QAT | Do not touch |",
        "|---|" + "|".join("---:" for _ in available) + "|---:|---:|---:|",
    ]
    for row in recommendations:
        values_text = " | ".join(
            fmt(values.get((row["component"], fmt_name))) for fmt_name in available
        )
        lines.append(
            f"| {row['component']} | {values_text} | {row['safe_to']} | "
            f"{row['needs_qat']} | {row['do_not_quantize']} |"
        )
    lines.extend([
        "",
        "## Recommendation logic",
        "",
        "- `safe_to` = most aggressive format whose worst-QP mean penalty is <= 1%.",
        "- `needs_qat` = INT8 is yellow/red; QAT may recover it.",
        "- `do_not_quantize` = even FP16/BF16 is not green; keep FP32.",
        "- Values are worst-QP mean penalty percentages.",
        "",
        "## Files",
        "",
        "- `component_precision_metrics.csv`",
        "- `component_precision_penalty.csv`",
        "- `component_precision_d_psnr.csv`",
        "- `component_precision_size.csv`",
        "- `recommendations.csv`",
        "- `component_precision_table.png`",
        "- `component_precision_penalty_heatmap.png`",
        "- `component_precision_d_psnr_heatmap.png`",
    ])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"formats": available, "components": len(components)}, indent=2))


if __name__ == "__main__":
    main()
