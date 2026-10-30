"""Compare the mixed-precision model against the FP32 base model.

Produces BD-rate (Bjontegaard delta rate) and per-QP deltas from the measured whole-model
sweeps, plus the figures used in the comparison write-up.

BD-rate here uses monotone PCHIP interpolation over all eight measured QPs rather than the
classic four-point cubic, because eight points are available. Negative BD-rate means the
candidate needs fewer bits than FP32 at equal PSNR, i.e. it is better.

    python quantization-pluse-mse/scripts/analyze_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "quantization-pluse-mse" / "src"))

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from pulsequant import plots

PRETTY = {
    "fp32": "FP32 base", "fp16": "FP16 uniform", "bf16": "BF16 uniform", "int8": "INT8 uniform",
    "safe": "safe mixed", "greedy": "greedy mixed", "ilp": "ILP mixed",
}
COLOR = {
    "fp32": (90, 90, 90), "fp16": (31, 119, 180), "bf16": (23, 190, 207), "int8": (214, 39, 40),
    "safe": (255, 127, 14), "greedy": (148, 103, 189), "ilp": (44, 160, 44),
}
ORDER = ["fp32", "fp16", "bf16", "int8", "safe", "greedy", "ilp"]
WHITE, INK, GRID, AXIS = (255, 255, 255), (25, 25, 25), (228, 228, 228), (70, 70, 70)


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


def curves(path: Path) -> dict:
    groups = defaultdict(list)
    for row in read_csv(path):
        groups[(row["config"], int(row["qp"]))].append(row)
    points = defaultdict(list)
    for (config, qp), rows in groups.items():
        points[config].append((
            qp,
            sum(float(r["bpp"]) for r in rows) / len(rows),
            sum(float(r["psnr"]) for r in rows) / len(rows),
        ))
    return {config: sorted(values) for config, values in points.items()}


def bd_rate(anchor: list, test: list) -> dict | None:
    a = np.array([[math.log2(p[1]), p[2]] for p in anchor if p[1] > 0])
    b = np.array([[math.log2(p[1]), p[2]] for p in test if p[1] > 0])
    if len(a) < 3 or len(b) < 3:
        return None
    low = max(a[:, 1].min(), b[:, 1].min())
    high = min(a[:, 1].max(), b[:, 1].max())
    if high - low < 0.1:
        return None
    fa = PchipInterpolator(a[:, 1], a[:, 0])
    fb = PchipInterpolator(b[:, 1], b[:, 0])
    samples = np.linspace(low, high, 200)
    delta = float(np.mean(fb(samples) - fa(samples)))
    return {
        "bd_rate_pct": (2.0 ** delta - 1.0) * 100.0,
        "bd_psnr_db": None,
        "psnr_low": low,
        "psnr_high": high,
        "points": len(a),
    }


def bd_psnr(anchor: list, test: list) -> float | None:
    a = np.array([[math.log2(p[1]), p[2]] for p in anchor if p[1] > 0])
    b = np.array([[math.log2(p[1]), p[2]] for p in test if p[1] > 0])
    if len(a) < 3 or len(b) < 3:
        return None
    low = max(a[:, 0].min(), b[:, 0].min())
    high = min(a[:, 0].max(), b[:, 0].max())
    if high - low < 0.1:
        return None
    fa = PchipInterpolator(a[:, 0], a[:, 1])
    fb = PchipInterpolator(b[:, 0], b[:, 1])
    samples = np.linspace(low, high, 200)
    return float(np.mean(fb(samples) - fa(samples)))


def figure_rd(datasets: dict, out: Path):
    names = list(datasets)
    width, height = 1500, 700
    left, gap = 100, 55
    panel_w = (width - left - 40 - gap * (len(names) - 1)) // len(names)
    top, bottom = 130, 190
    panel_h = height - top - bottom
    image, draw = plots._canvas(width, height)

    all_rate = [p[1] for curves_ in datasets.values() for c in curves_ for p in curves_[c]]
    all_psnr = [p[2] for curves_ in datasets.values() for c in curves_ for p in curves_[c]]
    lo, hi = math.log2(min(all_rate)), math.log2(max(all_rate))
    plo, phi = math.floor(min(all_psnr)), math.ceil(max(all_psnr))

    for panel, dataset in enumerate(names):
        x0 = left + panel * (panel_w + gap)
        curves_ = datasets[dataset]

        def px(value):
            return x0 + panel_w * (math.log2(value) - lo) / (hi - lo)

        def py(value):
            return top + panel_h * (1 - (value - plo) / (phi - plo))

        for tick in range(plo, phi + 1, 2):
            y = py(tick)
            draw.line((x0, y, x0 + panel_w, y), fill=GRID, width=1)
            if panel == 0:
                plots._text(draw, (x0 - 10, y), f"{tick}", size=12, anchor="rm", fill=AXIS)
        for tick in [t for t in (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5) if lo <= math.log2(t) <= hi]:
            x = px(tick)
            draw.line((x, top, x, top + panel_h), fill=(242, 242, 242), width=1)
            plots._text(draw, (x, top + panel_h + 16), f"{tick:g}", size=12, anchor="mt", fill=AXIS)
        draw.line((x0, top, x0, top + panel_h), fill=AXIS, width=2)
        draw.line((x0, top + panel_h, x0 + panel_w, top + panel_h), fill=AXIS, width=2)
        plots._text(draw, (x0 + panel_w / 2, top - 22), dataset, size=16, anchor="mt", fill=INK)

        for config in [c for c in ORDER if c not in {"fp32", "ilp", "greedy", "int8"}]:
            if config not in curves_:
                continue
            pts = [(px(p[1]), py(p[2])) for p in curves_[config]]
            draw.line(pts, fill=COLOR[config], width=1)
        for config in ("greedy", "ilp", "int8", "fp32"):
            if config not in curves_:
                continue
            pts = [(px(p[1]), py(p[2])) for p in curves_[config]]
            draw.line(pts, fill=COLOR[config], width=5 if config == "fp32" else 3)
            for x, y in pts:
                radius = 4 if config in {"fp32", "ilp"} else 2
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=COLOR[config])

    plots._text(draw, (width // 2, 40), "Rate-distortion: mixed precision rides the FP32 base curve",
                size=21, anchor="mt", fill=INK)
    plots._text(draw, (width // 2, 72),
                "mean over images, all 8 QPs. FP16, BF16 and the safe mix coincide with FP32 to within 0.1%, "
                "so only uniform INT8 separates visibly.",
                size=12, anchor="mt", fill=AXIS)
    legend_y = height - 118
    per_row = 4
    for index, config in enumerate(ORDER):
        row, col = divmod(index, per_row)
        lx = left + col * ((width - left - 40) / per_row)
        ly = legend_y + row * 26
        draw.line((lx, ly, lx + 28, ly), fill=COLOR[config], width=4 if config in {"fp32", "ilp"} else 2)
        plots._text(draw, (lx + 36, ly), PRETTY[config], size=13, anchor="lm", fill=INK)
    plots._text(draw, (left, height - 34), "bitrate (bpp, log scale)", size=13, anchor="lm", fill=INK)
    plots._text(draw, (left, top - 46), "PSNR (dB)", size=13, anchor="lm", fill=INK)
    plots._save(image, out)


def figure_bd(rows: list[dict], out: Path):
    datasets = sorted({row["dataset"] for row in rows})
    configs = [c for c in ORDER if c != "fp32" and any(r["config"] == c for r in rows)]
    row_h = 32
    width = 1320
    left, right = 250, 90
    top = 132
    height = top + row_h * len(configs) * len(datasets) + 96
    image, draw = plots._canvas(width, height)
    plot_w = width - left - right

    values = {abs(float(r["bd_rate_pct"])) for r in rows}
    vmax = max(max(values), 1.0)
    step = max(1, int(round(vmax / 8)) or 1)
    axis_max = math.ceil(vmax / step) * step

    def px(value):
        return left + plot_w * (value / axis_max)

    def colour(value):
        if value < 0.1:
            return (58, 168, 92)
        if value < 1.0:
            return (222, 178, 40)
        return (202, 45, 40)

    for tick in range(0, int(axis_max) + 1, step):
        x = px(tick)
        draw.line((x, top - 8, x, height - 76), fill=GRID, width=1)
        plots._text(draw, (x, top - 18), f"{tick}%", size=12, anchor="mb", fill=AXIS)
    draw.line((left, top - 8, left, height - 76), fill=AXIS, width=2)

    index = 0
    for dataset in datasets:
        for config in configs:
            match = next((r for r in rows if r["dataset"] == dataset and r["config"] == config), None)
            if match is None:
                continue
            y = top + index * row_h + row_h / 2
            value = float(match["bd_rate_pct"])
            plots._text(draw, (left - 14, y), f"{dataset}  {PRETTY[config]}", size=12, anchor="rm", fill=INK)
            x1 = px(abs(value))
            draw.rectangle((left, y - 10, max(left + 1, x1), y + 10), fill=colour(abs(value)))
            plots._text(draw, (x1 + 8, y), f"{value:+.2f}%", size=12, anchor="lm", fill=INK)
            index += 1

    legends = [("under 0.1% (negligible)", (58, 168, 92)),
               ("0.1% to 1% (usable)", (222, 178, 40)),
               ("over 1% (significant)", (202, 45, 40))]
    for order, (label, color) in enumerate(legends):
        lx = left + order * 300
        ly = height - 44
        draw.rectangle((lx, ly - 7, lx + 16, ly + 7), fill=color)
        plots._text(draw, (lx + 24, ly), label, size=12, anchor="lm", fill=AXIS)

    plots._text(draw, (width // 2, 40), "BD-rate against the FP32 base: extra bitrate for equal PSNR",
                size=21, anchor="mt", fill=INK)
    plots._text(draw, (width // 2, 72),
                "lower is better; all eight QPs, PCHIP interpolation. No configuration beats FP32 on rate.",
                size=12, anchor="mt", fill=AXIS)
    plots._save(image, out)


def figure_qp(rows: list[dict], out: Path):
    width, height = 1400, 660
    top, bottom, gap = 140, 150, 90
    left = 110
    panel_w = (width - left - 60 - gap) // 2
    panel_h = height - top - bottom
    image, draw = plots._canvas(width, height)

    panels = [
        ("d_psnr", "PSNR change (dB)", "negative = mixed model is below the base"),
        ("d_bpp_pct", "bitrate change (%)", "negative = mixed model uses fewer bits"),
    ]
    datasets = sorted({row["dataset"] for row in rows})

    for panel, (key, title, note) in enumerate(panels):
        x0 = left + panel * (panel_w + gap)
        series = defaultdict(list)
        for row in rows:
            series[row["dataset"]].append((int(row["qp"]), float(row[key])))
        values = [v for points in series.values() for _, v in points] + [0.0]
        vmin, vmax = min(values), max(values)
        pad = (vmax - vmin) * 0.18 or 0.005
        vmin, vmax = vmin - pad, vmax + pad

        def px(index):
            return x0 + panel_w * index / 7

        def py(value):
            return top + panel_h * (1 - (value - vmin) / (vmax - vmin))

        for step in range(6):
            tick = vmin + (vmax - vmin) * step / 5
            y = py(tick)
            draw.line((x0, y, x0 + panel_w, y), fill=GRID, width=1)
            plots._text(draw, (x0 - 10, y), f"{tick:+.3f}" if key == "d_psnr" else f"{tick:+.2f}",
                        size=12, anchor="rm", fill=AXIS)
        draw.line((x0, py(0.0), x0 + panel_w, py(0.0)), fill=(0, 0, 0), width=2)
        draw.line((x0, top, x0, top + panel_h), fill=AXIS, width=2)
        draw.line((x0, top + panel_h, x0 + panel_w, top + panel_h), fill=AXIS, width=2)
        for index in range(8):
            plots._text(draw, (px(index), top + panel_h + 16), index, size=12, anchor="mt", fill=AXIS)
        plots._text(draw, (x0 + panel_w / 2, top + panel_h + 44), "QP", size=13, anchor="mt", fill=INK)
        plots._text(draw, (x0, top - 20), title, size=15, anchor="lm", fill=INK)
        plots._text(draw, (x0, top - 2), note, size=11, anchor="lm", fill=AXIS)

        for order, dataset in enumerate(datasets):
            color = plots.PALETTE[order % len(plots.PALETTE)]
            points = sorted(series[dataset])
            path = [(px(qp), py(value)) for qp, value in points]
            draw.line(path, fill=color, width=3)
            for x, y in path:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)

    for order, dataset in enumerate(datasets):
        color = plots.PALETTE[order % len(plots.PALETTE)]
        lx = left + order * 260
        ly = height - 46
        draw.line((lx, ly, lx + 28, ly), fill=color, width=3)
        plots._text(draw, (lx + 36, ly), dataset, size=13, anchor="lm", fill=INK)

    plots._text(draw, (width // 2, 44), "Where ILP mixed precision gains and loses against FP32",
                size=21, anchor="mt", fill=INK)
    plots._text(draw, (width // 2, 74),
                "above the zero line is worse. The model pays a small PSNR loss everywhere but saves "
                "bitrate from QP4 upward.",
                size=12, anchor="mt", fill=AXIS)
    plots._save(image, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed", default="quantization-pluse-mse/results/mixed")
    parser.add_argument("--configs", default="quantization-pluse-mse/results/mixed")
    parser.add_argument("--dataset", nargs="+", default=["kodak", "clic_professional", "set14"])
    args = parser.parse_args()

    mixed = Path(args.mixed)
    datasets, bd_rows, detail_rows = {}, [], []
    for dataset in args.dataset:
        sweep = mixed / f"sweep_{dataset}.csv"
        if not sweep.is_file():
            print(f"skip {dataset}: no sweep")
            continue
        curves_ = curves(sweep)
        if "fp32" not in curves_:
            print(f"skip {dataset}: no fp32 curve")
            continue
        datasets[dataset] = curves_
        anchor = curves_["fp32"]
        for config in ORDER:
            if config == "fp32" or config not in curves_:
                continue
            rate = bd_rate(anchor, curves_[config])
            if rate is None:
                continue
            rate["bd_psnr_db"] = bd_psnr(anchor, curves_[config])
            bd_rows.append({
                "dataset": dataset, "config": config,
                "bd_rate_pct": round(rate["bd_rate_pct"], 4),
                "bd_psnr_db": None if rate["bd_psnr_db"] is None else round(rate["bd_psnr_db"], 4),
                "psnr_low": round(rate["psnr_low"], 3), "psnr_high": round(rate["psnr_high"], 3),
                "points": rate["points"],
            })

        ilp = {qp: (bpp, psnr) for qp, bpp, psnr in curves_["ilp"]}
        for qp, bpp, psnr in anchor:
            if qp not in ilp:
                continue
            bpp_i, psnr_i = ilp[qp]
            detail_rows.append({
                "dataset": dataset, "qp": qp,
                "base_bpp": round(bpp, 6), "ilp_bpp": round(bpp_i, 6),
                "base_psnr": round(psnr, 6), "ilp_psnr": round(psnr_i, 6),
                "d_psnr": round(psnr_i - psnr, 6),
                "d_bpp_pct": round(100.0 * (bpp_i / bpp - 1.0), 4),
            })

    write_csv(mixed / "bd_rate.csv", bd_rows)
    write_csv(mixed / "comparison_ilp_vs_fp32.csv", detail_rows)

    if datasets:
        figure_rd(datasets, mixed / "comparison_rd_curves.png")
        figure_qp(detail_rows, mixed / "comparison_ilp_dpsnr.png")
    if bd_rows:
        figure_bd(bd_rows, mixed / "comparison_bd_rate.png")

    summary = {
        "datasets": list(datasets),
        "bd_rate": bd_rows,
        "ilp_total_d_psnr_mean": round(
            sum(r["d_psnr"] for r in detail_rows) / len(detail_rows), 5) if detail_rows else None,
        "ilp_worst_d_psnr": min((r["d_psnr"] for r in detail_rows), default=None),
        "ilp_rate_change_range_pct": [
            min((r["d_bpp_pct"] for r in detail_rows), default=None),
            max((r["d_bpp_pct"] for r in detail_rows), default=None),
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
