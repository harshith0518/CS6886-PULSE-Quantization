"""Decision visualizations: which component to quantize, to which level, at which QP.

Reads the tracked per-precision surfaces only (`{fmt}/results/summary.csv`,
`{fmt}/results/configs_<fmt>.csv`, `artifacts/component_inventory.json`), so every
figure is reproducible from a clean checkout without the raw ledger.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "quantization-pluse-mse" / "src"))

from pulsequant import plots

FORMATS = ["fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "int8", "int4", "fp4_e2m1"]
REAL = {"fp16", "bf16", "int8"}
PRETTY = {
    "fp32": "FP32", "fp16": "FP16", "bf16": "BF16", "fp8_e4m3": "FP8-E4M3",
    "fp8_e5m2": "FP8-E5M2", "int8": "INT8", "int4": "INT4", "fp4_e2m1": "FP4-E2M1",
}
FAMILY = {"E": "encoder", "H": "entropy", "D": "decoder"}
FAMILY_COLOR = {"encoder": (31, 119, 180), "entropy": (214, 39, 40), "decoder": (44, 160, 44)}
GREEN, YELLOW = 1.0, 3.0
GREEN_C, YELLOW_C = (58, 168, 92), (222, 178, 40)

WHITE, INK, GRID, AXIS, GREY = (255, 255, 255), (25, 25, 25), (228, 228, 228), (70, 70, 70), (170, 170, 170)


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


def verdict(penalty):
    if penalty is None:
        return "none"
    if penalty <= GREEN:
        return "green"
    if penalty <= YELLOW:
        return "yellow"
    return "red"


def ramp(value):
    if value is None:
        return (232, 232, 232)
    t = (math.log10(max(value, 1e-6)) + 1.0) / 4.0
    t = min(1.0, max(0.0, t))
    if t < 0.5:
        u = t / 0.5
        return (int(58 + 197 * u), int(168 + 67 * u), int(92 - 42 * u))
    u = (t - 0.5) / 0.5
    return (255, int(235 - 190 * u), int(50 - 40 * u))


def ink_on(color):
    return WHITE if (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]) < 150 else INK


def fmt_value(value, digits=2):
    if value is None:
        return "-"
    if abs(value) < 0.005:
        return "0"
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.{digits}g}"


def size_text(megabytes):
    if megabytes < 0.1:
        return f"{megabytes:.3f}"
    return f"{megabytes:.2f}"


def log10_of(value, floor=1e-6):
    return math.log10(max(value, floor))


def axis_x(value, lo, hi, x0, width):
    t = (log10_of(value) - lo) / (hi - lo)
    return x0 + width * min(1.0, max(0.0, t))


def axis_y(value, lo, hi, y0, height):
    t = (log10_of(value) - lo) / (hi - lo)
    return y0 + height * (1 - min(1.0, max(0.0, t)))


def axis_title(draw, x, y, text):
    plots._text(draw, (x, y), text, size=13, anchor="lm", fill=INK)


def log_ticks(lo, hi, candidates):
    return [t for t in candidates if lo <= math.log10(t) <= hi]


def draw_labels(draw, items, font_size=12):
    font = plots._font(font_size)
    placed = []
    for x, y, text in items:
        width = draw.textlength(text, font=font)
        for dx, dy in ((8, 0), (8, -13), (8, 13), (-8 - width, 0), (8, -26), (8, 26)):
            box = (x + dx, y + dy - 7, x + dx + width, y + dy + 7)
            if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3]
                       for b in placed):
                draw.text((x + dx, y + dy), text, fill=INK, font=font, anchor="lm")
                placed.append(box)
                break
        else:
            draw.text((x + 8, y), text, fill=INK, font=font, anchor="lm")
            placed.append((x + 8, y - 7, x + 8 + width, y + 7))


class Dataset:
    def __init__(self, root: Path):
        inventory = json.loads((root / "artifacts" / "component_inventory.json").read_text(encoding="utf-8"))
        self.components = sorted(inventory.get("main_sweep", []))
        self.meta = {c: inventory["components"][c] for c in self.components}
        self.formats = [f for f in FORMATS if (root / f / "results" / "summary.csv").is_file()]
        self.total_params = sum(self.meta[c]["params"] for c in self.components)
        self.total_kmac = sum(self.meta[c]["kmac_per_px"] for c in self.components)

        self.qp = defaultdict(dict)
        self.size = {}
        self.bits = {}
        self.fp32 = {component: 0.0 for component in self.components}

        for fmt in self.formats:
            configs = {row["component"]: row for row in read_csv(root / fmt / "results" / f"configs_{fmt}.csv")}
            for component, row in configs.items():
                if component not in self.meta:
                    continue
                self.size[(component, fmt)] = number(row.get("quant_bytes"))
                self.bits[(component, fmt)] = number(row.get("effective_bits_per_param"))
                fp32 = number(row.get("fp32_bytes"))
                if fp32:
                    self.fp32[component] = fp32
            for row in read_csv(root / fmt / "results" / "summary.csv"):
                if row.get("control") != "none":
                    continue
                component = row.get("component")
                if component not in self.meta:
                    continue
                penalty = number(row.get("mean_penalty_pct"))
                if penalty is None:
                    continue
                qp = int(row["qp"])
                key = (component, fmt)
                current = self.qp[key].get(qp)
                self.qp[key][qp] = penalty if current is None else max(current, penalty)

        for component in self.components:
            if not self.fp32[component]:
                known = [self.size.get((component, f)) for f in self.formats]
                known = [s for s in known if s]
                self.fp32[component] = max(known, default=0.0) / 0.5
            self.size[(component, "fp32")] = self.fp32[component]
            self.qp[(component, "fp32")] = {q: 0.0 for q in range(8)}
            self.bits[(component, "fp32")] = 32.0

    def worst(self, component, fmt):
        values = self.qp.get((component, fmt), {})
        return max(values.values()) if values else None

    def worst_qp(self, component, fmt):
        values = self.qp.get((component, fmt), {})
        if not values:
            return None
        return max(values, key=lambda q: values[q])

    def safe_level(self, component):
        options = []
        for fmt in self.formats:
            penalty = self.worst(component, fmt)
            bits = self.bits.get((component, fmt))
            if penalty is None or bits is None or penalty > GREEN:
                continue
            options.append((bits, penalty, fmt))
        if not options:
            return "fp32"
        options.sort()
        return options[0][2]

    def tiers(self, component):
        grouped = defaultdict(list)
        for fmt in self.formats:
            size = self.size.get((component, fmt))
            penalty = self.worst(component, fmt)
            if size is None or penalty is None:
                continue
            grouped[round(size)].append((penalty, fmt))
        ladder = [(self.fp32[component], 0.0, "fp32")]
        for size in sorted(grouped, reverse=True):
            penalty, fmt = min(grouped[size])
            ladder.append((float(size), penalty, fmt))
        return ladder


def figure_ranking(data: Dataset, out: Path):
    rows = sorted(((c, data.worst(c, "int8") or 0.0) for c in data.components), key=lambda r: -r[1])
    width = 1500
    height = 150 + 40 * len(rows)
    image, draw = plots._canvas(width, height)
    left, top, plot_w = 110, 130, 790
    value_x = left + plot_w + 40
    info_x = value_x + 96
    plot_h = height - top - 90
    row_h = plot_h / len(rows)
    lo, hi = -1.0, 2.0

    for tick in log_ticks(lo, hi, [0.1, 0.3, 1, 3, 10, 30]):
        x = axis_x(tick, lo, hi, left, plot_w)
        draw.line((x, top, x, top + plot_h), fill=GRID, width=1)
        plots._text(draw, (x, top - 16), f"{tick:g}%", size=12, anchor="mb", fill=AXIS)
    draw.line((left, top, left, top + plot_h), fill=AXIS, width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=AXIS, width=2)
    for threshold, color, label in ((GREEN, GREEN_C, "1%"), (YELLOW, YELLOW_C, "3%")):
        x = axis_x(threshold, lo, hi, left, plot_w)
        draw.line((x, top, x, top + plot_h), fill=color, width=2)

    plots._text(draw, (value_x, top - 16), "penalty", size=12, anchor="mb", fill=AXIS)
    plots._text(draw, (info_x, top - 16), "params      share  kMAC/px   INT8", size=12, anchor="mb", fill=AXIS)

    for index, (component, penalty) in enumerate(rows):
        y = top + index * row_h + row_h / 2
        color = ramp(penalty)
        plots._text(draw, (left - 14, y), component, size=14, anchor="rm", fill=INK)
        x1 = axis_x(penalty, lo, hi, left, plot_w)
        draw.rectangle((left + 2, y - row_h * 0.3, x1, y + row_h * 0.3), fill=color)
        plots._text(draw, (value_x, y), f"{fmt_value(penalty)}%", size=12, anchor="lm", fill=INK)
        meta = data.meta[component]
        share = 100.0 * meta["params"] / data.total_params
        share_text = f"{share:5.2f}" if share < 1.0 else f"{share:5.1f}"
        plots._text(draw, (info_x, y),
                    f"{meta['params']:>9,}  {share_text}%  {meta['kmac_per_px']:6.2f}   {verdict(penalty)}",
                    size=12, anchor="lm", fill=INK)

    plots._text(draw, (width // 2, 40), "INT8 weight-only: which components survive",
                size=21, anchor="mt", fill=INK)
    plots._text(draw, (width // 2, 72),
                "worst of Kodak and CLIC across all 8 QPs, log scale; green line 1%, yellow line 3%",
                size=13, anchor="mt", fill=AXIS)
    plots._text(draw, (left, height - 40), "penalty = rate-equivalent cost %  (lower is safer)",
                size=12, anchor="lm", fill=AXIS)
    plots._save(image, out)


def figure_ladder(data: Dataset, out: Path):
    mean_bits = {}
    for fmt in data.formats:
        values = [data.bits.get((c, fmt)) for c in data.components]
        values = [v for v in values if v]
        mean_bits[fmt] = statistics.mean(values) if values else 99.0
    columns = ["fp32"] + sorted(data.formats, key=lambda f: -mean_bits[f])
    rows = sorted(data.components, key=lambda c: -(data.worst(c, "int8") or 0.0))
    cell_w, cell_h = 118, 42
    left, top = 230, 165
    right_block = 300
    width = left + cell_w * len(columns) + right_block
    height = top + cell_h * len(rows) + 105
    image, draw = plots._canvas(width, height)
    info_x = left + cell_w * len(columns) + 16

    for col, fmt in enumerate(columns):
        x = left + col * cell_w + cell_w / 2
        plots._text(draw, (x, top - 46), PRETTY[fmt], size=13, anchor="mb", fill=INK)
        if fmt != "fp32":
            bits = statistics.mean([data.bits[(c, fmt)] or 0 for c in data.components])
            plots._text(draw, (x, top - 26), f"{bits:.1f} b/w", size=11, anchor="mb", fill=AXIS)
        else:
            plots._text(draw, (x, top - 26), "32.0 b/w", size=11, anchor="mb", fill=AXIS)
    plots._text(draw, (info_x, top - 46), "chosen level", size=13, anchor="lm", fill=INK)
    plots._text(draw, (info_x, top - 26), "size at that level", size=11, anchor="lm", fill=AXIS)

    for row, component in enumerate(rows):
        y = top + row * cell_h
        plots._text(draw, (left - 14, y + cell_h / 2), component, size=14, anchor="rm", fill=INK)
        chosen = data.safe_level(component)
        for col, fmt in enumerate(columns):
            x = left + col * cell_w
            penalty = 0.0 if fmt == "fp32" else data.worst(component, fmt)
            color = (238, 242, 238) if fmt == "fp32" else ramp(penalty)
            draw.rectangle((x, y, x + cell_w - 2, y + cell_h - 2), fill=color)
            label = "0" if fmt == "fp32" else fmt_value(penalty)
            plots._text(draw, (x + cell_w / 2 - 1, y + cell_h / 2 - 1), label,
                        size=12, anchor="mm", fill=ink_on(color))
            if fmt == chosen:
                draw.rectangle((x + 1, y + 1, x + cell_w - 4, y + cell_h - 4),
                               outline=(20, 20, 20), width=3)

        size = data.size.get((component, chosen)) or data.fp32[component]
        saved = 100.0 * (1.0 - size / data.fp32[component]) if data.fp32[component] else 0.0
        plots._text(draw, (info_x, y + cell_h / 2 - 8), PRETTY[chosen], size=13, anchor="lm", fill=INK)
        plots._text(draw, (info_x, y + cell_h / 2 + 8),
                    f"{size_text(size / 1e6)} MB  -{saved:.0f}%", size=11, anchor="lm", fill=AXIS)

    total_fp32 = sum(data.fp32.values())
    total_safe = sum(data.size.get((c, data.safe_level(c))) or data.fp32[c] for c in data.components)
    plots._text(draw, (width // 2, 40), "Precision ladder: how far each component can be pushed",
                size=21, anchor="mt", fill=INK)
    plots._text(draw, (width // 2, 72),
                f"cell = worst-QP penalty %   boxed = most aggressive level still under 1%   "
                f"total {total_safe / 1e6:.2f} MB vs {total_fp32 / 1e6:.2f} MB FP32 "
                f"(-{100 * (1 - total_safe / total_fp32):.1f}%)",
                size=13, anchor="mt", fill=AXIS)
    plots._text(draw, (left, height - 52),
                "Columns are ordered by measured effective bits per weight, not by name.",
                size=12, anchor="lm", fill=AXIS)
    plots._text(draw, (left, height - 30),
                "INT4, FP8 and FP4 are simulated on this GPU: their columns are quality-only, never speed.",
                size=12, anchor="lm", fill=AXIS)
    plots._save(image, out)


def figure_exposure(data: Dataset, out: Path):
    rows = []
    for component in data.components:
        meta = data.meta[component]
        penalty = data.worst(component, "int8")
        rows.append((
            component,
            100.0 * meta["params"] / data.total_params,
            100.0 * meta["kmac_per_px"] / data.total_kmac,
            penalty if penalty is not None else 0.0,
        ))

    width, height = 1350, 820
    left, right, top, bottom = 130, 300, 130, 110
    image, draw = plots._canvas(width, height)
    plot_w, plot_h = width - left - right, height - top - bottom
    lo, hi = -2.0, math.log10(40)
    ylo, yhi = -1.0, math.log10(60)

    for tick in log_ticks(lo, hi, [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]):
        x = axis_x(tick, lo, hi, left, plot_w)
        draw.line((x, top, x, top + plot_h), fill=GRID, width=1)
        plots._text(draw, (x, top + plot_h + 18), f"{tick:g}%", size=12, anchor="mt", fill=AXIS)
    for tick in log_ticks(ylo, yhi, [0.1, 0.3, 1, 3, 10, 30]):
        y = axis_y(tick, ylo, yhi, top, plot_h)
        draw.line((left, y, left + plot_w, y), fill=GRID, width=1)
        plots._text(draw, (left - 12, y), f"{tick:g}%", size=12, anchor="rm", fill=AXIS)
    draw.line((left, top, left, top + plot_h), fill=AXIS, width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=AXIS, width=2)
    for threshold, color in ((GREEN, GREEN_C), (YELLOW, YELLOW_C)):
        draw.line((left, axis_y(threshold, ylo, yhi, top, plot_h),
                   left + plot_w, axis_y(threshold, ylo, yhi, top, plot_h)), fill=color, width=2)
    x5 = axis_x(5.0, lo, hi, left, plot_w)
    for y in range(int(top), int(top + plot_h), 12):
        draw.line((x5, y, x5, min(y + 6, top + plot_h)), fill=(150, 150, 150), width=1)
    plots._text(draw, (x5 + 6, top + 8), "5% of params", size=11, anchor="lt", fill=AXIS)

    verdict_color = {"green": GREEN_C, "yellow": YELLOW_C, "red": (204, 45, 40)}

    def bubble_radius(mac_share):
        return 2.5 + 3.4 * math.sqrt(max(mac_share, 0.05))

    labels = []
    for component, share, mac_share, penalty in rows:
        radius = bubble_radius(mac_share)
        x = axis_x(share, lo, hi, left, plot_w)
        y = axis_y(max(penalty, 0.1), ylo, yhi, top, plot_h)
        color = verdict_color[verdict(penalty)]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=color, outline=WHITE, width=2)
        labels.append((x + radius, y, component))
    draw_labels(draw, labels)

    legend_x = left + plot_w + 26
    for index, (name, color) in enumerate((("green INT8", GREEN_C), ("yellow INT8", YELLOW_C), ("red INT8", (204, 45, 40)))):
        y = top + 16 + index * 26
        draw.ellipse((legend_x, y - 7, legend_x + 14, y + 7), fill=color)
        plots._text(draw, (legend_x + 24, y), name, size=13, anchor="lm", fill=INK)
    for index, mac_share in enumerate((1, 5, 20)):
        y = top + 120 + index * 34
        radius = bubble_radius(mac_share)
        centre = legend_x + 8
        draw.ellipse((centre - radius, y - radius, centre + radius, y + radius), fill=(198, 198, 198))
        plots._text(draw, (centre + radius + 12, y), f"{mac_share}% of MACs", size=12, anchor="lm", fill=AXIS)
    plots._text(draw, (legend_x, top + 236), "x = share of params", size=12, anchor="lm", fill=AXIS)
    plots._text(draw, (legend_x, top + 254), "bubble = share of MACs", size=12, anchor="lm", fill=AXIS)

    plots._text(draw, (width // 2, 40), "Where the model mass sits versus how fragile it is",
                size=21, anchor="mt", fill=INK)
    plots._text(draw, (width // 2, 72),
                "INT8 weight-only, worst of Kodak and CLIC across all 8 QPs; both axes log",
                size=13, anchor="mt", fill=AXIS)
    axis_title(draw, left, top - 32, "INT8 penalty %")
    plots._text(draw, (left + plot_w, height - 34), "share of neural parameters", size=14, anchor="rm", fill=INK)
    plots._save(image, out)



def figure_qp(data: Dataset, out: Path):
    families = ["encoder", "entropy", "decoder"]
    width, height = 1500, 780
    left, top, gap = 100, 140, 34
    panel_w = (width - left - 50 - gap * 2) // 3
    panel_h = 380
    legend_top = top + panel_h + 92
    image, draw = plots._canvas(width, height)

    every = [v for c in data.components for v in data.qp.get((c, "int8"), {}).values()]
    ylo = -2.0
    yhi = math.log10(max(every + [1.0])) + 0.2

    for panel, family in enumerate(families):
        components = [c for c in data.components if FAMILY.get(c[0]) == family]
        x0 = left + panel * (panel_w + gap)

        def px(index):
            return x0 + panel_w * index / 7

        for tick in log_ticks(ylo, yhi, [0.01, 0.1, 1, 3, 10, 30, 100]):
            y = axis_y(tick, ylo, yhi, top, panel_h)
            draw.line((x0, y, x0 + panel_w, y), fill=GRID, width=1)
            if panel == 0:
                plots._text(draw, (x0 - 10, y), f"{tick:g}%", size=12, anchor="rm", fill=AXIS)
        for threshold, color in ((GREEN, GREEN_C), (YELLOW, YELLOW_C)):
            draw.line((x0, axis_y(threshold, ylo, yhi, top, panel_h),
                       x0 + panel_w, axis_y(threshold, ylo, yhi, top, panel_h)), fill=color, width=2)
        draw.line((x0, top, x0, top + panel_h), fill=AXIS, width=2)
        draw.line((x0, top + panel_h, x0 + panel_w, top + panel_h), fill=AXIS, width=2)
        for index in range(8):
            plots._text(draw, (px(index), top + panel_h + 14), index, size=12, anchor="mt", fill=AXIS)
        plots._text(draw, (x0 + panel_w / 2, top + panel_h + 38), "QP", size=13, anchor="mt", fill=INK)
        plots._text(draw, (x0 + panel_w / 2, top - 18), family, size=16, anchor="mt", fill=INK)

        for index, component in enumerate(components):
            color = plots.PALETTE[index % len(plots.PALETTE)]
            values = data.qp.get((component, "int8"), {})
            path_points = [(px(q), axis_y(values[q], ylo, yhi, top, panel_h))
                           for q in range(8) if q in values]
            if len(path_points) > 1:
                draw.line(path_points, fill=color, width=3)
            for x, y in path_points:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            col = index % 2
            row = index // 2
            lx = x0 + col * (panel_w * 0.5) + 4
            ly = legend_top + row * 21
            draw.line((lx, ly, lx + 20, ly), fill=color, width=3)
            plots._text(draw, (lx + 26, ly), component, size=12, anchor="lm", fill=INK)

    plots._text(draw, (width // 2, 40), "INT8 penalty versus QP: encoder damage grows, entropy damage shrinks",
                size=21, anchor="mt", fill=INK)
    plots._text(draw, (width // 2, 72),
                "shared log scale across panels; worst of Kodak and CLIC per QP; green 1%, yellow 3%",
                size=13, anchor="mt", fill=AXIS)
    axis_title(draw, left, top - 46, "penalty %")
    plots._save(image, out)


def figure_card(data: Dataset, out: Path):
    records = []
    for component in sorted(data.components, key=lambda c: -(data.worst(c, "int8") or 0.0)):
        meta = data.meta[component]
        int8 = data.worst(component, "int8")
        safe = data.safe_level(component)
        size = data.size.get((component, safe)) or data.fp32[component]
        share = 100.0 * meta["params"] / data.total_params
        mac_share = 100.0 * meta["kmac_per_px"] / data.total_kmac
        records.append({
            "component": component,
            "family": FAMILY.get(component[0], "other"),
            "params": meta["params"],
            "param_share_pct": round(share, 2),
            "kmac_per_px": meta["kmac_per_px"],
            "mac_share_pct": round(mac_share, 2),
            "int8_worst_penalty_pct": None if int8 is None else round(int8, 3),
            "int8_verdict": verdict(int8),
            "safe_level": safe,
            "safe_level_penalty_pct": round(data.worst(component, safe) or 0.0, 3),
            "safe_level_qp": data.worst_qp(component, safe),
            "size_mb": round(size / 1e6, 4),
            "fp32_mb": round(data.fp32[component] / 1e6, 4),
            "saved_pct": round(100.0 * (1 - size / data.fp32[component]), 1) if data.fp32[component] else 0.0,
            "needs_qat": verdict(int8) == "red" and max(share, mac_share) >= 5.0,
            "qat_candidate": verdict(int8) != "green" and max(share, mac_share) >= 5.0,
        })
    write_csv(out / "decision_summary.csv", records)

    headers = ["comp", "fam", "params", "%par", "kMAC/px", "INT8", "level", "pen@lvl", "QP", "MB", "save%", "QAT"]
    widths = [62, 66, 92, 60, 82, 84, 92, 82, 48, 72, 68, 66]
    left, top = 40, 170
    row_h = 33
    width = left + sum(widths) + 24
    height = top + row_h * (len(records) + 1) + 96
    image, draw = plots._canvas(width, height)

    x = left
    for header, w in zip(headers, widths):
        plots._text(draw, (x, top), header, size=13, anchor="lm", fill=AXIS)
        x += w
    draw.line((left, top + 14, left + sum(widths), top + 14), fill=AXIS, width=2)

    for index, row in enumerate(records):
        y = top + 30 + index * row_h
        x = left
        plots._text(draw, (x, y), row["component"], size=14, anchor="lm", fill=INK)
        x += widths[0]
        plots._text(draw, (x, y), row["family"][:3], size=12, anchor="lm", fill=AXIS)
        x += widths[1]
        plots._text(draw, (x, y), f"{row['params']:,}", size=12, anchor="lm", fill=INK)
        x += widths[2]
        plots._text(draw, (x, y), f"{row['param_share_pct']:.1f}", size=12, anchor="lm", fill=INK)
        x += widths[3]
        plots._text(draw, (x, y), f"{row['kmac_per_px']:.2f}", size=12, anchor="lm", fill=INK)
        x += widths[4]
        color = ramp(row["int8_worst_penalty_pct"])
        draw.rectangle((x, y - 11, x + widths[5] - 12, y + 11), fill=color)
        plots._text(draw, (x + 6, y), f"{fmt_value(row['int8_worst_penalty_pct'])}%",
                    size=12, anchor="lm", fill=ink_on(color))
        x += widths[5]
        plots._text(draw, (x, y), PRETTY[row["safe_level"]], size=12, anchor="lm", fill=INK)
        x += widths[6]
        plots._text(draw, (x, y), f"{row['safe_level_penalty_pct']:.2f}", size=12, anchor="lm", fill=INK)
        x += widths[7]
        plots._text(draw, (x, y), str(row["safe_level_qp"]), size=12, anchor="lm", fill=AXIS)
        x += widths[8]
        plots._text(draw, (x, y), size_text(row["size_mb"]), size=12, anchor="lm", fill=INK)
        x += widths[9]
        plots._text(draw, (x, y), f"{row['saved_pct']:.0f}", size=12, anchor="lm", fill=INK)
        x += widths[10]
        plots._text(draw, (x, y), "YES" if row["needs_qat"] else "no", size=12, anchor="lm",
                    fill=(200, 40, 40) if row["needs_qat"] else AXIS)

    total_fp32 = sum(r["fp32_mb"] for r in records)
    total_safe = sum(r["size_mb"] for r in records)
    plots._text(draw, (left, height - 52),
                f"At a 1% per-component budget: {total_safe:.2f} MB vs {total_fp32:.2f} MB FP32 "
                f"(-{100 * (1 - total_safe / total_fp32):.1f}%)",
                size=15, anchor="lm", fill=INK)
    plots._text(draw, (left, height - 30),
                "QAT = red at INT8 while holding >= 5% of params or MACs. "
                "No component meets that trigger; E1 is the only large yellow.",
                size=12, anchor="lm", fill=AXIS)
    plots._text(draw, (width // 2, 40), "Per-component decision card", size=22, anchor="mt", fill=INK)
    plots._text(draw, (width // 2, 74),
                "level = most aggressive precision whose worst-QP penalty stays under 1%; "
                "QP = the QP that drives that worst case",
                size=12, anchor="mt", fill=AXIS)
    plots._save(image, out / "decision_card.png")
    return records


def greedy_frontier(data: Dataset, budget=GREEN):
    ladders = {c: data.tiers(c) for c in data.components}
    index = {c: 0 for c in data.components}
    penalty_of = {c: 0.0 for c in data.components}
    size_of = {c: data.fp32[c] for c in data.components}
    steps = [(sum(size_of.values()) / 1e6, 0.0)]
    assignment = {c: "fp32" for c in data.components}

    for _ in range(len(data.components) * len(FORMATS) + 1):
        best = None
        for component in data.components:
            ladder = ladders[component]
            if index[component] + 1 >= len(ladder):
                continue
            candidate_size, candidate_penalty, fmt = ladder[index[component] + 1]
            saved = (size_of[component] - candidate_size) / 1e6
            delta_penalty = candidate_penalty - penalty_of[component]
            if delta_penalty <= 0:
                efficiency = float("inf") if saved > 1e-9 else 0.0
            elif saved <= 1e-9:
                continue
            else:
                efficiency = saved / delta_penalty
            if best is None or efficiency > best[0]:
                best = (efficiency, component, fmt, candidate_size, candidate_penalty)
        if best is None:
            break
        _, component, fmt, candidate_size, candidate_penalty = best
        index[component] += 1
        penalty_of[component] = candidate_penalty
        size_of[component] = candidate_size
        total_penalty = sum(penalty_of.values())
        steps.append((sum(size_of.values()) / 1e6, total_penalty))
        if total_penalty <= budget:
            assignment = {c: ladders[c][index[c]][2] for c in data.components}
    return steps, assignment


def figure_budget(data: Dataset, out: Path):
    steps, levels = greedy_frontier(data)
    width, height = 1300, 790
    left, right, top, bottom = 120, 320, 130, 110
    image, draw = plots._canvas(width, height)
    plot_w, plot_h = width - left - right, height - top - bottom

    cap = 100.0
    steps = [s for s in steps if s[1] <= cap] or steps[:2]

    uniform = {}
    for fmt in ["fp16", "bf16", "int8"]:
        size = sum(data.size.get((c, fmt)) or 0 for c in data.components) / 1e6
        penalty = sum(data.worst(c, fmt) or 0 for c in data.components)
        uniform[fmt] = (size, penalty)
    fp32_size = sum(data.fp32.values()) / 1e6
    xlo, xhi = 0.0, math.log10(max(fp32_size, max(v[0] for v in uniform.values())) * 1.15)
    ylo, yhi = -1.5, math.log10(cap) + 0.12

    def px(value):
        if value <= 0:
            return left
        return left + plot_w * min(1.0, (math.log10(value) - math.log10(1.0)) / (xhi - math.log10(1.0)))

    def py(value):
        return axis_y(max(value, 0.1), ylo, yhi, top, plot_h)

    for tick in log_ticks(ylo, yhi, [0.03, 0.1, 1, 3, 10, 100]):
        y = py(tick)
        draw.line((left, y, left + plot_w, y), fill=GRID, width=1)
        plots._text(draw, (left - 12, y), f"{tick:g}%", size=12, anchor="rm", fill=AXIS)
    for tick in log_ticks(math.log10(1.0), xhi, [1, 2, 4, 8, 16, 32, 64]):
        x = px(tick)
        draw.line((x, top, x, top + plot_h), fill=(242, 242, 242), width=1)
        plots._text(draw, (x, top + plot_h + 18), tick, size=12, anchor="mt", fill=AXIS)
    for threshold, color in ((GREEN, GREEN_C), (YELLOW, YELLOW_C)):
        draw.line((left, py(threshold), left + plot_w, py(threshold)), fill=color, width=2)
    draw.line((left, top, left, top + plot_h), fill=AXIS, width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=AXIS, width=2)

    draw.line([(px(x), py(y)) for x, y in steps], fill=(31, 119, 180), width=4)
    for x, y in steps:
        draw.ellipse((px(x) - 5, py(y) - 5, px(x) + 5, py(y) + 5), fill=(31, 119, 180))

    budget_size = sum(data.size.get((c, levels[c])) or data.fp32[c] for c in data.components) / 1e6
    draw.ellipse((px(budget_size) - 11, py(GREEN) - 11, px(budget_size) + 11, py(GREEN) + 11),
                 outline=(20, 20, 20), width=3)
    plots._text(draw, (px(budget_size) + 16, py(GREEN) - 16), f"{budget_size:.2f} MB at 1%",
                size=12, anchor="lm", fill=INK)

    for name, (x, y) in uniform.items():
        cx, cy = px(x), py(y)
        if y > 10 ** (yhi - 0.05):
            cy = top + 8
            draw.polygon([(cx, cy - 7), (cx - 7, cy + 5), (cx + 7, cy + 5)], fill=(214, 39, 40))
        else:
            draw.rectangle((cx - 6, cy - 6, cx + 6, cy + 6), fill=(214, 39, 40))
        plots._text(draw, (cx + 13, cy), PRETTY[name], size=12, anchor="lm", fill=INK)
    draw.ellipse((px(fp32_size) - 6, py(0.0) - 6, px(fp32_size) + 6, py(0.0) + 6), fill=(120, 120, 120))
    plots._text(draw, (px(fp32_size), py(0.0) - 20), "FP32", size=12, anchor="mm", fill=INK)

    counts = defaultdict(int)
    for _, fmt in levels.items():
        counts[PRETTY[fmt]] += 1
    breakdown = "  ".join(f"{k} {v}" for k, v in sorted(counts.items()))

    legend_x = left + plot_w + 26
    draw.line((legend_x, top + 16, legend_x + 30, top + 16), fill=(31, 119, 180), width=4)
    plots._text(draw, (legend_x + 38, top + 16), "greedy mixed", size=13, anchor="lm", fill=INK)
    draw.rectangle((legend_x + 8, top + 38, legend_x + 22, top + 52), fill=(214, 39, 40))
    plots._text(draw, (legend_x + 38, top + 45), "uniform precision", size=13, anchor="lm", fill=INK)
    plots._text(draw, (legend_x, top + 88), "Prediction is additive:", size=12, anchor="lm", fill=INK)
    plots._text(draw, (legend_x, top + 106), "single-component penalties", size=12, anchor="lm", fill=AXIS)
    plots._text(draw, (legend_x, top + 124), "summed. The whole-model", size=12, anchor="lm", fill=AXIS)
    plots._text(draw, (legend_x, top + 142), "run must confirm it.", size=12, anchor="lm", fill=AXIS)
    plots._text(draw, (legend_x, top + 182), "Greedy assignment at 1%:", size=12, anchor="lm", fill=INK)
    plots._text(draw, (legend_x, top + 200), breakdown, size=11, anchor="lm", fill=AXIS)
    plots._text(draw, (legend_x, top + 218),
                f"{budget_size:.2f} MB, -{100 * (1 - budget_size / fp32_size):.1f}% vs FP32",
                size=12, anchor="lm", fill=INK)

    plots._text(draw, (width // 2, 40), "Size budget frontier: greedy mixed versus uniform precision",
                size=21, anchor="mt", fill=INK)
    plots._text(draw, (width // 2, 72),
                "x log scale; uniform INT4 and FP4 exceed 100% and are not shown",
                size=13, anchor="mt", fill=AXIS)
    axis_title(draw, left, top - 30, "predicted penalty %")
    plots._text(draw, (left + plot_w, height - 34), "total neural size (MB)", size=14, anchor="rm", fill=INK)
    plots._save(image, out)
    return steps, levels, budget_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="quantization-pluse-mse")
    parser.add_argument("--output", default="quantization-pluse-mse/results/decision")
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    data = Dataset(root)
    if not data.formats:
        raise SystemExit(f"no completed precision surfaces under {root}")

    figure_ranking(data, output / "decision_component_ranking.png")
    figure_ladder(data, output / "decision_precision_ladder.png")
    figure_exposure(data, output / "decision_exposure.png")
    figure_qp(data, output / "decision_qp_sensitivity.png")
    records = figure_card(data, output)
    steps, levels, budget_size = figure_budget(data, output / "decision_budget_frontier.png")

    total_fp32 = sum(data.fp32.values())
    total_safe = sum(row["size_mb"] for row in records) * 1e6

    safe_assignment = {c: data.safe_level(c) for c in data.components}
    safe_assignment = {c: f for c, f in safe_assignment.items() if f != "fp32"}
    greedy_assignment = {c: levels[c] for c in data.components if levels[c] != "fp32"}
    for name, assignment, note in (
        ("safe", safe_assignment, "most aggressive precision under a 1% worst-QP penalty per component"),
        ("greedy", greedy_assignment, "greedy path at a 1% additive penalty budget"),
    ):
        payload = {
            "name": name,
            "note": note,
            "assignment": dict(sorted(assignment.items())),
        }
        (output / f"decision_assignment_{name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(json.dumps({
        "formats": data.formats,
        "components": len(data.components),
        "fp32_mb": round(total_fp32 / 1e6, 2),
        "safe_level_mb": round(total_safe / 1e6, 2),
        "safe_level_saving_pct": round(100 * (1 - total_safe / total_fp32), 2),
        "green_at_int8": sum(1 for r in records if r["int8_verdict"] == "green"),
        "yellow_at_int8": sum(1 for r in records if r["int8_verdict"] == "yellow"),
        "red_at_int8": sum(1 for r in records if r["int8_verdict"] == "red"),
        "needs_qat": [r["component"] for r in records if r["needs_qat"]],
        "qat_candidates": [r["component"] for r in records if r["qat_candidate"]],
        "safe_components_quantized": len(safe_assignment),
        "greedy_components_quantized": len(greedy_assignment),
        "greedy_1pct_size_mb": round(budget_size, 2),
        "greedy_1pct_saving_pct": round(100 * (1 - budget_size / (total_fp32 / 1e6)), 1),
        "greedy_1pct_levels": {k: v for k, v in sorted(levels.items())},
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
