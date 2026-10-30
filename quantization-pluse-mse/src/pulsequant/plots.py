"""Small Pillow-based plots so the harness has no plotting dependency."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PALETTE = [
    (31, 119, 180),
    (214, 39, 40),
    (44, 160, 44),
    (148, 103, 189),
    (255, 127, 14),
    (23, 190, 207),
    (227, 119, 194),
    (188, 189, 34),
]


def _font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _canvas(width=1200, height=700):
    image = Image.new("RGB", (width, height), "white")
    return image, ImageDraw.Draw(image)


def _text(draw, xy, text, size=14, fill=(30, 30, 30), anchor=None):
    draw.text(xy, str(text), fill=fill, font=_font(size), anchor=anchor)


def _save(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _nice_ticks(vmin, vmax, count=6):
    if vmin == vmax:
        vmin -= 1
        vmax += 1
    span = vmax - vmin
    step = 10 ** math.floor(math.log10(span / max(1, count - 1)))
    for factor in (1, 2, 5, 10):
        if span / (step * factor) <= count:
            step *= factor
            break
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    ticks = []
    value = start
    while value <= end + step * 0.5:
        ticks.append(round(value, 10))
        value += step
    return ticks


def line_chart(path, title, x_labels, series, y_label):
    width, height = 1200, 700
    left, right, top, bottom = 110, 300, 80, 110
    image, draw = _canvas(width, height)
    values = [value for _, data, _ in series for value in data if value is not None]
    if not values:
        values = [0.0]
    vmin, vmax = min(values + [0.0]), max(values + [0.0])
    ticks = _nice_ticks(vmin, vmax)
    vmin, vmax = ticks[0], ticks[-1]
    plot_w = width - left - right
    plot_h = height - top - bottom

    def px(index):
        return left + (plot_w * index / max(1, len(x_labels) - 1))

    def py(value):
        return top + plot_h * (1 - (value - vmin) / (vmax - vmin))

    for tick in ticks:
        y = py(tick)
        draw.line((left, y, left + plot_w, y), fill=(230, 230, 230), width=1)
        _text(draw, (left - 12, y), f"{tick:.3g}", size=13, anchor="rm")
    if vmin <= 0 <= vmax:
        draw.line((left, py(0), left + plot_w, py(0)), fill=(0, 0, 0), width=1)
    draw.line((left, top, left, top + plot_h), fill=(80, 80, 80), width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(80, 80, 80), width=2)
    for index, label in enumerate(x_labels):
        _text(draw, (px(index), top + plot_h + 18), label, size=13, anchor="mt")

    for series_index, (name, data, color) in enumerate(series):
        points = [(px(i), py(value)) for i, value in enumerate(data) if value is not None]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        legend_y = top + 24 + series_index * 28
        draw.line((left + plot_w + 24, legend_y, left + plot_w + 60, legend_y), fill=color, width=4)
        _text(draw, (left + plot_w + 68, legend_y), name, size=14, anchor="lm")

    _text(draw, (width // 2, 34), title, size=20, anchor="mt")
    _text(draw, (width // 2, height - 28), "QP", size=15, anchor="mm")
    _text(draw, (24, top + plot_h // 2), y_label, size=15, anchor="lm")
    _save(image, path)


def bar_chart(path, title, categories, series, y_label):
    width, height = 1200, 700
    left, right, top, bottom = 110, 300, 80, 120
    image, draw = _canvas(width, height)
    values = [value for _, data, _ in series for value in data if value is not None]
    if not values:
        values = [0.0]
    vmin, vmax = min(values + [0.0]), max(values + [0.0])
    ticks = _nice_ticks(vmin, vmax)
    vmin, vmax = ticks[0], ticks[-1]
    plot_w = width - left - right
    plot_h = height - top - bottom

    def py(value):
        return top + plot_h * (1 - (value - vmin) / (vmax - vmin))

    for tick in ticks:
        y = py(tick)
        draw.line((left, y, left + plot_w, y), fill=(230, 230, 230), width=1)
        _text(draw, (left - 12, y), f"{tick:.3g}", size=13, anchor="rm")
    if vmin <= 0 <= vmax:
        draw.line((left, py(0), left + plot_w, py(0)), fill=(0, 0, 0), width=1)
    draw.line((left, top, left, top + plot_h), fill=(80, 80, 80), width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(80, 80, 80), width=2)

    group_w = plot_w / max(1, len(categories))
    bar_w = group_w * 0.8 / max(1, len(series))
    for cat_index, category in enumerate(categories):
        base_x = left + cat_index * group_w + group_w * 0.1
        _text(draw, (left + cat_index * group_w + group_w / 2, top + plot_h + 20),
              category, size=14, anchor="mt")
        for series_index, (name, data, color) in enumerate(series):
            value = data[cat_index]
            if value is None:
                continue
            x0 = base_x + series_index * bar_w
            x1 = x0 + bar_w * 0.9
            y0 = py(0)
            y1 = py(value)
            draw.rectangle((x0, min(y0, y1), x1, max(y0, y1)), fill=color)
    for series_index, (name, _, color) in enumerate(series):
        legend_y = top + 24 + series_index * 28
        draw.rectangle((left + plot_w + 24, legend_y - 7, left + plot_w + 52, legend_y + 7), fill=color)
        _text(draw, (left + plot_w + 60, legend_y), name, size=14, anchor="lm")

    _text(draw, (width // 2, 34), title, size=20, anchor="mt")
    _text(draw, (24, top + plot_h // 2), y_label, size=15, anchor="lm")
    _save(image, path)


def h_bar_chart(path, title, labels, values, x_label):
    width = 1200
    height = max(500, 80 + 34 * len(labels))
    left, right, top, bottom = 240, 100, 80, 70
    image, draw = _canvas(width, height)
    vmax = max(values + [0.0])
    vmin = min(values + [0.0])
    if vmax == vmin:
        vmax = vmin + 1
    plot_w = width - left - right
    plot_h = height - top - bottom
    row_h = plot_h / max(1, len(labels))

    def px(value):
        return left + plot_w * (value - vmin) / (vmax - vmin)

    draw.line((left, top, left, top + plot_h), fill=(80, 80, 80), width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(80, 80, 80), width=2)
    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + index * row_h + row_h / 2
        _text(draw, (left - 12, y), label, size=13, anchor="rm")
        x0 = px(0)
        x1 = px(value)
        draw.rectangle((min(x0, x1), y - row_h * 0.3, max(x0, x1), y + row_h * 0.3),
                       fill=(31, 119, 180))
        _text(draw, (max(x0, x1) + 8, y), f"{value:.2f}", size=12, anchor="lm")
    _text(draw, (width // 2, 34), title, size=20, anchor="mt")
    _text(draw, (width // 2, height - 24), x_label, size=14, anchor="mm")
    _save(image, path)


def heatmap(path, title, row_labels, col_labels, values, vmin=None, vmax=None):
    width = 1200
    height = max(500, 120 + 44 * len(row_labels))
    left, right, top, bottom = 220, 80, 90, 90
    image, draw = _canvas(width, height)
    flat = [value for row in values for value in row if value is not None]
    if vmin is None:
        vmin = min(flat + [0.0])
    if vmax is None:
        vmax = max(flat + [0.0])
    if vmin == vmax:
        vmax = vmin + 1
    plot_w = width - left - right
    plot_h = height - top - bottom
    cell_w = plot_w / max(1, len(col_labels))
    cell_h = plot_h / max(1, len(row_labels))

    for row_index, row_label in enumerate(row_labels):
        _text(draw, (left - 12, top + row_index * cell_h + cell_h / 2),
              row_label, size=13, anchor="rm")
        for col_index, value in enumerate(values[row_index]):
            x0 = left + col_index * cell_w
            y0 = top + row_index * cell_h
            if value is None:
                color = (235, 235, 235)
                label = "-"
            else:
                t = (value - vmin) / (vmax - vmin)
                if value < 0:
                    color = (int(220 - 180 * t), int(220 - 100 * t), 240)
                else:
                    color = (240, int(220 - 150 * t), int(220 - 180 * t))
                label = f"{value:.2f}"
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), fill=color, outline="white")
            _text(draw, (x0 + cell_w / 2, y0 + cell_h / 2), label, size=12, anchor="mm")

    for col_index, label in enumerate(col_labels):
        _text(draw, (left + col_index * cell_w + cell_w / 2, top + plot_h + 18),
              label, size=13, anchor="mt")
    _text(draw, (width // 2, 34), title, size=20, anchor="mt")
    _save(image, path)


def dashboard(path, title, image_paths, cols=2, thumb_width=600, thumb_height=350):
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    rows = max(1, math.ceil(len(images) / cols))
    width = cols * thumb_width
    height = 70 + rows * thumb_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    _text(draw, (width // 2, 30), title, size=22, anchor="mt")
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        thumb = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (col * thumb_width, 70 + row * thumb_height))
    _save(canvas, path)


def scatter(path, title, points, x_label, y_label):
    width, height = 1200, 700
    left, right, top, bottom = 110, 300, 80, 110
    image, draw = _canvas(width, height)
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    xmin, xmax = min(xs + [0.0]), max(xs + [0.0])
    ymin, ymax = min(ys + [0.0]), max(ys + [0.0])
    xticks = _nice_ticks(xmin, xmax)
    yticks = _nice_ticks(ymin, ymax)
    xmin, xmax = xticks[0], xticks[-1]
    ymin, ymax = yticks[0], yticks[-1]
    plot_w = width - left - right
    plot_h = height - top - bottom

    def px(value):
        return left + plot_w * (value - xmin) / (xmax - xmin)

    def py(value):
        return top + plot_h * (1 - (value - ymin) / (ymax - ymin))

    for tick in yticks:
        y = py(tick)
        draw.line((left, y, left + plot_w, y), fill=(230, 230, 230), width=1)
        _text(draw, (left - 12, y), f"{tick:.3g}", size=13, anchor="rm")
    for tick in xticks:
        x = px(tick)
        draw.line((x, top, x, top + plot_h), fill=(240, 240, 240), width=1)
        _text(draw, (x, top + plot_h + 18), f"{tick:.3g}", size=13, anchor="mt")
    draw.line((left, top, left, top + plot_h), fill=(80, 80, 80), width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(80, 80, 80), width=2)

    for index, (x, y) in enumerate(points):
        color = PALETTE[index % len(PALETTE)]
        draw.ellipse((px(x) - 6, py(y) - 6, px(x) + 6, py(y) + 6), fill=color)
    _text(draw, (width // 2, 34), title, size=20, anchor="mt")
    _text(draw, (width // 2, height - 28), x_label, size=15, anchor="mm")
    _text(draw, (24, top + plot_h // 2), y_label, size=15, anchor="lm")
    _save(image, path)
