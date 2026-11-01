"""Generate the 10-slide submission deck from the measured results.

Ten slides for a 25-minute slot. Content is merged, not removed: the knee, the 4-bit verdict and the
QAT setup move into the slides that depend on them.

    python quantization-pluse-mse/scripts/make_deck.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "quantization-pluse-mse"
OUT = ROOT / "submission" / "PULSE-quantization-deck.pptx"

NAVY = RGBColor(0x1E, 0x27, 0x61)
ICE = RGBColor(0xCA, 0xDC, 0xFC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MINT = RGBColor(0x02, 0xC3, 0x9A)
RED = RGBColor(0xD6, 0x45, 0x45)
SLATE = RGBColor(0x36, 0x45, 0x4F)
MUTED = RGBColor(0x7A, 0x8C, 0xA3)
CARD = RGBColor(0xEF, 0xF3, 0xF9)

BODY = "Calibri"
HEAD = "Cambria"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def textbox(s, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = anchor
    return frame


def para(frame, text, size, color, bold=False, first=False, space_after=6,
         font=BODY, align=PP_ALIGN.LEFT):
    p = frame.paragraphs[0] if first else frame.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    return p


def title(s, text, subtitle=None):
    frame = textbox(s, 0.62, 0.42, 12.1, 1.0)
    para(frame, text, 40, NAVY, bold=True, first=True, font=HEAD, space_after=0)
    if subtitle:
        sub = textbox(s, 0.62, 1.30, 12.1, 0.45)
        para(sub, subtitle, 17, MUTED, first=True, space_after=0)


def footer(s, text):
    frame = textbox(s, 0.62, 6.95, 12.1, 0.35)
    para(frame, text, 11, MUTED, first=True, space_after=0)


def card(s, x, y, w, h, big, label, accent=MINT, big_size=40, fill=CARD,
         anchor=MSO_ANCHOR.MIDDLE):
    shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.adjustments[0] = 0.08
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.shadow.inherit = False
    frame = shape.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(0.18)
    frame.margin_top = Inches(0.14)
    frame.margin_bottom = Inches(0.10)
    frame.vertical_anchor = anchor
    para(frame, big, big_size, accent, bold=True, first=True, font=HEAD, space_after=2)
    para(frame, label, 13, SLATE, space_after=0)
    return shape


def image(s, path, x, y, w, h):
    path = Path(path)
    if not path.is_file():
        return
    with Image.open(path) as handle:
        iw, ih = handle.size
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    s.shapes.add_picture(str(path), Inches(x + (w - dw) / 2), Inches(y + (h - dh) / 2),
                         Inches(dw), Inches(dh))


def table(s, x, y, w, h, header, rows, widths=None, size=13, highlight=()):
    shape = s.shapes.add_table(len(rows) + 1, len(header), Inches(x), Inches(y), Inches(w), Inches(h))
    tbl = shape.table
    if widths:
        for index, fraction in enumerate(widths):
            tbl.columns[index].width = Emu(int(Inches(w) * fraction))
    for col, text in enumerate(header):
        cell = tbl.cell(0, col)
        cell.text = ""
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if col == 0 else PP_ALIGN.RIGHT
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = True
        run.font.color.rgb = WHITE
        run.font.name = BODY
    for r, row in enumerate(rows, start=1):
        for col, text in enumerate(row):
            cell = tbl.cell(r, col)
            cell.text = ""
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD if r % 2 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if col == 0 else PP_ALIGN.RIGHT
            run = p.add_run()
            run.text = str(text)
            run.font.size = Pt(size)
            run.font.color.rgb = NAVY if r - 1 in highlight else SLATE
            run.font.bold = r - 1 in highlight
            run.font.name = BODY


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


def dark_slide():
    s = slide()
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    bg.shadow.inherit = False
    return s


# ================================================================== 1 title
s = dark_slide()
frame = textbox(s, 0.9, 1.9, 11.5, 1.9)
para(frame, "Which bits can we drop?", 54, ICE, bold=True, first=True, font=HEAD, space_after=8)
para(frame, "Component-wise quantization of a learned image codec", 24, WHITE, space_after=0)
frame = textbox(s, 0.9, 3.75, 11.5, 0.9)
para(frame, "PULSE-MSE  |  14.18M parameters  |  8 quality settings  |  one weight file",
     17, ICE, first=True, space_after=6)
para(frame, "We keep 39 times more quality than uniform INT8 at the same size.", 17, MINT, space_after=0)
card(s, 0.9, 5.0, 3.5, 1.2, "21.677 MB", "60.4 percent smaller. Loss is 0.94 percent.", big_size=32)
card(s, 4.65, 5.0, 3.5, 1.2, "18.664 MB", "65.9 percent smaller. Loss is 1.60 percent.", big_size=32)
card(s, 8.40, 5.0, 3.5, 1.2, "39x", "better than uniform INT8 at the same size", big_size=32)
notes(s, "One sentence: we measured which parts of a codec tolerate low precision, solved the choice as an "
         "integer program, and used fine-tuning where the choice was aggressive. Three numbers: two "
         "operating points and the ratio against the naive baseline.")

# ================================================================== 2 problem
s = slide()
title(s, "One precision for all parts is the wrong tool", "the parts are not equal")
table(s, 0.62, 2.0, 12.1, 1.6,
      ["Group", "Parameters", "Share", "What it means"],
      [["Encoder E0-E5", "12.13M", "85.6%", "four blocks alone hold 11.6 MB"],
       ["Entropy H0-H3", "1.34M", "9.4%", "takes 73 percent of a decode"],
       ["Decoder D0-D8", "708k", "5.0%", "small, but one 4.7k block loses 8 percent"]],
      widths=[0.22, 0.16, 0.12, 0.50], size=14)
frame = textbox(s, 0.62, 3.95, 12.1, 2.6)
para(frame, "Sensitivity does not follow size.", 20, NAVY, bold=True, first=True, font=HEAD, space_after=10)
para(frame, "A 2.89M encoder block loses 0.5 percent at 8 bits.", 17, SLATE, space_after=6)
para(frame, "A 4.7k renderer block loses 8 percent at the same precision.", 17, SLATE, space_after=6)
para(frame, "One weight file serves all eight settings. A choice per setting is not deployable.",
     17, SLATE, space_after=10)
para(frame, "So the task is an allocation problem: one precision per part, decided once.", 18, NAVY, bold=True,
     space_after=0)
footer(s, "artifacts/component_inventory.json")
notes(s, "The imbalance is the whole motivation. Uniform precision over-spends on robust parts and "
         "under-protects fragile ones. Note the inversion: the largest parts are the cheapest.")

# ================================================================== 3 method
s = slide()
title(s, "How we measured it", "69,233 measurements. One part at a time.")
for index, (head, body) in enumerate([
        ("7 formats", "FP16, BF16, FP8-E4M3, FP8-E5M2, INT8, INT4, FP4-E2M1"),
        ("19 parts x 8 settings", "weights only. All other parts stay in FP32."),
        ("Kodak and CLIC", "24 and 41 images. All eight quality settings.")]):
    card(s, 0.62 + index * 4.12, 2.05, 3.85, 1.5, head, body, accent=NAVY, big_size=24,
         anchor=MSO_ANCHOR.TOP)
frame = textbox(s, 0.62, 3.85, 12.1, 1.4)
para(frame, "Penalty = 100 x [(1 + dbpp/100) x 2^(-dPSNR/slope) - 1]", 17, SLATE, first=True, space_after=6)
para(frame, "One number holds a quality change and a bitrate change. Zero means equal to FP32.",
     15, MUTED, space_after=0)
card(s, 0.62, 5.35, 5.85, 1.35, "Additive: 0.96-1.05", "Single-part costs add up, so the solver can sum them.",
     big_size=26, anchor=MSO_ANCHOR.TOP)
card(s, 6.87, 5.35, 5.85, 1.35, "FP32 gate passes", "The reference bitstream matches at all eight settings.",
     big_size=26, anchor=MSO_ANCHOR.TOP)
footer(s, "scripts/pipeline.py (metric) | analyze_additivity.py (check)")
notes(s, "Two safeguards make the numbers usable. The FP32 gate proves the baseline is the baseline. The "
         "additivity test proves single-part costs can be added. Without both, the optimizer input is "
         "invalid.")

# ================================================================== 4 map
s = slide()
title(s, "Three parts refuse 8 bits", "13 of 19 are safe. The three that are not are small.")
image(s, W / "results/analysis/component_precision_penalty_heatmap.png", 0.62, 1.95, 7.6, 4.4)
frame = textbox(s, 8.5, 2.05, 4.2, 4.3)
para(frame, "Intolerant at INT8", 19, RED, bold=True, first=True, font=HEAD, space_after=8)
for text in ["H3 entropy step: 38.15%", "H0 hyper encoder: 10.09%", "D5 renderer: 8.06%"]:
    para(frame, text, 14, SLATE, space_after=4)
para(frame, "Together 3.8 percent of parameters.", 13, MUTED, space_after=12)
para(frame, "Cheap at INT8", 19, MINT, bold=True, font=HEAD, space_after=8)
for text in ["E2 / E3: 0.51 / 0.43%", "E1 / E4: 1.35 / 0.71%", "D1: 0.12%"]:
    para(frame, text, 14, SLATE, space_after=4)
footer(s, "results/analysis/component_precision_penalty_heatmap.png")
notes(s, "Read a cell as the percentage penalty to quantize that part alone. The four large encoder blocks "
         "are the cheapest rows in the table, which is why every tier targets them.")

# ================================================================== 5 allocation + frontier
s = slide()
title(s, "We solved the choice as an integer program", "one variable per part. Shared across all eight settings.")
frame = textbox(s, 0.62, 1.95, 5.9, 4.5)
para(frame, "Why shared variables", 18, NAVY, bold=True, first=True, font=HEAD, space_after=6)
para(frame, "One weight file serves all eight settings. A different assignment per setting would need "
            "eight files.", 15, SLATE, space_after=12)
para(frame, "Constraints", 18, NAVY, bold=True, font=HEAD, space_after=6)
para(frame, "One quality limit for each setting. Eight limits in total.", 15, SLATE, space_after=12)
para(frame, "Objective", 18, NAVY, bold=True, font=HEAD, space_after=6)
para(frame, "Minimise bytes. Not latency: we have no measured latency table per format, so an objective "
            "using one would optimize invented data.", 15, SLATE, space_after=0)
table(s, 6.75, 2.0, 5.95, 1.9, ["Point", "Size", "Kodak", "CLIC"],
      [["1% champion", "21.677 MB", "0.67%", "0.85%"],
       ["1.8% variant", "18.664 MB", "1.27%", "1.60%"],
       ["INT8 tier + QAT", "14.401 MB", "1.85%", "2.87%"]],
      widths=[0.36, 0.28, 0.18, 0.18], size=13, highlight=(0, 1))
frame = textbox(s, 6.75, 4.15, 5.95, 2.2)
para(frame, "The knee is between 1.6 and 1.8 percent: 2.5 MB falls away in one step.", 15, SLATE,
     first=True, space_after=8)
para(frame, "The 1.8 percent assignment beats the 2 percent one we first chose. It is 0.275 MB larger and "
            "better on all three sets.", 15, SLATE, space_after=8)
para(frame, "Solved with HiGHS. A greedy baseline matches it within a few percent.", 14, MUTED,
     space_after=0)
footer(s, "results/final/SUBMISSION_POINTS.md | results/night/ilp_fine_sweep.json")
notes(s, "The shared-variable constraint is deployability, not convenience. The knee matters: an earlier "
         "choice at 2 percent looked good until a finer sweep found 1.8 percent, which is both smaller "
         "and better.")

# ================================================================== 6 39x
s = slide()
title(s, "The comparison that decides it", "uniform INT8 is smaller and 39 times worse")
chart_data = CategoryChartData()
chart_data.categories = ["uniform INT8\n13.9 MB", "1.8% variant\n18.7 MB", "1% champion\n21.7 MB"]
chart_data.add_series("Worst-setting penalty (%)", (26.184, 1.268, 0.672))
chart = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.62), Inches(1.95),
                           Inches(7.7), Inches(4.5), chart_data).chart
chart.has_title = True
chart.chart_title.text_frame.text = "Worst-setting penalty on Kodak. Lower is better."
chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
chart.has_legend = False
plot = chart.plots[0]
plot.gap_width = 60
plot.has_data_labels = True
plot.data_labels.show_value = True
plot.data_labels.number_format = '0.00"%"'
plot.data_labels.number_format_is_linked = False
plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
plot.data_labels.font.size = Pt(14)
plot.series[0].format.fill.solid()
plot.series[0].format.fill.fore_color.rgb = NAVY
for index, color in enumerate((RED, MUTED, MINT)):
    point = plot.series[0].points[index]
    point.format.fill.solid()
    point.format.fill.fore_color.rgb = color
chart.value_axis.has_major_gridlines = True
chart.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xE4, 0xEA, 0xF2)
chart.value_axis.tick_labels.font.size = Pt(11)
chart.category_axis.tick_labels.font.size = Pt(11)
frame = textbox(s, 8.6, 2.05, 4.1, 4.4)
para(frame, "26.18%", 40, RED, bold=True, first=True, font=HEAD, space_after=2)
para(frame, "uniform INT8 at 13.9 MB", 14, SLATE, space_after=16)
para(frame, "0.67%", 40, MINT, bold=True, font=HEAD, space_after=2)
para(frame, "our champion at 21.7 MB", 14, SLATE, space_after=16)
para(frame, "Uniform INT8 also varies 2.9 times across datasets. The mixed model varies 1.4 times. A "
            "limit on the total gives quality and stability.", 15, NAVY, bold=True, space_after=0)
footer(s, "results/mixed/ | 7 configurations, 3 datasets")
notes(s, "This is the slide that justifies the project. Uniform quantization produces a smaller file and "
         "loses 39 times more quality on Kodak, 67 times on CLIC. The problem is granularity, not "
         "quantization.")

# ================================================================== 7 4-bit and QAT setup
s = slide()
title(s, "4-bit failed. That is why we fine-tuned.", "Group scaling helped by 5 to 22 times. Not enough.")
table(s, 0.62, 2.0, 6.9, 2.4,
      ["Part", "INT8 cost/MB", "INT4 cost/MB"],
      [["E3", "0.14", "31.4"], ["E2", "0.17", "26.1"], ["E4", "0.24", "31.2"],
       ["E1", "0.46", "157.5"], ["D5", "1712", "not viable"]],
      widths=[0.28, 0.36, 0.36], size=13)
frame = textbox(s, 0.62, 4.55, 6.9, 2.0)
para(frame, "Cost per MB is the decision number: penalty points paid for one megabyte saved.",
     15, SLATE, first=True, space_after=8)
para(frame, "Group-32 scaling cut INT4 weight error by 40 percent. The cost stays about 100 times that of "
            "the 8-bit step.", 15, SLATE, space_after=8)
para(frame, "So 4-bit is not worth shipping alone. Can fine-tuning pay for it?", 16, NAVY, bold=True,
     space_after=0)
card(s, 7.85, 2.0, 4.85, 1.5, "Constrained, not penalised", "Hold bitrate at its old value. Minimise "
     "distortion. A fixed penalty drifts in both directions.", accent=NAVY, big_size=22,
     anchor=MSO_ANCHOR.TOP)
card(s, 7.85, 3.65, 4.85, 1.5, "Entropy stays frozen", "The bundle holds integer CDF tables. Move the "
     "weights and encoder and decoder disagree.", accent=NAVY, big_size=22, anchor=MSO_ANCHOR.TOP)
card(s, 7.85, 5.3, 4.85, 1.35, "12,000 steps", "2.2 percent of the pretraining steps. A repair, not a new "
     "training.", accent=NAVY, big_size=22, anchor=MSO_ANCHOR.TOP)
footer(s, "results/qat/WHY_THESE_COMPONENTS.md | scripts/qat_run.py")
notes(s, "Two ideas on one slide. Left: 4-bit is expensive per byte, so it needs help. Right: the three "
         "design choices that let fine-tuning help without breaking the bitstream.")

# ================================================================== 8 QAT results
s = slide()
title(s, "QAT repairs what the choice leaves", "61 to 98 percent. And zero where nothing was left.")
table(s, 0.62, 2.0, 8.1, 2.3,
      ["Tier", "Size", "Before", "2k", "12k", "Removed"],
      [["INT8 tier", "14.401 MB", "8.61%", "8.61%", "3.36%", "61.0%"],
       ["INT4, 2 blocks", "13.572 MB", "119.18%", "25.27%", "20.53%", "82.8%"],
       ["INT4, 4 blocks", "11.369 MB", "1771.71%", "84.22%", "41.84%", "97.6%"],
       ["ILP 2% assignment", "18.389 MB", "1.936%", "not run", "1.936%", "0.0%"]],
      widths=[0.26, 0.18, 0.15, 0.13, 0.13, 0.15], size=13)
card(s, 8.95, 2.0, 3.75, 1.1, "0.0%", "gain on the assignment the solver optimized", accent=RED, big_size=30)
card(s, 8.95, 3.2, 3.75, 1.1, "61-98%", "of damage removed on hand-chosen tiers", big_size=30)
frame = textbox(s, 0.62, 4.6, 12.1, 1.9)
para(frame, "Fine-tuning replaces a good allocation. It does not extend it.", 18, NAVY, bold=True,
     first=True, space_after=8)
para(frame, "At 2,000 steps the INT8 tier gained nothing. We first concluded that QAT only helps severe "
            "damage. At 12,000 steps it removed 61 percent. The first result was a step-budget effect.",
     15, MUTED, space_after=8)
para(frame, "This is our most useful finding, not a failure: the repair is as large as the damage that the "
            "choice left behind.", 16, SLATE, space_after=0)
footer(s, "results/final/qat_attempt/summary.csv | results/night/qat_int8_12k/summary.csv")
notes(s, "Two-sided result. QAT is a repair mechanism, and the size of the repair is set by how badly the "
         "assignment was chosen. On our optimized model it did nothing, which is the correct answer. State "
         "the correction honestly: our first reading was wrong.")

# ================================================================== 9 latency
s = slide()
title(s, "We did not get a speedup. Here is why.", "Quantization accelerates the part that is not the bottleneck.")
frame = textbox(s, 0.62, 1.95, 7.6, 2.4)
para(frame, "A decode takes 88 ms.", 20, NAVY, bold=True, first=True, font=HEAD, space_after=8)
para(frame, "Entropy coding takes 80 ms. The network takes 8 ms.", 17, SLATE, space_after=6)
para(frame, "Convolutions are 9 percent of a decode and 27 percent of a round trip.", 17, SLATE,
     space_after=6)
para(frame, "Quantization cannot fix the part that dominates.", 17, NAVY, bold=True, space_after=0)
card(s, 8.45, 1.95, 4.25, 1.15, "+4.79%", "latency for the mixed model. 95 percent CI +1.27 to +9.80.",
     accent=RED, big_size=30)
card(s, 8.45, 3.2, 4.25, 1.15, "-0.39%", "energy change. The interval includes zero.", accent=MUTED,
     big_size=30)
frame = textbox(s, 0.62, 4.65, 12.1, 1.9)
para(frame, "We make no speed claim for FP8, INT4 or FP4. These formats run as simulated quantization.",
     16, SLATE, first=True, space_after=8)
para(frame, "Only FP16 and BF16 have real kernels on this GPU. They save energy, not time.", 16, SLATE,
     space_after=8)
para(frame, "A real integer speedup needs integer kernels. That is an engineering task, not a policy "
            "choice.", 16, NAVY, bold=True, space_after=0)
footer(s, "results/metrics/ | paired interleaved benchmark, 50 pairs per cell")
notes(s, "The honest negative. Quantization makes convolutions cheaper, and convolutions are 9 percent of a "
         "decode. Reporting a speedup here would have been wrong.")

# ================================================================== 10 conclusion
s = dark_slide()
frame = textbox(s, 0.9, 0.85, 11.5, 0.8)
para(frame, "What we would ship, and what would change it", 34, ICE, bold=True, first=True, font=HEAD,
     space_after=0)
card(s, 0.9, 1.85, 6.0, 1.35, "21.677 MB", "champion. 60.4 percent smaller, 0.94 percent loss.", big_size=32)
card(s, 0.9, 3.3, 6.0, 1.35, "18.664 MB", "size-first variant. 65.9 percent smaller, 1.60 percent loss.",
     big_size=32)
frame = textbox(s, 0.9, 4.85, 6.0, 1.3)
para(frame, "Do not use uniform INT8 at 13.9 MB. It is smaller and 39 times worse.", 15, RED, bold=True,
     first=True, space_after=6)
para(frame, "Do not use 4-bit without fine-tuning.", 15, RED, bold=True, space_after=0)
frame = textbox(s, 7.2, 1.85, 5.5, 4.3)
para(frame, "Limits we state first", 20, MINT, bold=True, first=True, font=HEAD, space_after=8)
for text in ["Checkpoint selection under-reports the QAT gain. The probe said no change; the full test "
             "said 8.61 to 3.36 percent.",
             "Weights only. No activation quantization, so W8A8 is not tested.",
             "4-bit is simulated, so its speed is not measured.",
             "One entropy lane. The largest unused latency lever.",
             "QAT uses one seed and 100 images. The direction holds. The exact numbers are single-run."]:
    para(frame, text, 13, ICE, space_after=6)
frame = textbox(s, 0.9, 6.35, 11.5, 0.7)
para(frame, "Granularity is the lever. Fine-tuning repairs what a choice leaves. An optimal choice leaves "
            "nothing.", 17, MINT, bold=True, first=True, space_after=0)
notes(s, "Close on the trade-off and the limits. If asked what we would do next: fix checkpoint selection, "
         "then test real integer kernels, then multi-lane entropy coding.")

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(json.dumps({"deck": str(OUT), "slides": len(prs.slides._sldIdLst)}))
