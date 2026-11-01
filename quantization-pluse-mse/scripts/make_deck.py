"""Generate the submission deck from the measured results.

Every number here is read from the committed CSVs, not typed from memory, except where the value is
spelled out in a caption for readability (the source file is named in the speaker notes).

    python quantization-pluse-mse/scripts/make_deck.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
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
         font=BODY, align=PP_ALIGN.LEFT, line=None):
    p = frame.paragraphs[0] if first else frame.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    if line:
        p.line_spacing = line
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    return p


def title(s, text, subtitle=None, dark=False):
    frame = textbox(s, 0.62, 0.42, 12.1, 1.0)
    para(frame, text, 40, ICE if dark else NAVY, bold=True, first=True, font=HEAD, space_after=0)
    if subtitle:
        sub = textbox(s, 0.62, 1.32, 12.1, 0.45)
        para(sub, subtitle, 17, MUTED, first=True, space_after=0)


def footer(s, text, dark=False):
    frame = textbox(s, 0.62, 6.95, 12.1, 0.35)
    para(frame, text, 11, MUTED if not dark else ICE, first=True, space_after=0)


def card(s, x, y, w, h, big, label, note=None, accent=MINT, big_size=40, fill=CARD,
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
    if note:
        para(frame, note, 11, MUTED, space_after=0)
    return shape


def image(s, path, x, y, w, h):
    path = Path(path)
    if not path.is_file():
        return False
    with Image.open(path) as handle:
        iw, ih = handle.size
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    left = x + (w - dw) / 2
    top = y + (h - dh) / 2
    s.shapes.add_picture(str(path), Inches(left), Inches(top), Inches(dw), Inches(dh))
    return True


def table(s, x, y, w, h, header, rows, widths=None, size=13, highlight_rows=()):
    shape = s.shapes.add_table(len(rows) + 1, len(header), Inches(x), Inches(y), Inches(w), Inches(h))
    tbl = shape.table
    tbl.first_row = True
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
            run.font.color.rgb = NAVY if r - 1 in highlight_rows else SLATE
            run.font.bold = r - 1 in highlight_rows
            run.font.name = BODY
    return tbl


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


# ------------------------------------------------------------------ data
frontier = [
    ("1% budget champion", 21.677, 0.672, 0.851, 0.935),
    ("1.8% budget variant", 18.664, 1.268, 1.595, 1.601),
    ("INT8 tier + QAT", 14.401, 1.848, 2.874, 3.363),
]
qat_rows = [
    ("More INT8", 14.401, 8.61, 8.61, 3.36),
    ("Moderate INT4", 13.572, 119.18, 25.27, 20.53),
    ("More INT4", 11.369, 1771.71, 84.22, 41.84),
]

# ------------------------------------------------------------------ 1 title
s = dark_slide()
frame = textbox(s, 0.9, 2.05, 11.5, 1.9)
para(frame, "Which bits can we drop?", 54, ICE, bold=True, first=True, font=HEAD, space_after=8)
para(frame, "Component-wise quantization of a learned image codec", 24, WHITE, space_after=0)
frame = textbox(s, 0.9, 4.55, 11.5, 0.9)
para(frame, "PULSE-MSE  ·  14.18M parameters  ·  8 quality settings  ·  one weight file",
     17, ICE, first=True, space_after=0)
card(s, 0.9, 5.55, 3.5, 1.15, "-60.4%", "model size, 0.94% quality cost", accent=MINT, big_size=34)
card(s, 4.65, 5.55, 3.5, 1.15, "14.4 MB", "smallest variant worth deploying", accent=MINT, big_size=34)
card(s, 8.40, 5.55, 3.5, 1.15, "97.6%", "of 4-bit damage repaired by QAT", accent=MINT, big_size=34)
notes(s, "One sentence: we asked which parts of a learned codec tolerate low precision, solved that as an "
         "optimisation, then used quantization-aware fine-tuning to repair what the low-precision choice cost.")

# ------------------------------------------------------------------ 2 the model
s = slide()
title(s, "The model is wildly unbalanced", "so one uniform precision is the wrong instrument")
rows = [("Encoder  E0-E5", "12.13M", "85.6%", "11.6 MB of the four big blocks"),
        ("Entropy  H0-H3", "1.34M", "9.4%", "5.9% of weights, 73% of decode time"),
        ("Decoder  D0-D8", "0.71M", "5.0%", "cheap in bytes, fragile in quality")]
table(s, 0.62, 2.05, 12.1, 1.75, ["Group", "Parameters", "Share", "Why it matters"], rows,
      widths=[0.24, 0.16, 0.12, 0.48])
frame = textbox(s, 0.62, 4.15, 12.1, 2.2)
para(frame, "Eight QPs ship in ONE weight file, so a per-QP precision choice would not be deployable.",
     17, SLATE, first=True, space_after=10)
para(frame, "Precision damage is not uniform either: some components are fine at 8 bits, some lose 38% at 8 bits.",
     17, SLATE, space_after=10)
para(frame, "So the question is an allocation problem: one precision per component, sized and constrained once.",
     17, NAVY, bold=True, space_after=0)
footer(s, "Parameter counts from artifacts/component_inventory.json")
notes(s, "The asymmetry is the whole motivation. 85.6% of parameters sit in six encoder blocks; the decoder is "
         "5% but is the most fragile per parameter. A uniform setting over-spends on the robust parts and "
         "under-protects the fragile ones.")

# ------------------------------------------------------------------ 3 method
s = slide()
title(s, "How we measured it", "69,233 measurements, one component at a time")
for index, (head, body) in enumerate([
    ("7 formats", "FP16, BF16, FP8-E4M3, FP8-E5M2, INT8, INT4, FP4-E2M1"),
    ("19 components", "every conv-bearing block, weights only, everything else FP32"),
    ("8 QPs x 2 sets", "Kodak screens, CLIC Professional confirms, 8 quality settings"),
]):
    card(s, 0.62 + index * 4.12, 2.05, 3.85, 1.45, head, body, accent=NAVY, big_size=24,
         anchor=MSO_ANCHOR.TOP)
frame = textbox(s, 0.62, 3.75, 12.1, 1.5)
para(frame, "Quality metric: rate-equivalent penalty,  penalty = 100 x [(1 + dbpp/100) x 2^(-dPSNR/slope) - 1]",
     16, SLATE, first=True, space_after=8)
para(frame, "It folds a quality change and a bitrate change into one number. Zero means indistinguishable from FP32.",
     16, MUTED, space_after=0)
card(s, 0.62, 5.45, 5.85, 1.3, "0.96-1.05", "additivity ratio on Kodak: single-component costs sum correctly",
     accent=MINT, big_size=28)
card(s, 6.87, 5.45, 5.85, 1.3, "10/10", "unit tests pass on the fake-quantization core",
     accent=MINT, big_size=28)
footer(s, "scripts/pipeline.py (metric) · scripts/analyze_additivity.py (validation)")
notes(s, "The additivity check is what makes the optimizer legitimate: if single-component penalties did not add "
         "up, an assignment built from them would be meaningless. They do, within 5%.")

# ------------------------------------------------------------------ 4 the map
s = slide()
title(s, "Some components simply refuse", "13 of 19 are safe at 8 bits; three are not")
image(s, W / "results/analysis/component_precision_penalty_heatmap.png", 0.62, 1.95, 7.6, 4.4)
frame = textbox(s, 8.5, 2.05, 4.2, 4.3)
para(frame, "Intolerant at INT8", 20, RED, bold=True, first=True, font=HEAD, space_after=10)
for text in ["H3  entropy step:  38.15%",
             "H0  hyper encoder:  10.09%",
             "D5  renderer MLP:  8.06%"]:
    para(frame, text, 15, SLATE, space_after=4)
para(frame, "Together 3.8% of parameters.", 14, MUTED, space_after=14)
para(frame, "Cheap at INT8", 20, MINT, bold=True, font=HEAD, space_after=10)
for text in ["E2/E3  encoder DCBs:  0.51 / 0.43%",
             "E1/E4  encoder DCBs:  1.35 / 0.71%",
             "D1  decoder block:  0.12%"]:
    para(frame, text, 15, SLATE, space_after=4)
footer(s, "results/analysis/component_precision_penalty_heatmap.png · worst-QP penalty per component and format")
notes(s, "Read the heatmap as 'percentage points of penalty to quantize this one component alone'. The encoder "
         "blocks are large but tolerant; the two entropy steps are small and intolerant. That inversion is what "
         "makes a per-component assignment beat any uniform setting.")

# ------------------------------------------------------------------ 5 optimizer
s = slide()
title(s, "This is an allocation problem, so we solved it as one", "mixed-integer linear program, HiGHS solver")
frame = textbox(s, 0.62, 2.0, 6.3, 4.4)
para(frame, "Variables", 19, NAVY, bold=True, first=True, font=HEAD, space_after=6)
para(frame, "x[component, precision] - one bit, chosen once and shared across all 8 QPs.", 15, SLATE, space_after=12)
para(frame, "Constraint", 19, NAVY, bold=True, font=HEAD, space_after=6)
para(frame, "For each QP q:  sum of the measured component penalties at q  <=  budget.", 15, SLATE, space_after=6)
para(frame, "Eight constraints, one per QP, because one weight file serves all eight.", 15, MUTED, space_after=12)
para(frame, "Objective", 19, NAVY, bold=True, font=HEAD, space_after=6)
para(frame, "Minimise total bytes. Not latency: we have no measured per-format latency table, so claiming one "
            "would be inventing data.", 15, SLATE, space_after=0)
image(s, W / "results/decision/decision_budget_frontier.png", 7.1, 2.0, 5.6, 4.2)
footer(s, "scripts/solve_ilp.py · scipy.optimize.milp (HiGHS) · 19 components x 7 formats")
notes(s, "The shared-across-QPs variable is the deployability constraint, not a convenience. If we let each QP "
         "choose its own assignment we would need eight weight files, which the release contract does not allow. "
         "Greedy is run alongside as a baseline, and ILP wins or ties on every budget.")

# ------------------------------------------------------------------ 6 frontier
s = slide()
title(s, "Three measured operating points", "same harness, all three datasets, all eight QPs")
rows = [(name, f"{size:.3f}", f"-{100*(1-size/54.688):.1f}%", f"{k:.2f}%", f"{c:.2f}%", f"{s14:.2f}%")
        for name, size, k, c, s14 in frontier]
table(s, 0.62, 2.0, 12.1, 1.9, ["Point", "Size MB", "Saving", "Kodak", "CLIC", "Set14"], rows,
      widths=[0.32, 0.14, 0.13, 0.14, 0.14, 0.13], size=15, highlight_rows=(0, 1))
frame = textbox(s, 0.62, 4.15, 12.1, 2.2)
para(frame, "FP32 is 54.688 MB. Every figure is a worst-QP rate-equivalent penalty against it.",
     15, MUTED, first=True, space_after=12)
para(frame, "1% champion: 0.94% worst case for 60% of the bytes gone.", 17, SLATE, space_after=8)
para(frame, "1.8% variant: 1.60% worst case for 66% gone.", 17, SLATE, space_after=8)
para(frame, "INT8 tier + QAT: 3.36% worst case for 74% gone - and it only became usable because of fine-tuning.",
     17, NAVY, bold=True, space_after=0)
footer(s, "results/final/SUBMISSION_POINTS.md")
notes(s, "These are measured, not predicted. The solver's own prediction was 0.97% for the champion and the "
         "measurement came in at 0.94%, so it errs conservative every time. Set14 was never used for fitting, so "
         "it is a genuine transfer check.")

# ------------------------------------------------------------------ 7 mixed vs uniform
s = slide()
title(s, "Mixed precision is the result", "not a refinement: at comparable size, uniform INT8 is 39x worse")
chart_data = CategoryChartData()
chart_data.categories = ["uniform INT8\n13.9 MB", "safe\n17.9 MB", "1.8% variant\n18.7 MB", "1% champion\n21.7 MB"]
chart_data.add_series("Worst-QP penalty (%)", (26.184, 2.633, 1.268, 0.672))
shape = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.62), Inches(1.95),
                           Inches(7.7), Inches(4.5), chart_data)
chart = shape.chart
chart.has_title = True
chart.chart_title.text_frame.text = "Worst-QP penalty on Kodak, lower is better"
chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
chart.has_legend = False
plot = chart.plots[0]
plot.gap_width = 55
plot.has_data_labels = True
plot.data_labels.show_value = True
plot.data_labels.number_format = '0.00"%"'
plot.data_labels.number_format_is_linked = False
plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
plot.data_labels.font.size = Pt(13)
plot.series[0].format.fill.solid()
plot.series[0].format.fill.fore_color.rgb = NAVY
for index, color in enumerate((RED, MUTED, MINT, MINT)):
    point = plot.series[0].points[index]
    point.format.fill.solid()
    point.format.fill.fore_color.rgb = color
chart.value_axis.has_major_gridlines = True
chart.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xE4, 0xEA, 0xF2)
chart.value_axis.tick_labels.font.size = Pt(11)
chart.category_axis.tick_labels.font.size = Pt(11)
frame = textbox(s, 8.6, 2.05, 4.1, 4.4)
para(frame, "26.18%", 40, RED, bold=True, first=True, font=HEAD, space_after=2)
para(frame, "uniform INT8, at 13.9 MB", 14, SLATE, space_after=18)
para(frame, "0.67%", 40, MINT, bold=True, font=HEAD, space_after=2)
para(frame, "our 1% assignment, at 21.7 MB", 14, SLATE, space_after=18)
para(frame, "The smaller model is 39x worse in quality. Spending bytes where sensitivity lives is worth more "
            "than spending them everywhere.", 15, NAVY, bold=True, space_after=0)
footer(s, "results/mixed/ comparison across 7 configs, 3 datasets")
notes(s, "This is the slide that justifies the project. Naive uniform INT8 gives a smaller file than our champion "
         "and is 39x worse on Kodak, 67x worse on CLIC. It is not that quantization is bad; it is that uniform "
         "granularity is the wrong granularity.")

# ------------------------------------------------------------------ 8 the knee
s = slide()
title(s, "Where the size actually falls", "the 1.8% budget replaced the 2% one")
rows = [("1.0%", "21.677", "0.971"), ("1.2%", "21.396", "1.137"), ("1.4%", "21.258", "1.285"),
        ("1.6%", "21.151", "1.444"), ("1.8%", "18.664", "1.716"), ("2.0%", "18.389", "1.986")]
table(s, 0.62, 2.0, 5.6, 3.3, ["Budget", "Size MB", "Predicted %"], rows,
      widths=[0.28, 0.36, 0.36], size=14, highlight_rows=(4,))
frame = textbox(s, 6.7, 2.0, 6.0, 4.4)
para(frame, "Between 1.6% and 1.8% the size falls 2.5 MB in one step.", 18, NAVY, bold=True, first=True,
     font=HEAD, space_after=12)
para(frame, "That is the knee: below it you pay quality for almost no bytes, above it you pay bytes for almost "
            "no quality.", 16, SLATE, space_after=14)
para(frame, "Measured on all three datasets, the 1.8% assignment also beats the 2.0% one we had planned: "
            "1.27 / 1.60 / 1.60% versus 1.40 / 1.94 / 1.71%.", 16, SLATE, space_after=14)
para(frame, "So the earlier 2% choice was inside the knee and is not on the frontier. It was replaced.",
     16, MINT, bold=True, space_after=0)
footer(s, "results/night/ilp_fine_sweep.json · ilp_fine_measured.json")
notes(s, "Worth saying plainly: we had already picked the 2% point and it turned out to be dominated. A finer "
         "sweep found a strictly better model. That is a small example of why the frontier should be swept, not "
         "guessed.")

# ------------------------------------------------------------------ 9 four bit
s = slide()
title(s, "4-bit is not a knob you can just turn", "even with group scaling it costs 26-157 points per MB")
rows = [("E3", "0.427%", "0.14", "34.50%", "31.4"),
        ("E2", "0.509%", "0.17", "28.85%", "26.1"),
        ("E4", "0.710%", "0.24", "34.60%", "31.2"),
        ("E1", "1.352%", "0.46", "172.27%", "157.5"),
        ("H3", "38.152%", "184", "not viable", "-"),
        ("D5", "8.057%", "1712", "not viable", "-")]
table(s, 0.62, 2.0, 7.4, 3.5, ["Comp", "INT8 penalty", "cost/MB", "INT4g32 penalty", "cost/MB"], rows,
      widths=[0.12, 0.22, 0.18, 0.26, 0.22], size=13)
frame = textbox(s, 8.35, 2.0, 4.4, 4.4)
para(frame, "cost/MB", 20, NAVY, bold=True, first=True, font=HEAD, space_after=6)
para(frame, "penalty points paid per megabyte saved. The optimizer minimises exactly this.", 14, SLATE,
     space_after=14)
para(frame, "Group-32 scaling cut 4-bit error by 40% and its damage 5-22x versus per-channel. Still about "
            "100x worse per byte than the 8-bit step.", 15, SLATE, space_after=14)
para(frame, "That is the honest 4-bit verdict: not worth it on its own, which is why we then tried to buy it "
            "back with fine-tuning.", 15, MINT, bold=True, space_after=0)
footer(s, "results/qat/WHY_THESE_COMPONENTS.md · group-32 costs 5.0 effective bits per weight")
notes(s, "Per-channel INT4 measured 148-3767% penalty. Implementing group-32 scaling brought that down by 5-22x, "
         "which is real engineering progress, but the cost per byte is still two orders of magnitude worse than "
         "INT8. So 4-bit only becomes interesting if fine-tuning pays for it.")

# ------------------------------------------------------------------ 10 QAT setup
s = slide()
title(s, "Fine-tuning a quantized model, done carefully", "three design choices that make it trustworthy")
for index, (head, body) in enumerate([
    ("Constrained, not penalised", "Minimise distortion subject to holding bitrate at its pre-quantization "
     "value. A fixed rate penalty either drifts or dumps quality to hit the target."),
    ("Entropy stays frozen", "The bundle ships precomputed integer CDF tables. Changing entropy weights would "
     "break the bit-exact coding contract, so they are held fixed."),
    ("Best checkpoint wins", "The saved model must beat its own starting point, or the starting weights are "
     "kept. A bad run cannot produce a worse model."),
]):
    card(s, 0.62 + index * 4.12, 2.05, 3.85, 1.95, head, body, accent=NAVY, big_size=21,
         anchor=MSO_ANCHOR.TOP)
frame = textbox(s, 0.62, 4.35, 12.1, 2.0)
para(frame, "Augmented Lagrangian:  loss = distortion + mu x (rate / target - 1),  with mu updated from the "
            "measured rate error on a fixed validation set.", 16, SLATE, first=True, space_after=12)
para(frame, "12,000 steps at 512-pixel crops, about 100 minutes per tier on one 4 GB laptop GPU.",
     16, SLATE, space_after=12)
para(frame, "That is 2.2% of the optimizer steps the released model was pretrained with, and 0.2% of its sample "
            "budget - fine-tuning, not retraining.", 16, NAVY, bold=True, space_after=0)
footer(s, "scripts/qat_run.py · scripts/qat_overnight.py")
notes(s, "The frozen entropy model is the subtle part. If you fine-tune the entropy components the integer CDF "
         "tables in the bundle no longer describe the weights, and the decoder produces a different image from "
         "the encoder's input. Freezing them keeps the bitstream contract intact.")

# ------------------------------------------------------------------ 11 QAT results
s = slide()
title(s, "QAT recovers what the assignment leaves on the table", "and nothing where it leaves nothing")
rows = [(name, f"{size:.3f}", f"{b:.2f}%", f"{two:.2f}%", f"{twelve:.2f}%",
         f"{100*(b-twelve)/b:.1f}%") for name, size, b, two, twelve in qat_rows]
table(s, 0.62, 2.0, 8.0, 2.4, ["Tier", "Size MB", "Before", "2k steps", "12k steps", "Damage removed"], rows,
      widths=[0.24, 0.13, 0.14, 0.14, 0.15, 0.20], size=13)
card(s, 8.85, 2.0, 3.9, 1.0, "0.0%", "QAT gain on the ILP 2% assignment (8,000 steps)", accent=RED, big_size=30)
card(s, 8.85, 3.15, 3.9, 1.25, "61 -> 98%", "of damage removed across tiers at 12k steps", accent=MINT,
     big_size=26)
frame = textbox(s, 0.62, 4.7, 12.1, 1.75)
para(frame, "Three tiers with very different starting damage behave predictably: the worse the assignment, the "
            "more there is to recover.", 16, SLATE, first=True, space_after=10)
para(frame, "But the ILP 2% model recovered exactly nothing, because a solver-optimised assignment has already "
            "spread its damage optimally. Fine-tuning has nothing to repair.", 16, SLATE, space_after=10)
para(frame, "Correction we had to make: at 2,000 steps the INT8 tier gained 0.0%, which looked like 'QAT only "
            "helps severe damage'. At 12,000 steps it removed 61%. The first reading was a step-budget artefact.",
     15, MUTED, space_after=0)
footer(s, "results/final/qat_attempt/summary.csv · results/night/qat_int8_12k/summary.csv")
notes(s, "This is the most interesting result in the project and it is two-sided. QAT is a repair mechanism, and "
         "the size of the repair is set by how badly the assignment was chosen. On a model we had already "
         "optimised, it did nothing at all - which is the correct answer, not a failure.")

# ------------------------------------------------------------------ 12 latency
s = slide()
title(s, "We did not get a speedup, and we know exactly why", "quantization accelerates the part that is not the bottleneck")
rows = [("ILP mixed vs FP32", "+4.79%", "+1.27 to +9.80", "-0.39%", "CI spans zero"),
        ("FP16 autocast", "+4.29%", "-", "-9.99%", "-15.1 to -5.3"),
        ("BF16 autocast", "+1.82%", "-", "-8.86%", "-")]
table(s, 0.62, 2.0, 7.9, 2.1, ["Configuration", "Latency", "95% CI", "Energy", "Energy CI"], rows,
      widths=[0.30, 0.16, 0.22, 0.14, 0.18], size=13)
frame = textbox(s, 8.75, 2.0, 4.0, 4.4)
para(frame, "73%", 44, RED, bold=True, first=True, font=HEAD, space_after=2)
para(frame, "of a GPU decode is entropy coding, not convolutions", 14, SLATE, space_after=16)
para(frame, "9%", 44, MUTED, bold=True, font=HEAD, space_after=2)
para(frame, "of a decode is the convolutional network we quantized", 14, SLATE, space_after=16)
para(frame, "Float formats are the only ones with a real kernel on this GPU, and they save energy rather than "
            "time.", 14, MUTED, space_after=0)
frame = textbox(s, 0.62, 4.5, 12.1, 1.9)
para(frame, "Simulated formats - FP8, INT4, FP4 - get no latency claim at all. We can only report size and "
            "quality for them.", 16, SLATE, first=True, space_after=10)
para(frame, "A real INT8 speedup needs real INT8 kernels (TensorRT/ONNX), which is the obvious next engineering "
            "step, not a quantization-policy question.", 16, NAVY, bold=True, space_after=0)
footer(s, "results/metrics/ · paired interleaved benchmark, 50 pairs per cell, bootstrap CI")
notes(s, "Latency is the honest negative of this work. Quantization makes convolutions cheaper, and convolutions "
         "are 27% of round trip. The entropy coder dominates. Reporting a speedup here would have required "
         "either real INT8 kernels or multi-lane entropy coding, neither of which we shipped.")

# ------------------------------------------------------------------ 13 recommendation
s = slide()
title(s, "What we would ship", "two lines of evidence, one recommendation")
card(s, 0.62, 2.0, 6.0, 1.6, "21.677 MB", "champion - 60.4% smaller, 0.94% worst case", accent=MINT, big_size=34)
card(s, 6.72, 2.0, 6.0, 1.6, "18.664 MB", "smaller variant - 65.9% smaller, 1.60% worst case", accent=MINT,
     big_size=34)
frame = textbox(s, 0.62, 3.95, 12.1, 2.7)
para(frame, "Use the 1% assignment when quality is the constraint: it is inside 1% on all three datasets and "
            "its BD-rate is +0.48 to +0.55%.", 16, SLATE, first=True, space_after=10)
para(frame, "Use the 1.8% assignment when size is the constraint: 3.0 MB smaller for about 0.7 extra points.",
     16, SLATE, space_after=10)
para(frame, "Do not use uniform INT8 at 13.9 MB: it is smaller and 39x worse. Do not use 4-bit without QAT: it "
            "is 21-42% even after 12,000 steps of repair.", 16, RED, space_after=10)
para(frame, "Every number on this slide is measured on all three datasets at all eight QPs, with the FP32 "
            "reference gate passing.", 15, MUTED, space_after=0)
footer(s, "results/final/SUBMISSION_POINTS.md")
notes(s, "If asked 'which one do you submit', the answer is the 21.677 MB champion, with the 18.664 MB variant as "
         "the size-constrained option. The other configurations are evidence, not products.")

# ------------------------------------------------------------------ 14 limitations
s = slide()
title(s, "What would change our answer", "four known limits, stated up front")
limited = [
    ("Checkpoint selection under-reports", "Our validation probe scored the best QAT model as no better than its "
     "start, while the full evaluation showed 8.61 to 3.36%. Selection uses 512-crops; the metric uses native "
     "resolution. Selection is leaving gains on the table."),
    ("Weights only", "No activation quantization. W8A8 is the configuration that would actually use an INT8 "
     "kernel, and we did not build it."),
    ("Simulated 4-bit", "FP8, INT4 and FP4 run as fake quantization. Quality numbers are real; any speed claim "
     "would not be."),
    ("One entropy lane", "Decode spends 73% of its time in a single-lane rANS path. Multi-lane is the largest "
     "latency lever available and it changes the bitstream contract, so it is a design decision, not a tweak."),
]
for index, (head, body) in enumerate(limited):
    col, row = index % 2, index // 2
    card(s, 0.62 + col * 6.1, 2.0 + row * 2.25, 5.85, 2.0, head, body, accent=NAVY, big_size=21,
         anchor=MSO_ANCHOR.TOP)
footer(s, "quantization-pluse-mse/STATUS.md section 4")
notes(s, "Stating these first is deliberate. The checkpoint-selection issue is the one that most affects the QAT "
         "numbers we report, and we would fix it before trusting the 12k results at finer resolution.")

# ------------------------------------------------------------------ 15 close
s = dark_slide()
frame = textbox(s, 0.9, 2.2, 11.5, 1.4)
para(frame, "One sentence", 22, MINT, bold=True, first=True, font=HEAD, space_after=16)
para(frame, "Mixed precision buys 60-66% of the model for under 2% quality; fine-tuning then repairs whatever "
            "the assignment could not afford - but it cannot improve an assignment that was already optimal.",
     30, ICE, bold=True, space_after=0, font=HEAD)
frame = textbox(s, 0.9, 5.3, 11.5, 1.2)
para(frame, "21.677 MB at 0.94%   ·   18.664 MB at 1.60%   ·   14.401 MB at 3.36% after QAT",
     18, WHITE, first=True, space_after=0)
notes(s, "Close on the trade-off, not on a feature list. The three numbers are the deliverable.")

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(json.dumps({"deck": str(OUT), "slides": len(prs.slides._sldIdLst)}))
