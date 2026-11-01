"""Generate the condensed supplementary report as DOCX.

This replaces the earlier 13-page document with a five-page version in ASD-STE100 style: short
sentences, active voice, one idea per sentence. The graded document is submission/report.pdf; this
file holds the extra detail that does not fit in two pages.

    python quantization-pluse-mse/scripts/make_supplementary.py
"""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "quantization-pluse-mse"
OUT = ROOT / "submission" / "supplementary-notes.docx"

NAVY = RGBColor(0x00, 0x00, 0x00)
SLATE = RGBColor(0x00, 0x00, 0x00)
MUTED = RGBColor(0x00, 0x00, 0x00)

doc = Document()
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(10.5)
normal.font.color.rgb = SLATE
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.12

for level, size in ((1, 15), (2, 12)):
    style = doc.styles[f"Heading {level}"]
    style.font.name = "Cambria"
    style.font.size = Pt(size)
    style.font.color.rgb = NAVY
    style.font.bold = True
    style.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    style.paragraph_format.space_after = Pt(3)

section = doc.sections[0]
section.page_width = Inches(8.5)
section.page_height = Inches(11)
section.left_margin = section.right_margin = Inches(1.0)
section.top_margin = section.bottom_margin = Inches(0.85)


def h1(text):
    doc.add_heading(text, level=1)


def h2(text):
    doc.add_heading(text, level=2)


def body(text, size=10.5):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    return p


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.font.size = Pt(10.5)
    return p


def shade(cell, hex_color):
    element = OxmlElement("w:shd")
    element.set(qn("w:val"), "clear")
    element.set(qn("w:fill"), hex_color)
    cell._tc.get_or_add_tcPr().append(element)


def table(headers, rows, caption=None, size=9.5):
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, text in enumerate(headers):
        cell = tbl.rows[0].cells[index]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.RIGHT
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(size)
        run.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
        shade(cell, "D9D9D9")
    for r, row in enumerate(rows):
        cells = tbl.add_row().cells
        for index, text in enumerate(row):
            cells[index].text = ""
            p = cells[index].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.RIGHT
            run = p.add_run(str(text))
            run.font.size = Pt(size)
    for row in tbl.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    def rule(row, where):
        for cell in row.cells:
            props = cell._tc.get_or_add_tcPr()
            borders = OxmlElement("w:tcBorders")
            edge = OxmlElement(f"w:{where}")
            edge.set(qn("w:val"), "single")
            edge.set(qn("w:sz"), "8")
            edge.set(qn("w:color"), "000000")
            borders.append(edge)
            props.append(borders)
    rule(tbl.rows[0], "top")
    rule(tbl.rows[0], "bottom")
    rule(tbl.rows[-1], "bottom")
    if caption:
        p = doc.add_paragraph()
        run = p.add_run(caption)
        run.font.size = Pt(8.5)
        run.italic = True
        run.font.color.rgb = MUTED
        p.paragraph_format.space_after = Pt(9)


def code(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(8)


# ------------------------------------------------------------------ title
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("Which bits can be dropped? Supplementary material")
run.font.name = "Cambria"
run.font.size = Pt(20)
run.bold = True
run.font.color.rgb = NAVY

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub.add_run("Detail that does not fit in the two-page report. CS6886, November 2026.")
run.font.size = Pt(10)
run.font.color.rgb = MUTED

# ------------------------------------------------------------------ 1 summary
h1("1  What this document adds")
body("The submitted report has two pages. This document has the method detail, the full result tables "
     "and the engineering record. Read the report first. Use this document to check a number or to "
     "repeat the work.")
body("The project asks one question. Which parts of a learned image codec can use a low precision? The "
     "answer is an assignment: one precision for each part. The cost of each choice is measured, the "
     "assignment is solved as an integer program, and fine-tuning is then tested as a way to recover the "
     "remaining cost.")

# ------------------------------------------------------------------ 2 method
h1("2  Method")
h2("2.1  Measurement")
body("One part is quantized at a time. All other parts and all activations stay in FP32. The complete codec then runs "
     "on each image at each quality setting. The plan covers seven formats, 19 parts, "
     "two datasets and eight settings. That gives 69,233 rows.")
body("Two safeguards make a row valid. First, an FP32 gate must reproduce the reference bitstream and "
     "the reconstruction at all eight settings. Second, a hash check must confirm each weight "
     "restoration. No run failed this check.")

h2("2.2  The quality metric")
body("One number must hold a quality change and a bitrate change together. One metric does it: a "
     "rate-equivalent penalty:")
code("penalty = 100 * [ (1 + d_bpp/100) * 2^(-d_PSNR / slope) - 1 ]")
body("The value is zero when the model is equal to FP32. The optimizer limits the worst setting. The "
     "worst setting is the maximum over the eight settings of the mean over the images. Bjontegaard "
     "delta-rate is also reported as a second view, so that no claim depends on one metric.")

h2("2.3  Additivity")
body("The optimizer adds single-part costs. That step is correct only if the costs add. The condition was "
     "tested. Divide the measured penalty of a uniform INT8 model by the sum of the 19 single-part "
     "penalties. The ratio is 0.96 to 1.05 on Kodak and 1.02 to 1.11 on CLIC.")
body("One warning. Do not add the worst setting of each part. That sum gives 65.3 percent, but the true "
     "value at one setting is 26.7 percent. Add the costs for one setting, then take the maximum.")

h2("2.4  The allocation program")
body("One weight file serves all eight settings. A different assignment for each setting is therefore "
     "not deployable. The model uses one binary variable for each pair of a part and a format. Each "
     "variable is shared by all settings. The model has one quality constraint for each setting.")
body("The objective is the number of bytes. It is not latency. No measured latency table exists for "
     "each format on this GPU, and an objective that used such a table would optimize invented data. "
     "HiGHS solves the program, and a greedy baseline is also solved for comparison.")

h2("2.5  Fine-tuning")
body("Quantized convolutions run with a straight-through gradient. Distortion is then minimized while the "
     "bitrate is held at its pre-quantization value. A fixed rate penalty failed in both directions. A "
     "weak penalty lets the model move down the rate-distortion curve. A strong penalty destroys "
     "quality to obey the rate target. An augmented Lagrangian is used instead. The multiplier comes "
     "from the measured rate error on a fixed validation set.")
body("The entropy parts are frozen. The bundle contains integer CDF tables. If the entropy weights move, "
     "those tables no longer describe the weights, and the encoder and the decoder disagree. The saved "
     "checkpoint must be better than its own start. If it is not, the start weights are kept. A bad run "
     "therefore cannot ship a worse model.")
body("Training uses 100 DIV2K images that are disjoint from all test sets. The run uses 12,000 steps on "
     "512-pixel crops. That takes about 100 minutes for one tier on one 4 GB laptop GPU. This work uses "
     "2.2 percent of the optimizer steps and 0.2 percent of the sample budget of the pretraining. Thus "
     "it is a repair operation, not a new training.")

# ------------------------------------------------------------------ 3 results
h1("3  Results")
h2("3.1  The frontier")
table(["Point", "Size", "Saving", "Kodak", "CLIC", "Set14"],
      [["1% budget champion", "21.677 MB", "-60.4%", "0.672%", "0.851%", "0.935%"],
       ["1.8% budget variant", "18.664 MB", "-65.9%", "1.268%", "1.595%", "1.601%"],
       ["INT8 tier + QAT", "14.401 MB", "-73.7%", "1.848%", "2.874%", "3.363%"],
       ["uniform INT8", "13.904 MB", "-74.6%", "26.184%", "56.701%", "19.374%"],
       ["safe, no fine-tuning", "17.904 MB", "-67.3%", "2.633%", "4.301%", "2.810%"],
       ["greedy mixed", "24.565 MB", "-55.1%", "0.386%", "0.358%", "0.393%"],
       ["uniform FP16", "27.416 MB", "-49.9%", "0.038%", "0.047%", "0.056%"]],
      caption="Table 1. Measured configurations. Values are worst-setting rate-equivalent penalty "
              "against a 54.688 MB FP32 baseline. Source: results/mixed/, results/final/.")
body("The champion predicted 0.9715 percent and measured 0.94 percent. The prediction is conservative "
     "on each dataset. Set14 was not used for fitting, so the result there is a transfer test. It "
     "holds.")

h2("3.2  The knee")
table(["Budget", "Size", "Predicted penalty"],
      [["1.0%", "21.677 MB", "0.9715%"], ["1.2%", "21.396 MB", "1.1374%"],
       ["1.4%", "21.258 MB", "1.2851%"], ["1.6%", "21.151 MB", "1.4437%"],
       ["1.8%", "18.664 MB", "1.7162%"], ["2.0%", "18.389 MB", "1.9864%"]],
      caption="Table 2. A fine budget sweep. Between 1.6 and 1.8 percent the size falls 2.5 MB in one "
              "step. Source: results/night/ilp_fine_sweep.json.")
body("The 1.8 percent assignment beats the 2 percent assignment chosen first. It is 0.275 MB "
     "larger and better on all three datasets. The 2 percent point was replaced. This is why a frontier "
     "must be swept and not guessed.")

h2("3.3  Fine-tuning results")
table(["Tier", "Size", "Before", "2,000 steps", "12,000 steps", "Damage removed"],
      [["More INT8", "14.401 MB", "8.61%", "8.61%", "3.36%", "61.0%"],
       ["Moderate INT4", "13.572 MB", "119.18%", "25.27%", "20.53%", "82.8%"],
       ["More INT4", "11.369 MB", "1771.71%", "84.22%", "41.84%", "97.6%"],
       ["ILP 2% assignment", "18.389 MB", "1.936%", "not run", "1.936%", "0.0%"]],
      caption="Table 3. Fine-tuning outcome. The ILP row used 8,000 steps. Source: "
              "results/final/qat_attempt/summary.csv, results/night/qat_int8_12k/summary.csv.")
body("QAT removes most of the damage on assignments that a person chose. It removes no damage on the "
     "assignment that the solver optimized. Therefore fine-tuning replaces a good allocation. It does "
     "not extend it. The assignment decides the size of the repair.")

h2("3.4  Group scaling for 4-bit")
table(["Part", "INT8 penalty", "cost/MB", "int4g32 penalty", "cost/MB"],
      [["E3", "0.427%", "0.14", "34.50%", "31.4"],
       ["E2", "0.509%", "0.17", "28.85%", "26.1"],
       ["E4", "0.710%", "0.24", "34.60%", "31.2"],
       ["E1", "1.352%", "0.46", "172.27%", "157.5"],
       ["H3", "38.152%", "184", "not viable", "-"],
       ["D5", "8.057%", "1712", "not viable", "-"]],
      caption="Table 4. Cost per megabyte saved. Group-32 scaling cut INT4 weight error by 40 percent "
              "and damage by 5 to 22 times. Source: results/qat/WHY_THESE_COMPONENTS.md.")
body("Per-output-channel INT4 measured 148 to 3767 percent penalty. Group scaling made it much better. "
     "It is still about 100 times the cost of the 8-bit step for each megabyte saved. So 4-bit is not "
     "worth shipping without fine-tuning.")

h2("3.5  Latency and energy")
body("No configuration was faster than FP32. Paired interleaved benchmarking was used with 50 pairs for "
     "each cell. The mixed model is 4.79 percent slower, with a 95 percent confidence interval of "
     "+1.27 to +9.80. Its energy change is -0.39 percent, and the interval includes zero.")
body("The cause is arithmetic. A decode takes 88 ms. Entropy coding takes 80 ms of that time. The "
     "convolutional network takes 8 ms. Convolutions are 9 percent of a decode and 27 percent of a "
     "round trip. Quantization makes convolutions cheaper, so it cannot fix the bottleneck.")
body("No speed claim is made for FP8, INT4 or FP4. These formats run as simulated quantization. Only "
     "FP16 and BF16 use real kernels on this GPU. They save energy, not time.")

# ------------------------------------------------------------------ 4 record
h1("4  Engineering record")
body("The pipeline did not work on the first attempt. Three failures did not show an error message. They "
     "record them because each one is a general trap.")
bullet("Checkpoint selector floored at zero. The selector accumulated max(0, penalty). A negative "
       "penalty means an improvement, so the selector could never choose an improved checkpoint. Each "
       "run before the fix saved its step-0 weights and discarded its own gain. This one bug cost most "
       "of one night.")
bullet("Two different bitrates. The training loss sees a surrogate rate of about 0.8. The measured "
       "stream bitrate is about 0.32 for the same model. A comparison of the two drove the Lagrange "
       "multiplier in one direction on a false error.")
bullet("An inference-mode trap. The codec compress helper runs under inference mode. If you call it "
       "between training steps, it caches inference tensors on the entropy model, and the next backward "
       "pass fails. Validation must call the model methods under no-grad instead.")
body("A wrong conclusion was also made and corrected. At 2,000 steps the INT8 tier gained nothing, "
     "and the conclusion was that QAT only helps severe damage. At 12,000 steps the same tier removed 61 "
     "percent. The first result was a step-budget effect, not a property of the model.")

# ------------------------------------------------------------------ 5 limits
h1("5  Limitations")
bullet("Checkpoint selection under-reports the fine-tuning gain. A probe with 6 crops rated the best "
       "INT8 model as equal to its start (-0.08 percent). The full evaluation shows 8.61 to 3.36 "
       "percent. The saved checkpoints are probably not the best available.")
bullet("Weights only. Activations stay in FP32, so W8A8 is not tested. W8A8 is the configuration that "
       "would use an integer kernel.")
bullet("The 4-bit formats are simulated. Their quality numbers are real. Their speed is not modelled.")
bullet("The entropy coder uses one lane. Multi-lane coding is the largest unused latency lever, and it "
       "changes the bitstream contract. It is a design decision, not a tuning step.")
bullet("The fine-tuning results use one seed and 100 images. The direction holds for three tiers and "
       "two step counts. The exact percentages are single-run values.")

# ------------------------------------------------------------------ 6 reproduce
h1("6  How to reproduce")
code(r".\venv\Scripts\python.exe quantization-pluse-mse\scripts\eval_assignment.py "
     r"--model checkpoints/pulse-mse "
     r"--configs quantization-pluse-mse\results\ilp\assignment_ilp.json "
     r"--dataset kodak --output results\check")
body("The command gives 21.677 MB and 0.672 percent on Kodak, inside the FP32 gate. RUNBOOK.md has every "
     "other command, the expected runtime, and a troubleshooting list.")
body("The checkpoint is not tracked in the repository because it is 84 MB. Download the released bundle "
     "from https://huggingface.co/zhaoyangjia/PULSE. Place three files in checkpoints/pulse-mse/. The "
     "file entropy_control_int.pt is mandatory. The loader refuses a bundle without it, because a run "
     "without it would produce a different bitstream than the published model.")
body("Code: https://github.com/harshith0518/CS6886-PULSE-Quantization")

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(str(OUT))
print(json.dumps({"report": str(OUT), "paragraphs": len(doc.paragraphs), "tables": len(doc.tables)}))
