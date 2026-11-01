# Submission package

Everything a grader needs, plus the evidence behind every number.

## Headline result

A learned image codec (PULSE-MSE, 14.18M parameters) is compressed from **54.688 MB FP32** to
**21.677 MB** — a **60.4% reduction** — at a worst-case rate-equivalent quality cost of **0.94%**
and a Bjontegaard delta-rate of **+0.48 to +0.55%** across three datasets.

The decisive comparison is not against FP32 but against the obvious baseline: **uniform INT8 gives a
smaller file (13.904 MB) at 26.18% penalty**, so the mixed assignment is **39x better quality at
comparable size**. Granularity, not precision, is what matters.

## Deliverables

| File | What it is |
|---|---|
| `PULSE-quantization-report.pdf` | The report, 13 pages, 9 tables, 4 figures |
| `PULSE-quantization-report.docx` | Same report, editable |
| `PULSE-quantization-deck.pdf` | The presentation, 15 slides, with speaker notes |
| `PULSE-quantization-deck.pptx` | Same deck, editable |

Both documents are generated from committed scripts (`quantization-pluse-mse/scripts/make_report.py`
and `make_deck.py`) so any number can be traced to the CSV or PNG it came from.

## Three measured operating points

| Point | Size | Saving | Kodak | CLIC | Set14 |
|---|---|---|---|---|---|
| 1% budget champion | 21.677 MB | -60.4% | 0.672% | 0.851% | 0.935% |
| 1.8% budget variant | 18.664 MB | -65.9% | 1.268% | 1.595% | 1.601% |
| INT8 tier + QAT 12k | 14.401 MB | -73.7% | 1.848% | 2.874% | 3.363% |

Every figure is a worst-QP rate-equivalent penalty against the FP32 reference, measured on all three
datasets at all eight quality settings, with the FP32 gate passing. Set14 was never used for fitting.

## What the fine-tuning contributed

Quantization-aware fine-tuning removed **61%, 83% and 98%** of the damage on three hand-chosen
assignments (INT8 and two INT4 tiers), and **exactly 0%** on the assignment the solver had already
optimized. The finding: fine-tuning repairs what an assignment leaves on the table, and an optimal
assignment leaves nothing.

## Where the evidence lives

| Claim | File |
|---|---|
| All 69,233 sweep measurements | `quantization-pluse-mse/results/_ledger/` |
| Sensitivity map and decision surface | `quantization-pluse-mse/results/analysis/`, `results/decision/` |
| ILP assignments and predictions | `quantization-pluse-mse/results/ilp/` |
| Whole-model evaluation, BD-rate | `quantization-pluse-mse/results/mixed/` |
| Latency, energy, stage decomposition | `quantization-pluse-mse/results/metrics/` |
| QAT runs at 2k and 12k steps | `results/final/`, `results/night/` |
| Claim-to-proof map and citations | `quantization-pluse-mse/RESOURCES.md` |
| Full project checkpoint | `quantization-pluse-mse/STATUS.md` |

## Reproducing

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\solve_ilp.py --budget 1.0
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\eval_assignment.py --dataset kodak
python quantization-pluse-mse\scripts\make_report.py     # needs python-docx
python quantization-pluse-mse\scripts\make_deck.py       # needs python-pptx
```

The full sweep takes hours on a 4 GB laptop GPU; assignment evaluation takes minutes. Ten unit tests
cover the quantization core (`python -m unittest discover -s quantization-pluse-mse/tests`).

## Known limitations, stated up front

1. **Checkpoint selection under-reports fine-tuning gains.** The 6-crop validation probe scored the
   best INT8 model as no better than its start (-0.08%) while the full evaluation shows 8.61% to
   3.36%. Saved checkpoints are probably not the best available.
2. **Weights only** — no activation quantization, so W8A8 is not exercised.
3. **4-bit is simulated** (fake quantization): quality numbers are real, speed is not claimed.
4. **Single entropy lane** — the largest unused latency lever, and it changes the bitstream contract.
5. **Single seed, single pool** for fine-tuning — the qualitative finding holds across three tiers
   and two step counts; the exact percentages are single-run values.
6. **The penalty metric is project-defined**; BD-rate is reported alongside so no claim rests on it.
