# Submission package

## What is submitted

| File | Role | Size |
|---|---|---|
| **`report.pdf`** | **The graded report.** 2 pages including references (the brief allows 2 pages excluding references). Source in `report.tex`. | 2 pages |
| **`PULSE-quantization-deck.pdf`** | Presentation for the 25-minute slot. | 10 slides |
| `PULSE-quantization-deck.pptx` | Editable deck with speaker notes. | 10 slides |
| `supplementary-notes.pdf` | Extra detail that does not fit in two pages: full tables, method detail, engineering record. Not part of the graded submission. | 5 pages |
| `supplementary-notes.docx` | Same, editable. | 5 pages |

All three documents are generated from committed scripts: `make_supplementary.py`, `make_deck.py`, and
Tectonic for `report.tex`. Every table names the CSV it came from.

## Headline result

54.688 MB FP32 becomes **21.677 MB**, which is **60.4% smaller**, at a **0.94%** worst-setting quality
loss and a **+0.48 to +0.55%** BD-rate across three datasets.

The comparison that decides the argument: **uniform INT8 gives a smaller file (13.904 MB) at 26.18%
penalty**, so the mixed assignment is **39 times better in quality at the same size**.

| Configuration | Size | Saving | Kodak | CLIC | Set14 |
|---|---|---|---|---|---|
| ILP mixed, 1% budget | 21.677 MB | -60.4% | 0.672% | 0.851% | 0.935% |
| ILP mixed, 1.8% budget | 18.664 MB | -65.9% | 1.268% | 1.595% | 1.601% |
| INT8 tier + fine-tuning | 14.401 MB | -73.7% | 1.848% | 2.874% | 3.363% |

## What fine-tuning contributed

Fine-tuning removed **61%, 83% and 98%** of the damage on three hand-chosen assignments. It removed
**exactly 0%** on the assignment the solver had already optimized. Therefore fine-tuning repairs the
cost that an assignment leaves. An optimal assignment leaves none.

## Where the evidence lives

| Claim | File |
|---|---|
| 69,233 measurements | `quantization-pluse-mse/results/_ledger/` (regenerate with the pipeline) |
| Sensitivity map and decision surface | `quantization-pluse-mse/results/analysis/`, `results/decision/` |
| ILP precision maps | `quantization-pluse-mse/results/ilp/` |
| Whole-model evaluation and BD-rate | `quantization-pluse-mse/results/mixed/` |
| Latency, energy, stage profile | `quantization-pluse-mse/results/metrics/` |
| Fine-tuning runs at 2k and 12k steps | `results/final/`, `results/night/` |
| Claim-to-proof map and citations | `quantization-pluse-mse/RESOURCES.md` |
| Full project checkpoint | `quantization-pluse-mse/STATUS.md` |

## Limitations, stated first

1. **Checkpoint selection under-reports the fine-tuning gain.** A 6-crop probe rated the best INT8
   model as equal to its start. The full evaluation shows 8.61% to 3.36%. The saved checkpoints are
   probably not the best available.
2. **Weights only.** Activations stay in FP32, so W8A8 is not tested.
3. **4-bit is simulated.** Quality is measured. Speed is not claimed.
4. **One entropy lane.** The largest unused latency lever, and it changes the bitstream contract.
5. **One seed, one pool.** The direction holds across three tiers and two step counts. The exact
   percentages are single-run values.
6. **The penalty metric is project-defined.** BD-rate is reported alongside it, so no claim rests on
   one metric alone.

## Reproducing

See `RUNBOOK.md` in the repository root. Short version:

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\solve_ilp.py --budget 1.0
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\eval_assignment.py --dataset kodak
.\tools\tectonic\tectonic.exe submission\report.tex
```
