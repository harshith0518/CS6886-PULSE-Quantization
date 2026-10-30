# STATUS - project checkpoint

Last updated 2026-10-30. HEAD `9ad5aa3`, worktree clean, 508 tracked files, 10/10 unit tests passing.

Read this first after any break. It records what is finished, what the numbers are, and what is
still open. `RESOURCES.md` maps every claim to the file that proves it.

## 1. What this project is

Component-wise quantization study of the released `pulse-mse` model from PULSE (Jia et al.,
arXiv:2609.18602). The question: which components tolerate which precision, at which QP, and what
does a per-component precision assignment buy.

## 2. Finished and committed

| Area | State | Where |
|---|---|---|
| Weight-only sweep | complete, 7 formats x 19 components x {kodak, clic} x 8 QPs = 69,233 rows, 0 failures | `results/_ledger/` (gitignored) |
| FP32 reference | 4 datasets x 3 passes, noise floor recorded | `results/_ledger/reference/` |
| Decision surface | 6 figures + decision card + `decision_summary.csv` | `results/decision/` |
| ILP solver | HiGHS, 5 budgets, assignment written | `results/ilp/` |
| Whole-model evaluation | 7 configs x 8 QPs x 3 datasets, FP32 gate passes on all three | `results/mixed/` |
| Additivity test | holds: 0.96-1.05 Kodak, 1.02-1.11 CLIC | `results/mixed/additivity.csv` |
| BD-rate comparison | 3 datasets, per-QP tables, 3 figures | `results/mixed/COMPARISON.md` |
| Latency / energy | paired interleaved benchmark + stage profile | `results/metrics/` |
| Provenance | source segregation with verification tags | `RESOURCES.md` |

## 3. The numbers, in one place

**Model size:** 54.688 MB FP32 to **21.677 MB** for the ILP assignment, **-60.4%**.
Uniform FP16 27.416 MB, greedy mixed 24.565 MB, safe mixed 17.904 MB, uniform INT8 13.904 MB.

**Quality** (worst-QP penalty / BD-rate):

| Config | Kodak | CLIC | Set14 |
|---|---|---|---|
| ILP mixed | 0.672% / +0.477% | 0.851% / +0.552% | 0.935% / +0.486% |
| greedy mixed | 0.386% / +0.299% | 0.358% / +0.225% | 0.393% / +0.278% |
| uniform INT8 | 26.184% / +10.2% | 56.701% / +18.7% | 19.374% / +9.4% |

The ILP model was fitted on Kodak and CLIC only; Set14 is a genuine transfer test and it holds.

**Latency and energy** (paired, interleaved, 50 pairs per cell):

- ILP mixed is **4-6% slower** than FP32, not faster. Energy unchanged.
- Real FP16/BF16 autocast: energy **-9 to -12%**, latency neutral to worse.
- Stage profile at QP3 on GPU: decode 88 ms = **80 ms entropy + 8 ms network**; encode 131 ms =
  80 entropy + 51 network. **73% of round trip is entropy coding.**

**Why quantization cannot fix latency:** it accelerates convolutions, and convolutions are 27% of
round trip and only 9% of decode. This is arithmetic, not a measurement artefact.

## 4. Open items

| Item | Status | Notes |
|---|---|---|
| **Report and slides** | **not started** | the graded deliverable; every number it needs now exists |
| Entropy multi-lane decision | open | largest latency lever (~80 ms of an 88 ms decode). `decompress_image` already reads `entropy_lanes`; `compress_image` never emits more than 1. Changes the bitstream contract, so needs an explicit decision |
| W8A8 activations | not implemented | only weights were quantized |
| Group/block scaling for INT4/FP4/FP8 | not implemented | makes the 4-bit columns pessimistic, not a verdict |
| QAT | not triggered by rule, not run | the lever would be E1/E4, worth ~14-28% more size |
| INT8 kernels (TensorRT/ONNX) | not built | the only route to a real INT8 speedup |
| Latency profile sample size | 2 images, 2 repeats | structure is unambiguous, exact ms should be re-measured |

## 5. Known measurement traps (do not repeat)

1. **Never compare configurations from a sequential benchmark.** The same FP32 config drifted from
   248 ms to 308 ms (+25%) and 3.5 to 4.3 J (+20%) between the start and end of one run. Use
   `bench_interleaved.py`.
2. **Check for orphaned `python.exe` before benchmarking.** A leftover process held ~75% of a core
   and inflated every timing in one run.
3. **Never sum each component's own worst QP** to predict a whole-model penalty. That gave 65.3%
   where the true value at QP0 was 26.7%. Sum per QP instead.
4. **Peak VRAM for fake-quant configs is a harness artefact** (+52 MB from the FP32 restore clone),
   not a memory regression.
5. **Windows schannel is broken here** (`SEC_E_NO_CREDENTIALS`); Python `urllib` works. Use it for
   any download.

## 6. Environment

Python 3.12.6, PyTorch 2.6.0+cu124, CUDA 12.4, RTX 3050 Laptop 4 GB (`sm_86`), scipy 1.18.1,
nvidia-ml-py 13.615.71, Pillow. No pandas, matplotlib, ortools, onnx or TensorRT, and none is
needed: every table and figure is stdlib plus Pillow, and the solver is scipy's HiGHS wrapper.

## 7. Suggested next action

Write the report. The evidence is complete: 69k sweep rows, a validated additive model, an optimal
assignment, three-dataset validation, BD-rate, and a stage profile that explains the absent speedup.
`RESOURCES.md` section 4 is already a claim-to-proof map you can paste into the results section.
