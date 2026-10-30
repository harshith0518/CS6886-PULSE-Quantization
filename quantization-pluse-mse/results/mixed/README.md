# Whole-model evaluation: mixed precision on real bitstreams

The single-component sweep could only describe one component at a time. This directory holds
the whole-model experiment: one model carrying a different precision per component, measured
through the real encode/decode path on every image and every QP.

## Method

`src/pulsequant/core.py::apply_assignment` applies a per-component precision map in one pass.
`scripts/eval_assignment.py` measures a uniform format or a named assignment against the FP32
reference at all eight QPs and writes one row per image and QP.

```powershell
# whole model, uniform formats plus the three candidate assignments
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\eval_assignment.py `
  --dataset kodak --configs fp32 fp16 bf16 int8 safe greedy ilp
```

Assignments come from the decision surface (`safe`, `greedy`) and from the solver (`ilp`):

| Name | Source | Constraint |
|---|---|---|
| `safe` | `results/decision/decision_assignment_safe.json` | each component's own worst-QP penalty <= 1% |
| `greedy` | `results/decision/decision_assignment_greedy.json` | greedy path at a 1% total budget |
| `ilp` | `results/ilp/assignment_ilp.json` | MILP optimum at a 1% worst-QP total budget |

Analytic sizes use the measured per-component byte counts, so they are comparable with the
FP32 total of **54.688 MB** for the 19 main-sweep components.

## Correctness gate

`gate_<dataset>.json` requires FP32 to be bit-identical to the frozen reference, stream and
reconstruction hash, at all eight QPs.

| Dataset | Gate |
|---|---|
| kodak | `fp32_reproduces_reference: true`, 8 QPs |
| clic_professional | `fp32_reproduces_reference: true`, 8 QPs |

Everything below is therefore measured through the same path as the reference, not simulated.

## Results, Kodak, 24 images, QPs 0-7

| Config | Size MB | Saving | Worst-QP penalty | Mean penalty |
|---|---:|---:|---:|---:|
| FP32 | 54.688 | 0.0% | 0.000% | 0.000% |
| FP16 | 27.416 | 49.9% | 0.038% | 0.010% |
| BF16 | 27.416 | 49.9% | 0.106% | 0.060% |
| INT8 | 13.904 | 74.6% | 26.184% | 13.072% |
| safe | 17.904 | 67.3% | 2.633% | 1.559% |
| greedy | 24.565 | 55.1% | 0.386% | 0.303% |
| **ilp** | **21.677** | **60.4%** | **0.672%** | **0.461%** |

## Results, CLIC Professional, 41 images, QPs 0-7

| Config | Size MB | Saving | Worst-QP penalty | Mean penalty |
|---|---:|---:|---:|---:|
| FP32 | 54.688 | 0.0% | 0.000% | 0.000% |
| FP16 | 27.416 | 49.9% | 0.047% | 0.008% |
| BF16 | 27.416 | 49.9% | 0.155% | 0.010% |
| INT8 | 13.904 | 74.6% | 56.701% | 28.299% |
| safe | 17.904 | 67.3% | 4.301% | 2.757% |
| greedy | 24.565 | 55.1% | 0.358% | 0.259% |
| **ilp** | **21.677** | **60.4%** | **0.851%** | **0.597%** |

CLIC is roughly twice as harsh as Kodak for uniform INT8 and 1.6x for the `safe` assignment,
which is consistent with the earlier finding that E1 was green on Kodak and yellow on CLIC.
`greedy` and `ilp` transfer almost unchanged (0.386% -> 0.358% and 0.672% -> 0.851%), which is the
point of constraining the total rather than each component. Both stay inside the 1% budget on the
harder dataset.

## ILP per-QP behaviour, Kodak

| QP | bpp | PSNR | d_PSNR | d_bpp | penalty |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.1543 | 28.509 | -0.0041 | +0.311% | 0.429% |
| 1 | 0.2109 | 29.590 | -0.0047 | +0.251% | 0.382% |
| 2 | 0.2855 | 30.734 | -0.0096 | +0.108% | 0.351% |
| 3 | 0.3761 | 31.889 | -0.0211 | -0.043% | 0.439% |
| 4 | 0.4880 | 33.084 | -0.0289 | -0.199% | 0.399% |
| 5 | 0.6251 | 34.342 | -0.0401 | -0.325% | 0.439% |
| 6 | 0.7877 | 35.585 | -0.0592 | -0.498% | 0.577% |
| 7 | 0.9649 | 36.720 | -0.0634 | -0.458% | 0.672% |

At QP3 and above the quantization slightly *reduces* the bitrate: the encoder perturbation makes
the latents marginally cheaper to code, which partly offsets the PSNR loss. The penalty still
grows with QP, so QP7 remains the binding constraint.

## Additivity: the assumption is valid

`scripts/analyze_additivity.py` compares the measured whole-model penalty against the sum of the
single-component coefficients at the same QP and dataset.

| Config | Measured / predicted, Kodak | Measured / predicted, CLIC |
|---|---|---|
| uniform INT8 | 0.96 - 1.05 | 1.02 - 1.11 |
| safe | 0.75 - 0.95 | 0.92 - 1.00 |
| ilp | 0.83 - 1.34, all values < 0.7% | 0.97 - 1.11, all values < 0.9% |
| greedy | 0.73 - 1.66, values < 0.4% | 0.88 - 1.36, values < 0.4% |

Uniform INT8, the case that matters for the solver, is predicted within about 5% and slightly
under-predicted on CLIC. The single-component coefficients are therefore usable as ILP input
without re-measurement. BF16 ratios are meaningless because both sides are near zero.

The ILP run itself is a second check on the model: predicted 0.972% at QP7, measured 0.672%,
so the solver's estimate is conservative rather than optimistic.

## Reading the results

- Uniform INT8 is not usable: 26% on Kodak and 57% on CLIC. The three red components dominate.
- Uniform FP16 is a safe but blunt 50% saving at essentially zero cost.
- `ilp` is the best point found: **60.4% saving for 0.67% penalty**, and it transfers to CLIC.
- `safe` overspends: it pushes every component to its own 1% limit, which sums to 2.6-4.3%.
- `greedy` is slightly conservative in size but slightly better in quality than `ilp`; both are
  legitimate operating points, and the choice is a 2.9 MB versus 0.3% trade.

## Limitations

- Weight-only quantization. Activations are untouched; W8A8 would add damage this does not show.
- Fake quant, not a deployed kernel. Sizes are analytic byte counts, not measured runtime memory,
  and no latency or energy is claimed for any format.
- INT4, FP8 and FP4 remain per-output-channel-only, so they are pessimistic as 4-bit/8-bit float
  representatives.
- Two datasets. A third independent set was requested but no fourth image set exists in the repo
  and outbound TLS is blocked in this environment, so it could not be fetched. The harness accepts
  any `datasets/<name>`, so adding one is a single command.
