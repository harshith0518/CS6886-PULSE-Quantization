# ILP Mixed-Precision Plan

This document defines the mixed-precision selection problem for PULSE-MSE after the
component-wise PTQ sweep. It is a plan, not an implementation.

## 1. Why ILP

Uniform precision is easy but usually suboptimal:

- The encoder holds 85.6% of the parameters, but the decoder is only 5%.
- Some components may tolerate INT8; others may need FP16.
- The safe precision may depend on QP.
- The deployment budget is 4 GB VRAM on an RTX 3050.

ILP chooses one precision per component subject to quality and memory constraints.
It is only meaningful after the PTQ sweep provides measured coefficients.

## 2. Inputs

Required from `quantization-pluse-mse/results/`:

| File | Contents |
|---|---|
| `tables/sweep.csv` | per-image, per-component, per-precision, per-QP results |
| `tables/component_precision_summary.csv` | aggregated means and deltas |
| `tables/bd_rate.csv` | BD-rate by component and precision |
| `tables/failures.csv` | decode/CDF/bitstream failures |

Required columns:

```text
component, precision, qp, dataset, delta_psnr, delta_bpp,
encode_ms, decode_ms, peak_memory_mb, energy_j,
supported, simulated, decode_ok, cdf_ok, run_id
```

Do not invent missing coefficients. Measure or exclude.

## 3. Variables

```text
x[c, p, q] in {0, 1}
```

- `c` in the component set `C` (E0-E5, H0-H3, D0-D8).
- `p` in the precision set `P` (FP32, FP16, BF16, INT8, FP8, FP4, INT4).
- `q` in QPs 0-7.

If the assignment is shared across QPs, use `x[c, p]` instead.

## 4. Objective

Latency objective:

```text
min sum_{c,p,q} x[c,p,q] * latency[c,p,q]
```

Energy objective:

```text
min sum_{c,p,q} x[c,p,q] * energy[c,p,q]
```

Weighted objective:

```text
min alpha * normalized_latency + beta * normalized_energy
```

Use measured latency/energy only. For simulated formats, set the cost to the
simulated-format proxy or exclude them from the objective.

## 5. Constraints

### One precision per component

```text
sum_p x[c,p,q] = 1    for every c, q
```

### Quality

```text
sum_{c,p} x[c,p,q] * delta_psnr[c,p,q] <= epsilon[q]
```

Use `epsilon[q]` per QP if the sensitivity differs by QP.

### Memory

```text
sum_{c,p} x[c,p,q] * memory[c,p,q] <= 4096 MB
```

### Supported formats

```text
x[c,p,q] = 0    if p is not supported on the target
```

### Simulated formats

```text
x[c,p,q] = 0    for FP8/FP4/INT4 unless the assignment is explicitly
                a quality-only study
```

### Decode validity

```text
x[c,p,q] = 0    if decode_ok = false or cdf_ok = false
```

## 6. Solver

### Primary: OR-Tools CP-SAT

- Free and easy to install.
- Handles binary variables and linear constraints.
- Good for the expected problem size (19 components x 7 formats x 8 QPs).

### Fallback: Gurobi

Use only if available and if the model becomes large.

### Baseline: greedy/knapsack

Keep a greedy solver for comparison and sanity checks.

## 7. Validation

Validate the chosen assignment with the same protocol as the PTQ sweep:

1. Run the mixed assignment on Kodak.
2. Confirm on CLIC.
3. Compare against uniform INT8, uniform FP16, and FP32.
4. Report per-QP and per-image deltas.
5. Check decode success and CDF compatibility.
6. Check memory and latency on the RTX 3050.

If validation disagrees with the coefficients, update the coefficients, not the
story.

## 8. What ILP can and cannot claim

Can claim:

- A better quality/latency trade-off than uniform precision, if measured.
- A data-driven assignment under explicit constraints.

Cannot claim:

- Speedups from simulated FP8/FP4/INT4.
- Generalization beyond the measured datasets and QPs.
- A production deployment without a backend measurement.

## 9. Presentation story

1. PULSE is asymmetric: encoder 85.6%, decoder 5.0%.
2. Uniform precision is a blunt instrument.
3. We measured component x precision x QP sensitivity.
4. We formulated mixed precision as an ILP.
5. The solver chose a per-component assignment.
6. Validation shows the trade-off.
7. Limitations: simulated formats, 4 GB laptop, dataset scope.

## 10. Failure modes to watch

- Missing coefficients for some components.
- QP-dependent sensitivity that breaks a shared assignment.
- Grouped/depthwise convolution support in the deployment backend.
- CDF mismatch when entropy components are quantized.
- Memory underestimation from PyTorch allocator behavior.
- Overfitting the assignment to Kodak.
