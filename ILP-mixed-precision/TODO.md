# ILP Mixed-Precision - TODO and Progress

> Status: **unblocked**. The PTQ sweep is complete (69,233 rows, 135 configs, 0 failures).
> No coefficient table and no solver code exist yet.
> Model: `checkpoints/pulse-mse`.
> Hardware: RTX 3050 Laptop 4 GB, CUDA 12.4, PyTorch 2.6.0+cu124.
> Last updated: 2026-10-30.

## Progress at a glance

| Work package | Status | Evidence |
|---|---|---|
| ILP0. Inputs from PTQ sweep | **Unblocked** | `quantization-pluse-mse/results/_ledger/sweep.csv`, 69,233 rows |
| ILP1. Coefficient table | Ready to build | `configs/ilp_model.yaml`, `results/analysis/component_precision_metrics.csv` |
| ILP2. Solver implementation | Planned | `scripts/` is empty; `ortools` not installed |
| ILP3. Assignment validation | Blocked on ILP2 | needs a multi-component config in `pipeline.py` |
| ILP4. Presentation | Planned | `presentation/outline.md` |

---

### ILP0 - Confirm inputs and scope

**What to do**
- [ ] Confirm the PTQ sweep has covered all candidate components.
- [ ] Confirm which formats are real and which are simulated.
- [ ] Confirm the quality threshold per QP.
- [ ] Confirm the memory budget (4 GB).
- [ ] Confirm the objective: latency, energy, or weighted cost.
- [ ] Confirm whether the assignment is per QP or shared across QPs.
- [ ] Freeze the input table version and hash.

**What we have done**
- [x] Defined the dependency on `quantization-pluse-mse`.
- [x] Defined the candidate variables and constraints.
- [x] PTQ sweep exists: 7 formats x 19 components x 2 datasets x 8 QPs, 0 failures.
- [x] Real vs simulated settled: FP16, BF16 real quality and size; INT8 real with a TensorRT
  backend not yet built; FP8, INT4, FP4 simulated on `sm_86`.
- [x] QP dependence measured: INT8 damage grows with QP for encoder/decoder components
  (E0 0.154% -> 1.174%) and falls for entropy components (H3 38.152% -> 2.768%), so a
  per-QP assignment `x[c,p,q]` is justified over a shared `x[c,p]`.
- [ ] Objective not chosen yet: latency and energy coefficients do not exist, so the only
  fully measured objective today is model size in bytes per component.
- [ ] Input table version and hash not frozen.

---

### ILP1 - Build the coefficient table

**What to do**
- [ ] Merge component, precision, QP, PSNR, bpp, latency, memory, and validity.
- [ ] Add a `supported` flag for each format.
- [ ] Add a `simulated` flag for FP8/FP4/INT4.
- [ ] Add a `decode_ok` and `cdf_ok` flag.
- [ ] Fill missing coefficients only by measurement, never by guessing.
- [ ] Store the table as `results/coefficients.csv`.
- [ ] Record the source run IDs and hashes.

**What we have done**
- [x] Defined the required columns in `configs/ilp_model.yaml`.
- [x] Source data verified: `quantization-pluse-mse/results/_ledger/sweep.csv`
  (`bpp`, `psnr`, `bytes`, `stream_hash`, `recon_hash`, `decode_ok`) joined to `configs.csv`
  (`component`, `format`, `mode`, `granularity`, `fp32_bytes`, `quant_bytes`,
  `size_reduction_pct`, `effective_bits_per_param`).
- [x] Aggregates available: `results/analysis/component_precision_metrics.csv`
  (worst-QP mean penalty, max image penalty, size) and `recommendations.csv`.
- [ ] Latency, energy and peak-memory columns do not exist; `scripts/bench_memory_energy.py`
  is written but has never been run.
- [ ] No coefficient table yet.

---

### ILP2 - Implement the solver

**What to do**
- [ ] Implement the CP-SAT model with OR-Tools.
- [ ] Define one binary variable per component, precision, and QP.
- [ ] Add the one-precision-per-component constraint.
- [ ] Add the `delta_psnr <= epsilon` constraint.
- [ ] Add the memory constraint.
- [ ] Add the supported-format constraint.
- [ ] Add the simulated-format exclusion or penalty.
- [ ] Add the objective: latency or weighted latency/energy.
- [ ] Add a fallback greedy/knapsack solver for comparison.
- [ ] Save the solver status, objective, and assignment.

**What we have done**
- [x] Defined the formulation in `ilp.md`.
- [ ] No solver code yet.

---

### ILP3 - Validate the assignment

**What to do**
- [ ] Re-run the chosen assignment with the same evaluation protocol.
- [ ] Compare against uniform INT8, uniform FP16, and FP32.
- [ ] Check per-QP and per-image deltas.
- [ ] Check bitstream validity and CDF compatibility.
- [ ] Check memory and latency on the RTX 3050.
- [ ] Record failures and fallbacks.
- [ ] Update the assignment if validation disagrees with the coefficients.

**What we have done**
- [x] Defined the validation protocol.
- [ ] No assignment to validate yet.

---

### ILP4 - Presentation

**What to do**
- [ ] Write the slide-by-slide outline.
- [ ] Add the problem statement and why uniform precision is suboptimal.
- [ ] Add the variable and constraint formulation.
- [ ] Add the coefficient table and solver status.
- [ ] Add the chosen assignment heatmap.
- [ ] Add the validation comparison.
- [ ] Add limitations and simulated-vs-measured caveats.
- [ ] Add the final recommendation.

**What we have done**
- [x] Created `presentation/outline.md`.
- [x] Created `presentation/slides.md` with draft slide text.
- [ ] No rendered slides or figures yet.

---

### Immediate next actions

**What to do**
- [ ] Choose the objective from what is actually measured (size first; latency only after a bench).
- [ ] Validate `configs/ilp_model.yaml` against the actual sweep columns.
- [ ] Install OR-Tools.
- [ ] Build `results/coefficients.csv` from the ledger.
- [ ] Implement `scripts/solve_ilp.py` with a greedy baseline for comparison.
- [ ] Run the solver on the measured table.
- [ ] Implement a multi-component config in `pipeline.py` so the assignment can be validated.
- [ ] Validate the assignment.

**What we have done**
- [x] Created the ILP workspace and planning documents.
- [x] Defined the formulation and validation protocol.
- [x] Confirmed the PTQ sweep supplies the quality and size coefficients.
