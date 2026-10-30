# ILP Mixed-Precision Workspace

Status: **planning only**. No solver code or assignment has been implemented.

This folder is the dedicated workspace for mixed-precision selection after the
component-wise PTQ sweep. It is intentionally separate from
`quantization-pluse-mse/` so that the measurement data and the optimization
formulation do not get mixed together.

## Purpose

Given measured component x precision x QP results, choose one precision per
component such that:

- quality stays within an allowed `delta_psnr`,
- memory stays within 4 GB,
- only supported formats are used,
- latency or a latency/energy cost is minimized.

## Dependency

Do not start ILP until `quantization-pluse-mse` has produced:

- component x precision x QP PSNR and bpp,
- latency and memory coefficients,
- calibration and bitstream validity flags,
- a clear separation between simulated and measured formats.

## Folder layout

| Path | Meaning |
|---|---|
| `ilp.md` | Technical formulation, variables, constraints, solver plan |
| `TODO.md` | Task breakdown and progress tracking |
| `configs/ilp_model.yaml` | Machine-readable model configuration |
| `scripts/` | Planned solver and validation scripts |
| `results/` | Assignments, tables, and validation output |
| `reports/` | Final ILP report |
| `presentation/` | Slide outline, draft slide text, and speaking notes |

## Non-goals for now

- No ILP solve before the PTQ sweep exists.
- No invented coefficients.
- No speed claim from simulated FP8/FP4/INT4 formats.
- No full-model retraining.
- No production deployment build.

## Inputs required from the quantization workspace

- `quantization-pluse-mse/results/tables/sweep.csv`
- `quantization-pluse-mse/results/tables/component_precision_summary.csv`
- `quantization-pluse-mse/results/tables/bd_rate.csv`
- `quantization-pluse-mse/results/tables/failures.csv`

## Outputs

- `results/assignment.csv` - chosen precision per component and QP.
- `results/ilp_summary.json` - objective value, constraints, solver status.
- `results/validation.csv` - measured validation of the chosen assignment.
- `presentation/outline.md` - slide-by-slide ILP story.
