# ILP report

Status: empty. Blocked on the PTQ sweep and solver.

## Planned outline

1. Why uniform precision is suboptimal.
2. Inputs from the component-wise PTQ sweep.
3. ILP variables and constraints.
4. Solver and objective.
5. Chosen assignment.
6. Validation against uniform INT8/FP16/FP32.
7. Limitations and simulated-vs-measured separation.
8. Recommendation.

## Required tables

- Coefficient table.
- Solver summary.
- Assignment table.
- Validation table.
- Uniform baseline comparison.

## Required figures

- Component x QP assignment heatmap.
- Latency/PSNR Pareto plot.
- Energy/PSNR Pareto plot, if energy is measured.
