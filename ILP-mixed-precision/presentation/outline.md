# ILP Mixed Precision - Presentation Outline

Status: planning only. No slides rendered yet.

## Slide 1 - Title

**Title:** Mixed-precision selection for PULSE-MSE under a 4 GB budget

**Subtitle:** Component-wise sensitivity to an ILP assignment

**Speaker note:** We first measured component-wise precision damage, then used ILP
to choose one precision per component. We do not claim speed from simulated formats.

## Slide 2 - Why uniform precision is blunt

- Encoder: 85.6% of neural parameters.
- Decoder: 5.0%.
- Entropy neural parts: 9.4%.
- One uniform precision treats all three the same.
- QP changes the operating point; sensitivity may change with QP.

**Figure:** parameter-share bar chart.

## Slide 3 - What we measured

- One component at a time; everything else FP32.
- Component x precision x QP.
- Metrics: delta_psnr, delta_bpp, latency, memory, decode/CDF validity.
- Kodak for screening; CLIC for confirmation.

**Figure:** component x QP delta_psnr heatmap.

## Slide 4 - Which components are fragile

- Rank components by worst-case delta_psnr.
- Separate weight-only from weight+activation damage.
- Flag QP-dependent components.
- Flag components that fail decode or CDF checks.

**Table:** top fragile components with delta_psnr and delta_bpp.

## Slide 5 - The ILP formulation

```text
min sum x[c,p,q] * cost[c,p,q]

sum_p x[c,p,q] = 1
sum x[c,p,q] * delta_psnr[c,p,q] <= epsilon[q]
sum x[c,p,q] * memory[c,p,q] <= 4096 MB
x[c,p,q] = 0 for unsupported or simulated formats
```

**Speaker note:** Variables, objective, and constraints are all measured. No invented
coefficients.

## Slide 6 - Solver and data

- OR-Tools CP-SAT first.
- Gurobi only if available.
- Greedy/knapsack baseline for comparison.
- Coefficients from the PTQ sweep.
- Missing coefficients are measured or excluded.

**Figure:** solver status and objective value.

## Slide 7 - Chosen assignment

- Component x QP precision choice.
- Highlight components that stay FP16.
- Highlight components that can be INT8.
- Mark simulated formats as excluded.

**Figure:** assignment heatmap.

## Slide 8 - Validation

- Re-run the chosen assignment on Kodak and CLIC.
- Compare against uniform INT8, uniform FP16, FP32.
- Report delta_psnr, delta_bpp, latency, memory.
- Report decode/CDF failures.

**Table:** uniform vs mixed comparison.

## Slide 9 - Limitations

- FP8/FP4/INT4 are simulated on the RTX 3050.
- TensorRT on Windows is not yet validated.
- The assignment is only as good as the measured coefficients.
- Kodak-to-CLIC transfer must be checked.
- 4 GB VRAM limits the deployment scope.

## Slide 10 - Recommendation

- Use INT8 where the sweep says it is safe.
- Keep fragile components in FP16.
- Do not claim speed for simulated formats.
- Next step: validate the mixed assignment with a real backend.

## Backup slides

- Component inventory and parameter counts.
- Full precision format table.
- Per-QP sensitivity.
- Solver constraints and active set.
- Failure table.
- Reproducibility commands and hashes.
