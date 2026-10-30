# ILP Mixed Precision - Slide Content

Status: draft text. No visual theme applied.

---

## Slide 1 - Title

**Mixed-precision selection for PULSE-MSE under a 4 GB budget**

- Component-wise sensitivity to an ILP assignment
- Model: `pulse-mse`
- Hardware: RTX 3050 Laptop, 4 GB
- Date: 2026

**Speaker note:** This is the optimization stage, not the measurement stage. The
measurement stage is the component-wise PTQ sweep.

---

## Slide 2 - Why uniform precision is blunt

- Encoder: 12.13M parameters, 85.6%.
- Entropy neural parts: 1.34M, 9.4%.
- Decoder: 0.71M, 5.0%.
- One precision for all components ignores this asymmetry.
- QP 0-7 changes the operating point; sensitivity may change with QP.

**Figure:** parameter-share bar chart.

---

## Slide 3 - What we measured

- One component at a time; everything else FP32.
- Component x precision x QP.
- Metrics: delta_psnr, delta_bpp, latency, memory, decode/CDF validity.
- Kodak for screening; CLIC for confirmation.
- FP16/BF16/INT8 are real; FP8/FP4/INT4 are simulated.

**Figure:** component x QP delta_psnr heatmap.

---

## Slide 4 - Fragility map

- Rank components by worst-case delta_psnr.
- Separate weight-only from weight+activation damage.
- Flag QP-dependent components.
- Flag decode or CDF failures.
- Keep H4, H5, D9 as controls.

**Table:** top fragile components.

---

## Slide 5 - ILP formulation

```text
min sum x[c,p,q] * cost[c,p,q]

sum_p x[c,p,q] = 1
sum x[c,p,q] * delta_psnr[c,p,q] <= epsilon[q]
sum x[c,p,q] * memory[c,p,q] <= 4096 MB
x[c,p,q] = 0 for unsupported or simulated formats
```

- `c`: component.
- `p`: precision.
- `q`: QP.
- `x`: binary selection.

**Speaker note:** No invented coefficients. Missing data is measured or excluded.

---

## Slide 6 - Solver and data

- OR-Tools CP-SAT first.
- Gurobi only if available.
- Greedy/knapsack baseline.
- Coefficients from the PTQ sweep.
- Solver status and objective are saved.

**Figure:** solver summary.

---

## Slide 7 - Chosen assignment

- One precision per component per QP.
- Highlight components that stay FP16.
- Highlight components that can be INT8.
- Mark simulated formats as excluded.
- Show active constraints.

**Figure:** assignment heatmap.

---

## Slide 8 - Validation

- Re-run the assignment on Kodak and CLIC.
- Compare uniform INT8, uniform FP16, FP32.
- Report delta_psnr, delta_bpp, latency, memory.
- Report decode/CDF failures.
- If validation disagrees, fix coefficients, not the story.

**Table:** uniform vs mixed.

---

## Slide 9 - Limitations

- FP8/FP4/INT4 are simulated on the RTX 3050.
- TensorRT on Windows is not yet validated.
- The assignment is only as good as the measured coefficients.
- Kodak-to-CLIC transfer must be checked.
- 4 GB VRAM limits deployment scope.

---

## Slide 10 - Recommendation

- Use INT8 where the sweep says it is safe.
- Keep fragile components in FP16.
- Do not claim speed for simulated formats.
- Next step: validate with a real backend.
- Publish the assignment and the validation table.

---

## Backup slides

- Component inventory and parameter counts.
- Precision format table.
- Per-QP sensitivity.
- Solver constraints and active set.
- Failure table.
- Reproducibility commands and hashes.
