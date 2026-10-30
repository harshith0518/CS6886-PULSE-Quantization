# ILP results workspace

Status: empty. Blocked on the PTQ sweep.

## Planned files

| File | Contents |
|---|---|
| `coefficients.csv` | component x precision x QP coefficients |
| `assignment.csv` | chosen precision per component and QP |
| `greedy_assignment.csv` | greedy baseline assignment |
| `ilp_summary.json` | solver status, objective, constraints, runtime |
| `validation.csv` | measured validation of the chosen assignment |
| `input_check.json` | PTQ input completeness check |
| `figures/assignment_heatmap.png` | component x QP precision choice |
| `figures/pareto_latency_psnr.png` | uniform vs mixed |
| `figures/pareto_energy_psnr.png` | uniform vs mixed |

## Assignment schema

```text
component, qp, precision, delta_psnr, delta_bpp, latency_ms, memory_mb,
energy_j, supported, simulated, selected_by
```

## Validation schema

```text
component, qp, precision, dataset, psnr, bpp, delta_psnr, delta_bpp,
encode_ms, decode_ms, peak_memory_mb, energy_j, decode_ok, cdf_ok,
run_id
```

## Rules

- Report the solver status and objective.
- Report the constraints that are active.
- Report uniform INT8, uniform FP16, and FP32 baselines.
- Mark simulated formats explicitly.
- Do not hide infeasible or fallback cases.
