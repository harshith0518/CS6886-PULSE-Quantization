# Planned ILP scripts

No script has been implemented yet.

| Script | Purpose | Input | Output |
|---|---|---|---|
| `00_check_inputs.py` | Validate PTQ tables and columns | `../quantization-pluse-mse/results/tables` | `results/input_check.json` |
| `01_build_coefficients.py` | Merge component/precision/QP coefficients | sweep tables | `results/coefficients.csv` |
| `02_solve_ilp.py` | CP-SAT solve | coefficients + config | `results/assignment.csv`, `results/ilp_summary.json` |
| `03_greedy_baseline.py` | Greedy/knapsack comparison | coefficients | `results/greedy_assignment.csv` |
| `04_validate_assignment.py` | Re-run the chosen assignment | assignment + model | `results/validation.csv` |
| `05_make_ilp_figures.py` | Assignment heatmap and Pareto plots | results | `results/figures/*.png` |

## CLI contract

```text
--config configs/ilp_model.yaml
--coefficients results/coefficients.csv
--output results
--objective latency
--epsilon 0.20
--memory-mb 4096
--allow-simulated false
--time-limit 60
--seed 0
```

## Rules

- Never invent a missing coefficient.
- Never allow a simulated format when `simulated_allowed` is false.
- Never claim speed from a simulated format.
- Always save the solver status and objective value.
- Always validate the assignment with the same protocol as the PTQ sweep.
