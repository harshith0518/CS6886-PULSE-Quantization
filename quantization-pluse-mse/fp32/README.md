# FP32 - reference

Status: reference only. No quantization is applied here.

## Scope

- The released `pulse-mse` bundle.
- All neural components in FP32.
- Existing integer `entropy_model.scale_dec` and CDF/rANS path unchanged.
- All 8 QPs, Kodak and CLIC Professional.

## Why it exists

Every component/precision/QP result is compared against this row. Do not use a
different evaluation protocol for the reference.

## Inputs

- `../../checkpoints/pulse-mse`
- `../../datasets/kodak`
- `../../datasets/clic_professional`
- Existing baseline JSON in `../../benchmark_results/`

## Planned outputs

- `results/tables/baseline_recheck.csv`
- `results/figures/baseline_rd_curve.png`
- `results/raw/fp32__reference__...json`

## Notes

- This is the only row that is allowed to be called "the model".
- Re-run it after the entropy extension is rebuilt.
- Keep the original baseline JSON untouched.
