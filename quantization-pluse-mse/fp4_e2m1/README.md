# FP4 E2M1

Status: planned. **Simulated only** on the RTX 3050.

## Scope

- Simulated FP4 E2M1 fake quantization.
- 1 sign, 2 exponent, 1 mantissa bit.
- Positive finite values approximately {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
- Block-wise scale, block size 16.
- Weight-only only; no 4-bit activations.
- One component at a time; all other components FP32.
- All 8 QPs.

## What this format answers

- How much quality is left at 4-bit float?
- Which components need block scaling more than others?
- Is FP4 usable for any component without QAT?

## Planned config

```yaml
precision: fp4_e2m1
granularity: per_block_weight
block_size: 16
quant_mode: w_only
calibration: block_scale
native: false
simulated: true
```

## Planned outputs

- `results/raw/fp4_e2m1__...json`
- `results/tables/fp4_e2m1_summary.csv`
- `results/figures/fp4_e2m1_delta_psnr.png`

## Notes

- No latency, memory, or energy claim.
- Report block size with every result.
- Expect large damage in small components; that is a valid result.
