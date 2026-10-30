# FP8 E5M2

Status: planned. **Simulated only** on the RTX 3050 (`sm_86`).

## Scope

- Simulated FP8 E5M2 fake quantization.
- 1 sign, 5 exponent, 2 mantissa bits.
- Finite max 57344, min normal 2^-14.
- Wider range but lower mantissa precision than E4M3.
- Intended for activations with large dynamic range.
- One component at a time; all other components FP32.
- All 8 QPs.

## What this format answers

- Is the damage caused by range or by mantissa precision?
- Are activations safer in E5M2 than E4M3?
- Are weights too imprecise in E5M2?

## Planned config

```yaml
precision: fp8_e5m2
granularity: per_tensor_weight__per_tensor_activation
quant_mode: w_only_then_w_and_a_dynamic
calibration: none_first
native: false
simulated: true
```

## Planned outputs

- `results/raw/fp8_e5m2__...json`
- `results/tables/fp8_e5m2_summary.csv`
- `results/figures/fp8_e5m2_delta_psnr.png`

## Notes

- No latency, memory, or energy claim.
- Compare E5M2 activations against E4M3 activations directly.
