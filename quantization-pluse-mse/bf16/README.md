# BF16

Status: planned. Hardware-supported on the RTX 3050.

## Scope

- Cast weights and activations to BF16.
- No calibration.
- One component at a time; all other components remain FP32.
- All 8 QPs.

## What this format answers

- Does reduced mantissa precision alone damage PULSE?
- Is BF16 a safer control than FP16 for wide-range activations?
- Which components are mantissa-sensitive?

## Planned config

```yaml
precision: bf16
granularity: tensor_cast
quant_mode: w_and_a
calibration: none
native: true
```

## Planned outputs

- `results/raw/bf16__...json`
- `results/tables/bf16_summary.csv`
- `results/figures/bf16_delta_psnr.png`

## Notes

- BF16 keeps the FP32 exponent range but has 7 mantissa bits.
- Do not claim a speedup without a measured BF16 backend.
