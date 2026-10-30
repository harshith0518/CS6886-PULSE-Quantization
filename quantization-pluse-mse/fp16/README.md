# FP16

Status: planned. Hardware-supported on the RTX 3050.

## Scope

- Cast weights and activations to FP16.
- No calibration.
- One component at a time; all other components remain FP32.
- All 8 QPs.

## What this format answers

- Does FP16 change PSNR or bpp at all?
- Is FP16 a safe control before INT8?
- Does FP16 overflow or underflow in any component?

## Planned config

```yaml
precision: fp16
granularity: tensor_cast
quant_mode: w_and_a
calibration: none
native: true
```

## Planned outputs

- `results/raw/fp16__...json`
- `results/tables/fp16_summary.csv`
- `results/figures/fp16_delta_psnr.png`

## Notes

- FP16 has a limited exponent range; log overflow and saturation.
- Do not assume FP16 is lossless.
- Keep the entropy path integer and CPU-side.
