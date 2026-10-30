# FP8 E4M3

Status: planned. **Simulated only** on the RTX 3050 (`sm_86`).

## Scope

- Simulated FP8 E4M3 fake quantization.
- 1 sign, 4 exponent, 3 mantissa bits.
- Finite max 448, min normal 2^-6.
- Per-tensor weight scale by default; optional per-channel.
- Activation scale dynamic per tensor first, calibrated later.
- One component at a time; all other components FP32.
- All 8 QPs.

## What this format answers

- Does an 8-bit float with tight range preserve PULSE quality?
- Which components are range-limited rather than mantissa-limited?
- Does E4M3 differ from INT8 on the same component?

## Planned config

```yaml
precision: fp8_e4m3
granularity: per_tensor_weight__per_tensor_activation
quant_mode: w_only_then_w_and_a_dynamic
calibration: none_first
native: false
simulated: true
```

## Planned outputs

- `results/raw/fp8_e4m3__...json`
- `results/tables/fp8_e4m3_summary.csv`
- `results/figures/fp8_e4m3_delta_psnr.png`

## Notes

- No latency, memory, or energy claim.
- Log saturation and SQNR per tensor.
- If a component overflows 448, try E5M2 or per-channel scaling.
