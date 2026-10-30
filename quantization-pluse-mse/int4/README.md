# INT4

Status: **positive-control run completed**. Simulated only on the RTX 3050.

The current `int4/results/` data is the D1 positive control: per-output-channel
symmetric `qmax=7`, weight-only. It is not the final group-16 INT4 scheme. The
group-16 run is still pending.

## Scope

- Simulated INT4 fake quantization.
- Signed range [-8, 7] or symmetric [-7, 7].
- Current control: per-output-channel symmetric, `qmax=7`.
- Target follow-up: per-group symmetric, group size 16.
- Weight-only only; no 4-bit activations.
- One component at a time; all other components FP32.
- All 8 QPs.

## What this format answers

- How much quality is left at 4-bit integer?
- Which components need group scaling?
- Does INT4 differ from FP4 E2M1 on the same component?
- Is QAT necessary for any component?

## Planned config

```yaml
precision: int4
granularity: per_group_weight
group_size: 16
quant_mode: w_only
calibration: group_scale
native: false
simulated: true
```

## Outputs

- `int4/results/sweep_int4.csv`
- `int4/results/configs_int4.csv`
- `int4/results/weight_stats_int4.csv`
- `int4/results/summary.csv`
- `int4/results/reference.json`
- `int4/results/manifests.json`
- `int4/results/figures/*.png`
- `int4/results/README.md`

## Notes

- No latency, memory, or energy claim.
- Report group size with every result.
- Expect large damage in the encoder's large 1x1 convolutions; that is a
  legitimate finding, not a failure.
