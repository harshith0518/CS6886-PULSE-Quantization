# INT8

Status: **active**. D1 and E1 weight-only runs completed on Kodak and CLIC.
Primary deployable low-precision target.

## Scope

- INT8 fake quantization first, then TensorRT INT8 for deployment.
- Weights: per-output-channel symmetric, range [-127, 127].
- Activations: per-tensor symmetric, range [-127, 127].
- Weight-only first; weights plus dynamic activations second;
  weights plus calibrated activations last.
- One component at a time; all other components FP32.
- All 8 QPs.

## What this format answers

- Which components tolerate INT8?
- Does INT8 damage depend on QP?
- Does activation quantization add damage beyond weights?
- Which components need QAT or a higher precision?

## Planned config

```yaml
precision: int8
granularity: per_output_channel_weight__per_tensor_symmetric_activation
quant_mode: w_only_then_w_and_a_dynamic_then_w_and_a_static
calibration: none_then_percentile_99_9
native: true_via_tensorrt
simulated: false_for_quality
```

## Outputs

- `int8/results/sweep_int8.csv`
- `int8/results/configs_int8.csv`
- `int8/results/weight_stats_int8.csv`
- `int8/results/summary.csv`
- `int8/results/reference.json`
- `int8/results/manifests.json`
- `int8/results/figures/*.png`
- `int8/results/README.md`

## Notes

- The released `entropy_model.scale_dec` is already INT8. Treat it as a control.
- Verify grouped/depthwise convolution support in TensorRT.
- Do not use test images for calibration.
