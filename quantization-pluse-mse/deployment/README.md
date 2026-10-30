# Deployment and energy workspace - conditional

Status: empty and intentionally blocked until the quality sweep is understood.

## What can be measured on the RTX 3050

| Format | Quality | Latency | Memory | Energy |
|---|---|---|---|---|
| FP32 | yes | yes | yes | yes |
| FP16 | yes | yes | yes | yes |
| BF16 | yes | yes | yes | yes |
| INT8 | yes | yes via TensorRT | yes | yes |
| FP8 E4M3 | simulated only | no | no | no |
| FP8 E5M2 | simulated only | no | no | no |
| FP4 E2M1 | simulated only | no | no | no |
| INT4 | simulated only | no | no | no |

## Planned path

1. Export encoder and decoder to ONNX.
2. Build TensorRT FP16 and INT8 engines.
3. Verify grouped/depthwise conv and PixelShuffle support.
4. Record every operator that falls back to FP16.
5. Keep the entropy path integer and CPU-side.
6. Measure latency, memory, and energy with the same protocol.

## Existing starting point

```text
deployment/npu_qualcomm_snapdragon_8_elite/export_onnx.py
```

The H100 path is Linux-specific and is not a Windows TensorRT path.

## Planned outputs

- `deployment/onnx/`
- `deployment/engines/`
- `results/tables/deployment.csv`
- `results/figures/pareto_latency_psnr.png`
- `results/figures/pareto_energy_psnr.png`

## Rules

- No simulated-format speed or energy claim.
- State the measurement scope and sampling interval.
- Compare identical inputs, QPs, and timing boundaries.
