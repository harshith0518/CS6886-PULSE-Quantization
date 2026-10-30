# Finalized mixed-precision assignment

The model submitted for this project. One weight file serves all eight QPs, so a single
precision is assigned per component. Generated from `assignment_ilp.json` and
`assignment_ilp_detail.csv`; do not hand-edit.

## Summary

| Item | Value |
|---|---:|
| Components quantized | 19 of 19 |
| FP32 baseline | 54.688 MB |
| Mixed-precision model | **21.677 MB** |
| Size reduction | **60.36%** |
| Budget | worst-QP penalty <= 1.0% |
| Predicted worst-QP penalty | 0.9715% |
| Precision mix | FP16 x12, BF16 x5, INT8 x2 |

Measured worst-QP penalty: **0.672% Kodak, 0.851% CLIC, 0.935% Set14**.
BD-rate against FP32: **+0.477% / +0.552% / +0.486%**.

## Encoder (6 components)

| Comp | Precision | Parameters | Size MB | Share | Role | Module prefix |
|---|---|---:|---:|---:|---|---|
| **E0** | **BF16** | 393,728 | 0.7885 | 3.6% | patch_embedding | `encoder.patch_embed` |
| **E1** | **FP16** | 2,893,824 | 5.7989 | 26.8% | encoder_dcb_0 | `encoder.blocks.0` |
| **E2** | **INT8** | 2,893,824 | 2.9292 | 13.5% | encoder_dcb_1 | `encoder.blocks.1` |
| **E3** | **INT8** | 2,893,824 | 2.9292 | 13.5% | encoder_dcb_2 | `encoder.blocks.2` |
| **E4** | **FP16** | 2,893,824 | 5.7989 | 26.8% | encoder_dcb_3 | `encoder.blocks.3` |
| **E5** | **FP16** | 164,672 | 0.3310 | 1.5% | latent_projection | `encoder.norm_out`, `encoder.proj_out` |

## Entropy (4 components)

| Comp | Precision | Parameters | Size MB | Share | Role | Module prefix |
|---|---|---:|---:|---:|---|---|
| **H0** | **FP16** | 303,456 | 0.6127 | 2.8% | hyper_encoder | `entropy_model.hyper_enc` |
| **H1** | **FP16** | 154,688 | 0.3142 | 1.4% | hyper_decoder | `entropy_model.hyper_dec` |
| **H2** | **BF16** | 166,656 | 0.3366 | 1.6% | em_step_1 | `entropy_model.em_step1` |
| **H3** | **FP16** | 207,296 | 0.4173 | 1.9% | em_step_2 | `entropy_model.em_step2` |

## Decoder (9 components)

| Comp | Precision | Parameters | Size MB | Share | Role | Module prefix |
|---|---|---:|---:|---:|---|---|
| **D0** | **BF16** | 46,224 | 0.0927 | 0.4% | initial_projection | `decoder.up0` |
| **D1** | **FP16** | 142,832 | 0.2876 | 1.3% | decoder_dcb_0 | `decoder.blocks.0` |
| **D2** | **BF16** | 142,832 | 0.2876 | 1.3% | decoder_dcb_1 | `decoder.blocks.1` |
| **D3** | **FP16** | 246,528 | 0.4946 | 2.3% | latent_upsampling | `decoder.renderer.latent_upsample` |
| **D4** | **BF16** | 111,360 | 0.2243 | 1.0% | condition_projection | `decoder.renderer.condition_proj` |
| **D5** | **FP16** | 4,704 | 0.0096 | 0.0% | renderer_mlp | `decoder.renderer.mlp` |
| **D6** | **FP16** | 7,056 | 0.0144 | 0.1% | adaln | `decoder.renderer.adaln` |
| **D7** | **FP16** | 2,352 | 0.0048 | 0.0% | output_projection | `decoder.renderer.output` |
| **D8** | **FP16** | 2,304 | 0.0046 | 0.0% | renderer_input | `decoder.renderer.input_proj` |

## Totals

| | Parameters | Size MB |
|---|---:|---:|
| Quantized neural weights | 13,671,984 | 21.6767 |
| Kept FP32 by rule | 540,672 Meta Prior lookup + RMSNorm scales | not counted above |

Excluded from quantization by design: `entropy_model.scale_dec` (already integer),
`entropy_model.meta_prior` and `entropy_bottleneck` (lookup tables), all norms and biases,
and the `q_scale_*`, `q_basic`, `qp_bias`, `position_bias` controls.

## How to rebuild

```python
import json
from pulsequant.core import apply_assignment
assignment = json.load(open('quantization-pluse-mse/results/ilp/assignment_ilp.json'))['assignment']
handle = apply_assignment(model, assignment)   # fake-quantizes in place, restore() undoes it
```

Quantization is per-output-channel symmetric, applied to `Conv2d` weights only.
INT8 uses qmax=127; FP16 and BF16 are direct casts.
