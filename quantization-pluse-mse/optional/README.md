# Optional formats

Status: not scheduled. These are only considered after the core formats are done.

| Format | Bits | Scaling | Why optional |
|---|---:|---|---|
| NF4 | 4 | block-wise non-uniform | useful for weight-only LLM-style quantization, unproven here |
| MXFP4 | 4 | OCP microscaling, block 32 | needs a block-scaled runtime |
| MXFP8 | 8 | OCP microscaling, block 32 | needs a block-scaled runtime |
| FP6 E3M2 | 6 | per-tensor/per-block | no native support |
| FP6 E2M3 | 6 | per-tensor/per-block | no native support |
| TF32 | 19 effective | compute mode | not a storage format; not a weight sweep |

Do not add these to the main heatmap unless the core FP8/FP4/INT4 results show
that the extra format is needed to explain a component's behaviour.
