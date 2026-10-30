# QAT workspace - conditional

Status: empty and intentionally blocked until the PTQ map exists.

## Trigger

Run QAT only if a component that matters fails the PTQ gate:

```text
delta_psnr > 0.50 dB
or delta_bpp > 5%
or decode/CDF failure
```

Do not run QAT just because a low-bit format looks interesting.

## Scope

- Fine-tune only the fragile component.
- Freeze every other component.
- Use fake quantization with straight-through estimators.
- Learn weight scales and clipping parameters.
- Optionally learn activation scales.

## Data

- DIV2K/Flickr2K/CLIC train subset.
- Disjoint from Kodak and CLIC validation.
- 512x512 crops, fixed seed.
- 500-2000 crops for a short recovery run.

## Hardware

- RTX 3050 4 GB.
- Batch 1-2, gradient checkpointing, AMP.
- Start with 500-1000 steps and validate every 250.

## Planned outputs

- `qat/checkpoints/`
- `qat/results/`
- `results/tables/qat_comparison.csv`

## Decision

QAT is successful only if it recovers the fragile component without damaging
other QPs and without changing the entropy bitstream contract.
