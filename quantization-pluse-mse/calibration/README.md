# Calibration workspace

Status: empty. Calibration data has not been added yet.

## Why this folder exists

Kodak and CLIC Professional are evaluation sets. They must not be used to
calibrate quantization scales or to fine-tune QAT. This folder is reserved for a
disjoint calibration set and its derived statistics.

## Recommended calibration set

| Option | Size | Pros | Cons |
|---|---:|---|---|
| DIV2K validation | 100 images | standard, clean, disjoint | needs download |
| CLIC 2020 train subset | 100-300 images | same domain as CLIC eval | needs download and split |
| Flickr2K subset | 100-300 images | high resolution | needs download |

Use 32-128 crops of 512x512, fixed seed. Store image hashes and source paths.

## Planned files

```text
calibration/
  images/                  # or a manifest pointing outside the repo
  manifest.json            # source, hashes, crop coordinates, seed
  activation_ranges.npz     # per component, QP, tensor
  clipping_search.json     # percentile/MSE-optimal results
  README.md
```

## Planned statistics

For each component, QP, and tensor:

- min, max, mean, std
- percentile 99.9 and 99.99
- saturation rate at candidate clips
- SQNR at candidate clips
- MSE-optimal clip

## Modes to compare

1. Shared across QPs.
2. Per QP.

The main sweep starts with shared calibration, then checks per-QP calibration on
the components that show QP-dependent damage.
