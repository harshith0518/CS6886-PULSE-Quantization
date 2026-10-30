# QAT fine-tuning pool

A pool for quantization-aware fine-tuning that is **disjoint from every evaluation set** used in this
project. It is not the pretraining corpus; see `../metrics/qat_cost.md` for why QAT does not need one.

## Contents

| Item | Value |
|---|---|
| Source | DIV2K **validation** split (`DIV2K_valid_HR.zip`) |
| URL | <https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip> |
| Images | 100, RGB, 1356x2040 class (24 distinct sizes from 1356x2040 to 2040x2040) |
| Size | 428.7 MB |
| Per-file SHA-256 | `div2k_valid/manifest.json` |
| Fetched | 2026-10-30, 602 s at about 0.7 MB/s |

## Why this and not the pretraining data

The released `pulse-mse` was pretrained on Open Images plus SAM-1B (LR stage) then Flickr2K +
DIV2K train + CLIC train (HR stage, 4,035 images). That corpus is not reproducible here: Open
Images is roughly 1.9M images and SAM-1B roughly 11M, together terabytes, and the DIV2K *train*
archive itself served at only ~0.1 MB/s, which is hours for 3.5 GB.

QAT is fine-tuning, not pretraining. A few hundred representative natural images supply enough
random 512x512 crops for 500-2,000 steps, which is the regime measured in `qat_cost.md`. This pool
provides 100 images, which at batch 4 and 1,000 steps is roughly 2,500 crops, so there is ample
variation.

## Leakage control

`tools/fetch_qat_pool.py` refuses to build a pool from any of `kodak`, `clic_professional`,
`set14` or `kodak_thumb`. Verified after download: **zero filename overlap** with all three
evaluation sets.

Caveat to state in a report: DIV2K validation is a normal, widely used fine-tuning source, but it is
the *validation* split of DIV2K rather than its training split. If a reviewer asks whether the QAT
pool came from training data, the honest answer is "DIV2K validation, disjoint from all reported
test sets".

## Usage

```powershell
# re-fetch from scratch
.\venv\Scripts\python.exe tools\fetch_qat_pool.py --source div2k_valid

# larger pool, slow download, 800 images
.\venv\Scripts\python.exe tools\fetch_qat_pool.py --source div2k_train
```

Pass `datasets/qat_pool/div2k_valid` to the training runner's data arguments. For a 512-crop run:

```powershell
.\venv\Scripts\python.exe train_mse.py --init checkpoints/pulse-mse `
  --train-data datasets/qat_pool/div2k_valid --crop-size 512 --max-steps 100
```

Note that the stock runner trains the whole backbone and has no block-selective freezing option, so
a QAT run restricted to chosen components needs a custom loop. `datasets/qat_pool/` is gitignored;
only this README and any manifest are tracked.
