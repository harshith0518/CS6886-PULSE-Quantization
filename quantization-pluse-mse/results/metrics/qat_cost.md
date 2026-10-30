# Training data and the cost of a QAT trial

Two questions: what the released `pulse-mse` was trained on, and what one QAT trial would cost on
this hardware.

## 1. Training data for the baseline model

Authoritative source: `docs/training.md` in this repo, which documents this specific model's stages.

Training is two-stage, and the released `pulse-mse` is the MSE-objective member of the family.

| Stage | Argument | Data | Count |
|---|---|---|---|
| MSE LR, crop 512 | `--train-data` | Open Images training images + SAM-1B, prepared per the CoD 512-resolution pipeline (`anno_512/hq_anno.txt`); 512-pixel short edge, centre-cropped 512x512 | not stated numerically |
| MSE HR, crop 1024 | `--hr-data` | Flickr2K + DIV2K train + CLIC train | **4,035 images** |
| Perceptual Stage II / ROI | `--roi-data` | the same HR pool with face and text annotations, ROI-biased sampling | not used by `pulse-mse` |

HR breakdown: Flickr2K 2,650, DIV2K training 800, CLIC training 585. The docs state CLIC validation
and test images are excluded to prevent data leakage, and that LSDIR is not included. The
256-resolution pool including ImageNet-21K is explicitly not used.

Cross-check: the paper PDF mentions DIV2K and CLIC but the extracted text was too mangled by
encoding to quote reliably, so the counts above come from `docs/training.md`, not from the paper.

**Important for us:** none of this data is in the repository. `datasets/` holds only the three
evaluation sets. Any QAT run would need the training sets downloaded first.

## 2. Full training scale

From `train_mse.py --check-config`:

| Stage | Crop | Batch | Steps | Optimiser | Runtime precision |
|---|---|---:|---:|---|---|
| `mse_lr` | 512 | 8 | 500,000 | SOAP | bf16 |
| `mse_hr` | 1024 | 8 | 50,000 | SOAP | bf16 |

550,000 optimizer steps total. `lambda_min` 8.0, `lambda_max` 80.0, gradient clip 1.0. Reproducing
this from scratch on a 4 GB laptop is not realistic, which is why the project rule forbids it.

## 3. Cost of one QAT trial, measured

`scripts/bench_qat_step.py` runs real forward + rate/distortion loss + backward + optimiser step,
using the codec's own `RDObjective` and training forward, on a 512x512 crop at QP3.

| Batch | Step time | Peak VRAM |
|---:|---:|---:|
| 1 | 0.162 s | 415 MB |
| 2 | 0.130 s | 714 MB |
| 4 | 0.157 s | 1,328 MB |

Projection at the measured 0.13 s per step:

| Steps | Compute time |
|---:|---|
| 500 | **1.1 min** |
| 1,000 | **2.2 min** |
| 2,000 | 4.3 min |

The HR crop of 1024 could **not** be measured, because no image in the repository is large enough to
supply a 1024x1024 crop and the slice silently clamps. An earlier attempt produced a misleading
number that was deleted rather than reported. Expect roughly 4x the 512-crop step time.

## 4. What this actually implies

**Compute is not the obstacle.** One trial of 1,000 steps is a couple of minutes. Even 10 trial
configurations with 2,000 steps each is under an hour of GPU time.

**The real costs are elsewhere:**

| Cost | Status |
|---|---|
| Training data | absent; Open Images + SAM-1B and Flickr2K/DIV2K/CLIC-train must be downloaded |
| STE fake-quant layer | not implemented in the training path; needs a straight-through wrapper around `fake_quant` |
| Block-selective freezing | the stock runner trains the whole backbone; no `--freeze-except` option exists, so a custom loop is needed (~100 lines) |
| Evaluation per trial | already solved: `eval_assignment.py` gives a three-dataset verdict in about 10 minutes |
| Hyper-parameter search | the 1-2 minute step cost multiplies by however many trials |

**A rule needs revisiting.** The project rule says not to train the full model on the 4 GB card.
That is true for from-scratch training, but a *fine-tuning* step fits comfortably: 1,328 MB at
batch 4, against a 4 GB budget. QAT of the whole backbone at crop 512 is VRAM-feasible here; the
constraint is data and engineering, not memory.

## 5. Bottom line

| Question | Answer |
|---|---|
| Baseline training data | LR: Open Images + SAM-1B at 512 crops. HR: Flickr2K + DIV2K train + CLIC train = 4,035 images at 1024 crops |
| Full training | 550,000 steps, batch 8, bf16 |
| One QAT trial, 1,000 steps | **about 2 minutes of GPU time**, 714 MB at batch 2 |
| Practical blocker | training data and a ~100-line STE/freezing harness, not compute |

## Reproduce

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\bench_qat_step.py --crop 512 --batch 1 2 4
```
