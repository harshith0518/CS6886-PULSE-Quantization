# PULSE-MSE Quantization Workspace

Status: **sweep complete, synthesis pending**. The weight-only screen is finished for 7 formats
across all 19 components on full-resolution Kodak and CLIC, all 8 QPs: 69,233 rows, 135 configs,
0 failures. What remains is analysis, the QAT/greedy/ILP decision, the mixed-precision model,
and the write-up.

This folder is the working area for the component-wise precision study of the released
`pulse-mse` bundle. The goal is to answer, one component at a time:

> Which parts of PULSE-MSE tolerate FP16/BF16/FP8/INT8/FP4/INT4, at which QP,
> and where does quality or bitrate actually break?

The companion documents are:

- `TODO.md` - complete task breakdown and progress tracking.
- `quantization.md` - technical plan for the quantization study.
- `configs/` - machine-readable component, precision, and experiment manifests.
- `scripts/pipeline.py` - one entry point: `inventory`, `reference`, `sweep`, `export`.
- `src/pulsequant/` - three universal modules: `core.py`, `eval.py`, `plots.py`.
- `results/_ledger/` - raw append-only sweep, progress, manifests, reference, and summary.
- `int8/results/`, `int4/results/`, ... - per-precision review surfaces with tables and plots.
- `fp32/`, `fp16/`, `bf16/`, `fp8_e4m3/`, `fp8_e5m2/`, `int8/`, `fp4_e2m1/`, `int4/` - per-precision workspaces.
- `qat/`, `deployment/` - conditional follow-up work.
- `ilp/` - short pointer to the dedicated `../ILP-mixed-precision/` workspace.
- `optional/` - NF4, MXFP4, MXFP8, FP6; not scheduled.

## Model under test

`checkpoints/pulse-mse`:

| Item | Value |
|---|---:|
| Architecture | PULSE-MSE base, `M=320`, `z_ch=96`, hidden 144, renderer 48 |
| Neural parameters | 14,179,432 |
| Loaded parameters with Meta Prior lookup table | 14,720,104 |
| FP32 neural size | 54.09 MB |
| INT8 neural size | 13.52 MB |
| Encoder share | 12,133,696 (85.6%) |
| Entropy share | 1,337,224 (9.4%) |
| Decoder share | 708,504 (5.0%) |
| Convolution weights | 14,127,360 (99.6% of neural parameters) |

Important: `entropy_model.scale_dec` is **already integer** in the released bundle.
`entropy_control_int.pt` replaces the float scale decoder with
`IntegerLinearCDFIndexDecoder`, and the Meta Prior is a fitted lookup table. The
baseline is therefore already a mixed FP32-neural + INT8-scale-decoder system, not a
pure FP32 model.

## Baseline already reproduced

Existing artifacts in `benchmark_results/`:

| QP | Kodak thumb bpp | Kodak thumb PSNR | CLIC bpp | CLIC PSNR |
|---:|---:|---:|---:|---:|
| 0 | 0.2970 | 27.769 | 0.1122 | 30.421 |
| 1 | 0.3701 | 28.931 | 0.1518 | 31.439 |
| 2 | 0.4640 | 30.145 | 0.2028 | 32.470 |
| 3 | 0.5764 | 31.359 | 0.2659 | 33.488 |
| 4 | 0.7115 | 32.623 | 0.3433 | 34.495 |
| 5 | 0.8705 | 33.966 | 0.4417 | 35.548 |
| 6 | 1.0695 | 35.280 | 0.5590 | 36.542 |
| 7 | 1.2471 | 36.323 | 0.6975 | 37.482 |

The table above is the **thumbnail** baseline and is smoke/G0a only. The full-resolution
Kodak FP32 reference is now present: QP0 **0.1538 bpp / 28.513 dB**, QP7 **0.9689 bpp /
36.783 dB**. Use full-resolution Kodak and CLIC for claims; use thumbnails only to match
the frozen `benchmark_results` JSON.

## Hardware reality on the RTX 3050 Laptop

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| Compute capability | `sm_86` (Ampere) |
| VRAM | 4 GB |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.12.6 |

Precision consequences:

- Native: FP32, FP16, BF16, INT8 (INT8 via TensorRT).
- No native FP8 on `sm_86`; FP8 must be simulated for quality/sensitivity.
- FP4 and INT4 have no reliable native laptop path; treat them as simulated formats
  for quality and as unverified for latency/energy.
- Any speed or energy claim must be measured, not inferred from simulated formats.

## Current environment status

- `MLCodec_extensions_cpp` was rebuilt on 2026-10-20 and imports successfully.
- Smoke test passed: `kodim01t.jpg`, QP 3, 0.6201 bpp, 30.556 dB, 89.4 ms decode.
- Installed: `numpy` 2.5.2, `scipy` 1.18.1, `pillow`, `PyYAML`, `tqdm`, `psutil`,
  `nvidia-ml-py` 13.615.71, `torch` 2.6.0+cu124.
- Not installed: `ortools` (needed for ILP), `pytest`, `onnx`, `onnxruntime`, `tensorrt`,
  `pandas`, `matplotlib`, `seaborn`, `pyarrow`.
- The analysis scripts are deliberately stdlib + PIL, so `pandas` and `matplotlib` are **not**
  required to reproduce any table or figure in `results/`.
- Tests run without `pytest`:
  `python -m unittest discover -s quantization-pluse-mse/tests` -> 4 passed.

- There is no separate calibration set and no QAT fine-tuning set inside `datasets/`.
  `datasets/clic_professional` (41 images) and `datasets/kodak` (24 images) are
  evaluation sets. See `quantization.md` for the data plan.

## Rules for every experiment

1. Change **one component at a time**; keep every other component FP32.
2. Use the same images, padding, QP, seed, and evaluation order as the baseline.
3. Separate **quality** (simulated formats) from **deployment** (hardware-supported formats).
4. Never claim speed or energy gains for a simulated format.
5. Keep test images out of calibration and QAT data.
6. Log the exact config, model hash, git commit, package versions, and run command.
7. If a component cannot be decoded bit-exactly, record the failure instead of hiding it.

## Implementation status

1. Entropy extension rebuilt and smoke-tested.
2. FP32 reference re-run through the harness on `kodak_thumb`, `kodak` and CLIC; G0a and noise floor recorded.
3. Component inventory generated; Table 4 check within 1% except hyper dec -1.36%.
4. Fake quantization for all 7 formats, in-place apply/restore, weight hashes, and controls implemented.
5. **Main weight-only screen complete**: 7 formats x 19 components x {Kodak 24, CLIC 41} x 8 QPs
   = 69,233 rows, 135 configs, 0 failures, `restored_ok` 135/135.
6. Per-precision review surfaces exist for all 7 formats at `{fmt}/results/` with 6 figures each;
   the raw ledger is under `results/_ledger/` (gitignored, regenerate with `pipeline.py sweep`).
7. Component x precision analysis in `results/analysis/`: table, 2 heatmaps, `recommendations.csv`.
8. Headline: 13/19 components are green at INT8 (70.4% of params, 70.8% of kMAC/px);
   Red are H0, H3, D5 (3.8% of params, 3.0% of kMAC/px). QAT is not triggered by rule.
9. Not yet done: whole-model/additivity, W8A8, calibration, group scaling, BD-rate, ILP,
   latency/energy, report and slides. A third evaluation dataset is not present in `datasets/`.
