# Runbook

This document tells a new user how to build the project and how to reproduce the reported numbers.
Read the sections in order. Each step gives the command and the expected result.

Time: about 15 minutes to the first verified result. About 2 hours for the full evaluation. About 1 day
to rebuild everything from raw measurements.

## 1. What this repository contains

The repository has two parts.

- **Upstream PULSE.** The folders `codec/`, `deployment/` and `configs/` hold the released codec. This
  code is unchanged.
- **The quantization study.** The folder `quantization-pluse-mse/` holds the measurement code, the
  solver, the fine-tuning code and all result tables.

The study answers one question. Which parts of the codec can use a low precision?

## 2. What you need

| Item | Requirement | Note |
|---|---|---|
| Operating system | Windows 10 or 11, or Linux | The commands below use PowerShell. Remove `.exe` for bash. |
| Python | 3.12 | Version 3.13 is not tested. |
| GPU | 4 GB of video memory | One RTX 3050 Laptop did all the work. |
| CUDA | 12.4 | You can use the CPU for evaluation. Fine-tuning needs the GPU. |
| Disk | 10 GB free | The model is 84 MB. The data is 0.5 GB. |
| Compiler | MSVC Build Tools on Windows | You need it one time, to build the entropy extension. |

## 3. Get the code

```powershell
git clone https://github.com/harshith0518/CS6886-PULSE-Quantization.git
cd CS6886-PULSE-Quantization
```

## 4. Build the environment

Do these steps in order.

1. Create a virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Upgrade pip.

```powershell
python -m pip install --upgrade pip
```

3. Install the Python packages.

```powershell
pip install -r requirements.txt
```

4. Build the native entropy extension.

```powershell
pip install -e . --no-build-isolation
```

5. Make sure the build worked.

```powershell
python -c "import MLCodec_extensions_cpp; print('entropy extension ok')"
```

The output must contain `entropy extension ok`. Every encode and decode call uses this extension.
If the check fails, go to section 10.

## 5. Get the model

The model is not in the repository. It is 84 MB.

1. Download the released `pulse-mse` bundle from <https://huggingface.co/zhaoyangjia/PULSE>.

2. Copy three files into `checkpoints/pulse-mse/`.

```
checkpoints/pulse-mse/
    config.json
    model.pt
    entropy_control_int.pt      <- you must have this file
```

3. Make sure `entropy_control_int.pt` is present. This file holds the integer cumulative distribution
   function (CDF) tables and the scale decoder. The loader stops if the file is absent. This behaviour
   is deliberate. Without the file, the codec makes a different bitstream than the published model.

## 6. Get the data

| Set | Images | Folder | Purpose |
|---|---|---|---|
| Kodak | 24 | `datasets/kodak/` | Screening |
| CLIC Professional | 41 | `datasets/clic_professional/` | Confirmation |
| Set14 | 14 | `datasets/set14/` | Transfer check. Never used for fitting. |
| DIV2K validation | 100 | `datasets/qat_pool/div2k_valid/` | Fine-tuning only |

Images are not in the repository. Download Kodak, CLIC Professional and Set14 from their usual
sources.

To get the fine-tuning pool, run this command.

```powershell
.\venv\Scripts\python.exe tools\fetch_qat_pool.py --source div2k_valid
```

The command downloads DIV2K validation, extracts it and writes a SHA-256 manifest. The command refuses
to build a pool from an evaluation set. Keep this behaviour. A pool that contains test images makes
all downstream numbers invalid.

## 7. Check the installation

1. Run the unit tests.

```powershell
.\venv\Scripts\python.exe -m unittest discover -s quantization-pluse-mse\tests
```

Expected output: `Ran 10 tests` and `OK`.

2. Run one image at one quality setting.

```powershell
.\venv\Scripts\python.exe evaluate.py --model checkpoints/pulse-mse `
  --input datasets/kodak/kodim01.png --output results\smoke.json `
  --qps 3 --device cuda --threads 1
```

Expected result: `smoke.json` holds a finite `bpp` and a finite `psnr`.
A `psnr` near 4 dB is wrong. The cause is the model bundle or the entropy extension. The cause is not
the code.

## 8. Reproduce the reported numbers

### 8.1 The headline result

Run this command for each dataset. Use `kodak`, `clic_professional` and `set14`.

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\eval_assignment.py `
  --model checkpoints/pulse-mse `
  --configs quantization-pluse-mse\results\ilp\assignment_ilp.json `
  --dataset kodak --output results\check_kodak
```

The command writes four files.

| File | Contents |
|---|---|
| `sweep_<dataset>.csv` | One row for each image and quality setting |
| `configs_<dataset>.csv` | One row for each configuration: size, saving, worst penalty |
| `summary_<dataset>.csv` | Mean values for each quality setting |
| `manifest_<dataset>.json` | Run metadata |

Compare your result with the report. Open `results\check_kodak\configs_kodak.csv` and find the `ilp` row.

| Value | Expected |
|---|---|
| `quant_mb` | 21.677 |
| `worst_qp_mean_penalty_pct` | 0.672 |

A difference of 0.02 or less is normal. A larger difference means a different model bundle or a
different assignment file.

This command also runs the FP32 gate. The gate reproduces the reference bitstream and the
reconstruction hashes at all eight quality settings. The gate stops the run if the baseline is wrong.

### 8.2 The other operating points

Run this command. It takes less than 1 minute.

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\solve_ilp.py --budget 1.0 1.8 --headline 1.0
```

The command prints the predicted size and the predicted penalty for each budget. It writes
`results/ilp/assignment_ilp.json`.

| Budget | Size | Predicted penalty |
|---|---|---|
| 1.0% | 21.677 MB | 0.9715% |
| 1.8% | 18.664 MB | 1.7162% |

The prediction is always conservative. The measured value is better on every dataset.

To evaluate a different operating point, give its file to `eval_assignment.py` with `--configs`.

### 8.3 The sensitivity sweep

You need this step only if you change the model or the set of formats. All later steps use its output.
It takes about 2.5 hours for each format.

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\pipeline.py inventory
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\pipeline.py sweep --format int8
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\analyze_additivity.py
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\analyze_comparison.py --config ilp
```

The raw ledger is about 85 MB. It is not in the repository. These commands make it again.

### 8.4 Fine-tune one tier

This step needs the GPU. It takes about 100 minutes for one tier.

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\qat_overnight.py `
  --tier tier1_more_int8 --steps 12000 --objective augmented_lagrangian `
  --eval-every 250 --datasets kodak clic_professional set14
```

The tiers are in `quantization-pluse-mse/configs/qat_tiers.json`. Each run writes a loadable bundle
and a `summary.csv` with the penalty before and after.

Keep two properties if you change this code.

- The saved checkpoint must be better than its start. If it is not, the code keeps the start weights.
  This property prevents a bad run from shipping a worse model.
- The entropy parts must stay frozen. If they move, the integer CDF tables no longer agree with the
  weights.

### 8.5 Rebuild the documents

1. Get the TeX engine.

```powershell
.\venv\Scripts\python.exe tools\fetch_tectonic.py
```

2. Set the TeX cache directory. The default directory is outside the workspace, and the build fails
   there.

```powershell
$env:TECTONIC_CACHE_DIR = "$PWD\.scratch\tectonic-cache"
```

3. Build the 2-page report.

```powershell
.\tools\tectonic\tectonic.exe submission\report.tex
```

4. Build the supplementary report and the slide deck. These commands need `python-docx` and
   `python-pptx`.

```powershell
python quantization-pluse-mse\scripts\make_supplementary.py
python quantization-pluse-mse\scripts\make_deck.py
```

## 9. Expected runtimes

| Task | Time |
|---|---|
| Unit tests | 2 seconds |
| One image, one quality setting | Less than 1 second |
| Evaluate one assignment on Kodak | About 2 minutes |
| Evaluate one assignment on CLIC | About 4 minutes |
| Evaluate one assignment on Set14 | About 1.5 minutes |
| ILP solve, one budget | Less than 1 second |
| Sensitivity sweep, one format, two datasets | About 2.5 hours |
| Fine-tune 12,000 steps, one tier | About 100 minutes |
| Rebuild the 2-page report | About 10 seconds |

## 10. Troubleshooting

**The message `ModuleNotFoundError: MLCodec_extensions_cpp` appears.**
The native extension did not build. Run `pip install -e . --no-build-isolation` again. Read the
compiler output. On Windows, you must install MSVC Build Tools. The error message names the missing
tool.

**The message `model.pt and entropy_control_int.pt are from different bundles` appears.**
The bundle is not consistent. You mixed files from two downloads, or you put a fine-tuned `model.pt`
next to an `entropy_control_int.pt` from a different checkpoint. Download the bundle again. Or run the
fine-tuning export again, which rebinds the hash.

**The message `FileNotFoundError: .../entropy_control_int.pt` appears.**
You downloaded `model.pt` only. Download the full bundle.

**`Invoke-WebRequest` or `curl` fails with `SEC_E_CREDENTIALS`.**
Windows schannel is broken on this machine. Use Python `urllib`. All download scripts here use it.

**The message `CUDA out of memory` appears.**
Use `--max-images` to reduce the work. Or evaluate one dataset at a time. One 512-pixel fine-tuning run
needs about 3.8 GB. Close other programs that use the GPU.

**A benchmark result is 25% slower than usual.**
Look for an orphaned `python.exe` process. One orphan process caused this error in one of our runs.
End the process, then benchmark again.

**Your numbers differ from the report by more than 1 point.**
Check three things in this order. Check the model hash. Check that the FP32 gate passed. Check that
both runs use the same assignment file. The penalty metric is sensitive to a wrong baseline.

**A table in the report looks correct, but a number is wrong.**
Every table in the report names its source file in the caption. Use the CSV file as the authority.
Build the document again.

## 11. Where to look next

| Question | File |
|---|---|
| What was measured, and what was learned? | `quantization-pluse-mse/STATUS.md` |
| Which file proves a claim? | `quantization-pluse-mse/RESOURCES.md` |
| Why these parts and not other parts? | `quantization-pluse-mse/results/qat/WHY_THESE_COMPONENTS.md` |
| The graded report | `submission/report.pdf` |
| Every measured configuration | `quantization-pluse-mse/results/mixed/` |
