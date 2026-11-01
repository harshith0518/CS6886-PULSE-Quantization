# Runbook

How to set this project up and reproduce every number in the report. Written so that someone who has
never seen the repository can get from a fresh clone to a verified result without guessing.

Estimated total time: **15 minutes** to a first verified result, **~2 hours** to reproduce the full
evaluated frontier, **~1 day** to regenerate everything from raw measurements.

---

## 1. What you need

| Item | Requirement | Notes |
|---|---|---|
| OS | Windows 10/11 (tested) or Linux | Commands below are PowerShell; bash equivalents are the same minus the `.exe` |
| Python | **3.12** | 3.11 probably works; 3.13 is untested |
| GPU | **4 GB VRAM minimum** | An RTX 3050 Laptop (4 GB) ran everything here |
| CUDA | **12.4** (matching `torch 2.6.0+cu124`) | CPU-only works for evaluation, just slower |
| Disk | **~10 GB free** | Checkpoint 84 MB, datasets 0.5 GB, results ~1 GB, checkpoints written by QAT 84 MB each |
| Build tools | MSVC Build Tools on Windows | Needed once, to compile the entropy extension |

The CPU path is fine for evaluation and for the whole-model measurements; only fine-tuning really wants
the GPU.

## 2. Check out

```powershell
git clone https://github.com/harshith0518/CS6886-PULSE-Quantization.git
cd CS6886-PULSE-Quantization
```

## 3. Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

The last command compiles the native entropy extension. **Verify it before going further:**

```powershell
python -c "import MLCodec_extensions_cpp; print('entropy extension ok')"
```

If this fails, nothing else will work: every encode and decode call goes through that extension. See
Troubleshooting §9.

## 4. Get the checkpoint

The 84 MB model is **not tracked in git**. Download the released `pulse-mse` bundle from
<https://huggingface.co/zhaoyangjia/PULSE> and place three files here:

```
checkpoints/pulse-mse/
├── config.json
├── model.pt
└── entropy_control_int.pt      <- mandatory
```

`entropy_control_int.pt` holds the integer CDF tables and the scale decoder. The loader **refuses** a
bundle without it, deliberately: running without it would silently produce a different bitstream than
the published model.

## 5. Datasets

| Set | Images | Where |
|---|---|---|
| Kodak | 24 | `datasets/kodak/` — screening set |
| CLIC Professional | 41 | `datasets/clic_professional/` — confirmation set |
| Set14 | 14 | `datasets/set14/` — transfer set, never used for fitting |
| DIV2K validation | 100 | `datasets/qat_pool/div2k_valid/` — fine-tuning pool only |

Images are **not tracked**. Kodak and Set14 are small and standard; fetch them from their usual
sources, or reproduce the fine-tuning pool with:

```powershell
.\venv\Scripts\python.exe tools\fetch_qat_pool.py --source div2k_valid
```

That script downloads DIV2K validation, extracts it, writes a SHA-256 manifest, and **refuses to build
a pool from any evaluation set**. Keep that property if you modify it: a pool that overlaps a test set
makes every downstream number meaningless.

## 6. Verify the install (2 minutes)

```powershell
# 10 unit tests over the quantization core
.\venv\Scripts\python.exe -m unittest discover -s quantization-pluse-mse\tests

# one image, one quality setting, end to end
.\venv\Scripts\python.exe evaluate.py --model checkpoints/pulse-mse `
  --input datasets/kodak/kodim01.png --output results\smoke.json `
  --qps 3 --device cuda --threads 1
```

Expected: `tests` reports `OK`, and `smoke.json` contains a finite `bpp` and `psnr`. If `bpp` is 0 or
`psnr` is around 4 dB, the checkpoint or the entropy extension is wrong, not the code.

## 7. Reproduce the reported numbers

### 7.1 The headline frontier (~2 min per dataset)

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\eval_assignment.py `
  --model checkpoints/pulse-mse `
  --configs quantization-pluse-mse\results\ilp\assignment_ilp.json `
  --dataset kodak --output results\check_kodak
```

Run it once per dataset (`kodak`, `clic_professional`, `set14`). Each run writes:

| File | Contents |
|---|---|
| `sweep_<dataset>.csv` | one row per image per quality setting |
| `configs_<dataset>.csv` | one row per configuration: size, saving, worst-QP penalty |
| `summary_<dataset>.csv` | per-quality-setting aggregates |
| `manifest_<dataset>.json` | run metadata |

**Expected** (`configs_kodak.csv`, `ilp` row): `quant_mb = 21.677`, `worst_qp_mean_penalty_pct =
0.672`. Allow ±0.02 absolute for platform float differences; anything larger means a different
checkpoint or a modified assignment.

The FP32 gate runs inside this step. It reproduces the reference bitstream and reconstruction hashes at
all eight quality settings and will stop the run rather than let you compare against a wrong baseline.

### 7.2 The alternative operating points (< 1 min)

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\solve_ilp.py --budget 1.0 1.8 --headline 1.0
```

Prints the predicted size and penalty per budget and writes `results/ilp/assignment_ilp.json`. Expected:
1.0% → 21.677 MB / 0.9715% predicted; 1.8% → 18.664 MB / 1.7162% predicted. Predictions err
**conservative** — measurement beats them on every dataset.

Evaluate a variant by passing its assignment file to `eval_assignment.py --configs`.

### 7.3 The sensitivity sweep (hours — usually unnecessary)

Only needed if you change the model or the format set. Everything downstream depends on it.

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\pipeline.py inventory
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\pipeline.py sweep --format int8
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\analyze_additivity.py
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\analyze_comparison.py --config ilp
```

~2.5 hours per format for both datasets on a 4 GB GPU. The raw ledger (~85 MB) is gitignored and
regenerated here.

### 7.4 Fine-tuning a tier (GPU, ~100 min per tier)

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\qat_overnight.py `
  --tier tier1_more_int8 --steps 12000 --objective augmented_lagrangian `
  --eval-every 250 --datasets kodak clic_professional set14
```

Tiers are defined in `quantization-pluse-mse/configs/qat_tiers.json`. Each run writes a loadable
bundle plus `summary.csv` with before/after penalties.

Two properties to preserve if you modify this code:

- **The saved checkpoint must beat its own starting point**, or the starting weights are kept. This is
  why a bad run cannot ship a worse model.
- **Entropy components stay frozen.** Moving them invalidates the integer CDF tables in the bundle.

### 7.5 Rebuild the documents

```powershell
# 2-page LaTeX report (Tectonic is a single binary; fetch it if absent)
.\venv\Scripts\python.exe tools\fetch_tectonic.py
$env:TECTONIC_CACHE_DIR = "$PWD\.scratch\tectonic-cache"
.\tools\tectonic\tectonic.exe submission\report.tex

# longer supplementary report and the slide deck (need python-docx / python-pptx)
python quantization-pluse-mse\scripts\make_report.py
python quantization-pluse-mse\scripts\make_deck.py
```

The TeX cache **must** point inside the working directory on this machine: the default location is
outside the sandbox and package extraction is denied there.

## 8. Expected runtimes

| Task | Time on an RTX 3050 Laptop (4 GB) |
|---|---|
| Unit tests | 2 s |
| One image, one setting | < 1 s |
| Evaluate one assignment, Kodak | ~2 min |
| Evaluate one assignment, CLIC | ~4 min |
| Evaluate one assignment, Set14 | ~1.5 min |
| ILP solve, one budget | < 1 s |
| Sensitivity sweep, one format, both datasets | ~2.5 h |
| QAT, 12,000 steps, one tier | ~100 min |
| Rebuild the 2-page report | ~10 s (after the first package download) |

## 9. Troubleshooting

**`ModuleNotFoundError: MLCodec_extensions_cpp`** — the native extension did not build. Re-run
`pip install -e . --no-build-isolation` and read the compiler output. On Windows you need MSVC Build
Tools; the error names the missing toolset.

**`model.pt and entropy_control_int.pt are from different bundles`** — the bundle is inconsistent.
Either you mixed files from two downloads, or a fine-tuned `model.pt` was placed beside an
`entropy_control_int.pt` built from a different checkpoint. Re-download, or re-run the QAT export
which rebinds the hash after verifying the entropy tensors did not move.

**`FileNotFoundError: .../entropy_control_int.pt`** — download the full bundle, not just `model.pt`.

**PowerShell `Invoke-WebRequest` / `curl` fails with `SEC_E_CREDENTIALS`** — Windows schannel is broken
on this machine. Use Python's `urllib` (all download scripts here do) rather than another shell tool.

**CUDA out of memory** — reduce `--max-images`, or evaluate dataset by dataset. Fine-tuning a 512-crop
tier peaks around 3.8 GB; nothing will fit if another process holds the GPU. Check for orphaned
`python.exe` before benchmarking (it silently inflated one run by 25%).

**Numbers disagree by a lot (> 1 absolute point)** — check, in order: the checkpoint hash, that the
FP32 gate passed, and that you are comparing the same assignment file. The penalty metric is sensitive
to a wrong baseline.

**A table looks fine but a number is wrong** — every result table in the report names its source file
in the caption. Trust the CSV over the document and regenerate.

## 10. Where to look next

| Question | File |
|---|---|
| What exactly was measured, and what was learned? | `quantization-pluse-mse/STATUS.md` |
| Which file proves a specific claim? | `quantization-pluse-mse/RESOURCES.md` |
| Why these components and not others? | `quantization-pluse-mse/results/qat/WHY_THESE_COMPONENTS.md` |
| The 2-page submission | `submission/report.pdf` |
| Every measured configuration | `quantization-pluse-mse/results/mixed/` |
