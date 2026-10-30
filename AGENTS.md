# AGENTS.md - PULSE Repository Guide

## Purpose

This repository contains the PULSE neural image codec and a course project on
component-wise quantization of the released `pulse-mse` model. This file is the
entry point for any agent working here.

## Read order

1. `AGENTS.md` (this file)
2. `README.md`
3. `quantization-pluse-mse/README.md`
4. `quantization-pluse-mse/quantization.md`
5. `quantization-pluse-mse/TODO.md`
6. `ILP-mixed-precision/ilp.md` (only for ILP work)
7. `docs/deployment.md` and `docs/training.md` when relevant

## Workflow contract

The user prefers:

1. **Plan before editing the codebase.**
   - State the exact files to change, the change, and the verification.
   - Wait for approval before editing code.
2. **Small, testable steps.**
   - Edit the smallest unit.
   - Run the smallest relevant test.
   - Report the exact command and result.
3. **Confirm before expanding.**
   - Do not start a full sweep, QAT, ILP solve, or deployment build without
     explicit approval.
4. **Short, crisp, knowledge-dense responses.**
   - No preamble, filler, repetition, or restating the request.
   - Include exact paths, commands, numbers, and trade-offs.
   - Prefer bullets and tables over paragraphs.
   - No emojis unless requested.

## Project map

| Path | Meaning |
|---|---|
| `codec/` | Core model, entropy coding, runtime, checkpoint I/O |
| `configs/` | Training recipes, not deployment configs |
| `deployment/` | Platform runtimes: AMD CPU, H100, Snapdragon NPU |
| `tools/` | Data and teacher export utilities |
| `checkpoints/pulse-mse/` | The model under test |
| `datasets/kodak_thumb/` | 24-image 150x150 thumbnail smoke set |
| `datasets/kodak/` | 24-image full-resolution Kodak set (PNG, present) |
| `datasets/clic_professional/` | 41-image confirmation set |
| `benchmark_results/` | Existing FP32 baseline artifacts |
| `quantization-pluse-mse/` | Component-wise precision study workspace |
| `ILP-mixed-precision/` | Mixed-precision selection and presentation workspace |
| `ILP-mixed-precision/presentation/` | ILP slide outline and draft slide text |
| `artifacts/` | Paper, proposal deck, and reference image |

## Runtime

| Item | Value |
|---|---|
| Shell | Windows PowerShell 5.1 |
| Python | 3.12.6 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| GPU | RTX 3050 Laptop, 4 GB, `sm_86` |
| Venv | `.\venv\Scripts\python.exe` |

Activate with `.\venv\Scripts\Activate.ps1`.

Rebuild the native entropy extension with:

```powershell
.\venv\Scripts\python.exe -m pip install -e . --no-build-isolation
.\venv\Scripts\python.exe -c "import MLCodec_extensions_cpp; print('ok')"
```

## Model facts

- `pulse-mse`, base, MSE objective.
- `M=320`, `z_ch=96`, hidden 144, renderer 48, 2 decoder DCBs,
  entropy channels 128, `dico_block_variant=ca`.
- 14,179,432 neural parameters.
- 540,672 extra Meta Prior lookup parameters in the loaded model.
- Loaded total: 14,720,104.
- Conv weights: 14,127,360 (99.6% of neural parameters).
- Encoder 12,133,696 (85.6%), entropy 1,337,224 (9.4%),
  decoder 708,504 (5.0%).
- Eight QPs, 0-7.
- `entropy_model.scale_dec` is already integer in the released bundle.
- `entropy_control_int.pt` holds the integer scale decoder, CDFs, and side tables.
- Meta Prior and `entropy_bottleneck` are lookup/table-like, not float networks.
- `q_scale_*`, `q_basic`, `qp_bias`, and `position_bias` are tiny controls.

## Current state

- FP32 baseline reproduced on the 150x150 Kodak thumbnails and CLIC at all QPs.
- Entropy extension rebuilt and smoke-tested on 2026-10-20:
  `kodim01t.jpg`, QP 3, 0.6201 bpp, 30.556 dB, 89.4 ms decode.
- FP32 reference re-run through the harness on `kodak_thumb`, full-resolution `kodak`,
  and `clic_professional` (three passes each); G0a and noise floor recorded.
- Full-resolution Kodak PNGs are present: 24 files, 18 x 768x512 and 6 x 512x768.
- **Main weight-only sweep is complete** (`2026-10-24T12:47Z`): 7 formats x 19 components
  x {`kodak` 24, `clic_professional` 41} x 8 QPs = 69,233 rows, 135 configs, 0 failures,
  `restored_ok` 135/135. Formats: FP16, BF16, FP8-E4M3, FP8-E5M2, INT8, INT4, FP4-E2M1.
- Every config in the ledger is `w_only`; activations are not quantized yet.
- Per-precision review surfaces exist for all 7 formats, 6 figures each in
  `quantization-pluse-mse/{fmt}/results/figures/`.
- Analysis rebuilt `2026-10-25T01:29+05:30` in `quantization-pluse-mse/results/analysis/`:
  component x precision table, 2 heatmaps, `recommendations.csv`.
- Headline verdicts: 13/19 components green at INT8 (70.4% of listed params, 70.8% of
  kMAC/px); Red are H0, H3, D5 (3.8% of params, 3.0% of kMAC/px).
- INT8 penalty is QP-dependent in both directions: encoder/decoder damage grows with QP
  (E0 0.154% -> 1.174%, 7.6x) while entropy damage falls (H3 38.152% -> 2.768%, 13.8x).
- FP8-E4M3 is per-output-channel scaled and still loses to per-channel INT8
  (E1 14.8% vs 1.35%): 3 mantissa bits < 8-bit uniform.
- INT4/FP4 are per-output-channel only; group/block scaling is not implemented, so the
  4-bit columns are a lower bound on achievable 4-bit quality.
- Missing: whole-model multi-component configs, additivity, W8A8, calibration set,
  BD-rate, ILP solver, latency/energy bench, report, slides.
- `pandas`, `matplotlib`, `seaborn`, `pyarrow`, `ortools`, `onnx`, `onnxruntime`,
  `tensorrt`, and `pytest` are absent. The analysis scripts are stdlib + PIL only, so
  `pandas`/`matplotlib` are not required; `ortools` is required for ILP.
- Calibration and QAT data still missing. QAT is not triggered by rule: no component that
  is >= 5% of MACs or params is Red at INT8.

## Quantization rules

1. Quantize exactly one component at a time; keep everything else FP32.
2. Start with weight-only, then weight + dynamic activations, then weight +
   calibrated activations.
3. FP16, BF16, INT8 are real on this GPU; FP8, FP4, INT4 are simulated.
4. No latency, memory, or energy claim for simulated formats.
5. Keep `entropy_model.scale_dec` as an integer control; do not fake-quantize it.
6. Exclude `entropy_model.meta_prior` and `entropy_model.entropy_bottleneck`
   from weight quantization.
7. Keep `q_scale_*`, `q_basic`, `qp_bias`, `position_bias`, norms, and biases
   FP32 in the main sweep.
8. Use `datasets/kodak_thumb` for smoke and G0a only.
   Use full-resolution `datasets/kodak` and CLIC Professional for the real quantization loop.
9. Never use Kodak or CLIC validation for calibration or QAT.
10. Keep the FP32 baseline JSON unchanged; write new results under
    `quantization-pluse-mse/results/`.
11. Every result must record component, descendants, precision, granularity,
    quant mode, calibration, dataset, QP, model hash, git commit, and command.
12. If a bitstream or CDF mismatch occurs, record it; do not hide it.
13. Run the FP32 reference first, all 8 QPs, before any INT8 run.
14. D1 cannot be declared green without controls: identity stream hashes match,
    INT4 `qmax=7` hurts more than INT8, weight hashes change then restore, and
    output changes on a fixed input.
15. Mutate in place and restore from a cached FP32 clone; do not build a second
    quantized model. No per-run `empty_cache()`.
16. Log weight hashes, not `model_deleted`; `del` and `gc.collect()` prove nothing.
17. Resume key is `(config_id, dataset, image, qp)`, where `config_id` is the
    quant-config hash prefix; never include `run_id`.
18. Smoke images are the first 8 `kodak_thumb` files in sorted order, identical for
    reference and quantized runs.
19. Keep the raw append-only ledger in `quantization-pluse-mse/results/_ledger/`.
20. The review surface for a precision is `quantization-pluse-mse/{precision}/results/`.
    Use the three-table split: `sweep_<fmt>.csv` per image/QP, `configs_<fmt>.csv`
    per config assertion and analytic size, `weight_stats_<fmt>.csv` per layer.
    Keep it compact: one table and one all-component line plot per relative metric
    (d_PSNR, d_bpp, penalty, bpp, PSNR), plus one size table/plot.
    Do not generate per-component or per-dataset plot families.
21. Source code stays universal in `quantization-pluse-mse/src/` and `scripts/`;
    do not duplicate per-precision code. Use the single entry point
    `scripts/pipeline.py` with subcommands `inventory`, `reference`, `sweep`, `export`.

## ILP rules

- Run ILP only after the PTQ sweep produces coefficients.
- Variables: component x precision x QP.
- Objective: minimize latency or a weighted latency/energy cost.
- Constraints: `delta_psnr <= epsilon`, memory <= 4 GB, supported formats only,
  one precision per component.
- Use OR-Tools CP-SAT first; Gurobi only if available.
- Validate the mixed assignment with the same protocol as the sweep.
- Do not invent missing coefficients.
- Do not claim ILP speedups from simulated formats.

## Do not

- Do not edit code without a plan and approval.
- Do not run full sweeps, QAT, ILP, or deployment builds without approval.
- Do not overwrite baseline artifacts.
- Do not change the entropy CDF contract silently.
- Do not quantize lookup tables or scalar QP gates in the main sweep.
- Do not use test images for calibration or fine-tuning.
- Do not train the full model on the 4 GB laptop.
- Do not add comments unless asked.
- Do not commit unless explicitly asked.
- Do not add secrets or keys.
- Do not use emojis.

## Verification

Fast smoke test:

```powershell
.\venv\Scripts\python.exe evaluate.py --model checkpoints/pulse-mse `
  --input datasets/kodak_thumb/kodim01t.jpg `
  --output C:\Users\ASUS\AppData\Local\Temp\opencode\smoke.json `
  --qps 3 --device cuda --threads 1
```

Expected: JSON row with finite bpp and PSNR, no exception.

For quantization changes, run the one-component, one-precision, one-QP test
first, then report the exact command and result.

## Git

- Inspect `git status`, `git diff`, and `git log --oneline -10` before any commit.
- Stage only intended files.
- Do not commit checkpoints, datasets, results, or generated binaries unless
  explicitly requested.
