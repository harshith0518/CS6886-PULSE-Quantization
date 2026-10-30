# Resources and provenance

Every number in this workspace is produced by one of the scripts below from one of the data
sources below. This file lists what was used, and how confident we can be in each source.

## Verification legend

| Tag | Meaning |
|---|---|
| **[V]** | Verified in this session by reading the artefact or extracting its content |
| **[E]** | Verified by execution: the library or API was actually called and produced values |
| **[R]** | Standard reference from the literature; **not machine-verified in this session** |

Note on **[R]**: web search was unavailable in this environment (the search tool has no API key),
so the external references below are cited from established knowledge and were not re-checked
against a live index. Anyone writing this up for submission should confirm them before a
bibliography is final.

## 1. The model and its paper

| Resource | Detail | Status |
|---|---|---|
| `artifacts/2605.04563v1.pdf` | *PULSE: Unlocking Practical Image Compression on Single-Thread CPU*, Zhaoyang Jia, Tianyu Zhang, Zihan Zheng, Wenxuan Xie, Jiahao Li, Bin Li, Houqiang Li, Yan Lu. USTC and Microsoft Research Asia. 22 pages. arXiv 2605.04563v1 | [V] |
| Paper claims used | 5.2 kMAC/pixel neural receiver; integer linear CDF predictor plus meta prior for bit-exact entropy coding; 1080p decode in 126 ms on one CPU thread; compression comparable to HM | [V] |
| Upstream code | <https://github.com/microsoft/GenCodec/tree/main/PULSE> | [V] (stated in the paper) |
| `checkpoints/pulse-mse` | `config.json`, `model.pt`, `entropy_control_int.pt`. Model SHA-256 `c6026ee7...930ce5ce`, recorded as `base_model_hash` in every config row | [V] |
| Parameter counts | 14,179,432 neural; 14,720,104 loaded with the Meta Prior table; 19 main-sweep components cover 13,671,984 (96.4%) | [V] `artifacts/component_inventory.json` |

**Consistency check worth keeping.** The paper's 5.2 kMAC/pixel describes the *receiver*
(decoder) only. Our inventory reports 48.75 kMAC/px summed over all 19 components, which includes
the encoder that holds 85.6% of the parameters. These are not contradictory; do not compare them
as if they measured the same thing.

## 2. Data resources

Every evaluation set is gitignored and re-downloadable; provenance is committed.

| Resource | Contents | Provenance file | Status |
|---|---|---|---|
| `datasets/kodak/` | 24 PNG, 18x768x512 and 6x512x768 | `datasets/kodak/README.md` | [V] |
| `datasets/clic_professional/` | 41 PNG | upstream CLIC set | [V] |
| `datasets/set14/` | 14 PNG, 250x360 to 768x512, one grayscale | `datasets/set14/README.md` + `manifest.json` with per-file SHA-256 | [V] |
| `datasets/kodak_thumb/` | 24 JPEG at 150x150, smoke only | — | [V] |
| `benchmark_results/pulse_mse_20261011_*.json` | frozen FP32 baseline used for G0a | — | [V] |
| `quantization-pluse-mse/results/_ledger/reference/*/` | FP32 reference: `pass1`, `pass2`, `tf32_on`, `curve.json`, `noise_floor.json` per dataset | — | [V] |

Set14 was fetched on 2026-10-30 from `jbhuang0604/SelfExSR` via Python `urllib`, because Windows
schannel is broken in this environment (`SEC_E_NO_CREDENTIALS`) while Python's TLS stack works.
Per-file SHA-256 is in `datasets/set14/manifest.json` so the download is auditable.

## 3. Methods and where they come from

| Method | Used in | Source | Status |
|---|---|---|---|
| Rate-equivalent penalty `100*((1+d_bpp/100)*2^(-d_psnr/slope)-1)` | all penalty and budget numbers | **defined in this repository**, `scripts/pipeline.py:598`. Not a published metric | [V] |
| Bjøntegaard delta-rate | `scripts/analyze_comparison.py` | G. Bjøntegaard, "Calculation of average coding efficiency based on subjective quality scores" (2001); the BD-PSNR extension is the VCEG-AI11 contribution (2008) | [R] |
| Monotone cubic (PCHIP) interpolation | BD-rate curves | Fritsch and Carlson, "Monotone piecewise cubic interpolation", SIAM J. Numer. Anal. 17(2), 1980 | [R] |
| `scipy.interpolate.PchipInterpolator` | BD-rate curves | SciPy 1.18.1 | [E] |
| Mixed-integer linear programming | `scripts/solve_ilp.py` | HiGHS: Huangfu and Hall, "Parallelizing the dual revised simplex method", Math. Prog. Computation 10(1), 2018 | [R] |
| `scipy.optimize.milp` | solver, no extra dependency | SciPy 1.18.1 wrapping HiGHS | [E] |
| Per-output-channel symmetric weight quantization | `pulsequant/core.py` | Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR 2018 | [R] |
| Straight-through estimator (proposed QAT only, not run) | not implemented | Bengio, Léonard, Courville, "Estimating or propagating gradients through stochastic neurons", 2013 | [R] |
| NVML total-energy counter and power sampling | `scripts/bench_configs.py` | NVIDIA Management Library, `nvmlDeviceGetTotalEnergyConsumption`, `nvmlDeviceGetPowerUsage` | [E] |
| Simulated-format methodology: no speed or energy claim for fake quantization | `results/mixed/COMPARISON.md` | this repository's own rule, `AGENTS.md` rule 4 | [V] |

## 4. Claim to proof map

Each headline statement and the artefact that demonstrates it.

| Claim | Value | Proof file |
|---|---|---|
| Sweep completeness | 69,233 rows, 135 configs, 0 failures | `results/_ledger/sweep.csv`, `results/_ledger/configs.csv` (`restored_ok` 135/135) |
| FP32 reference is reproducible | bit-identical stream and recon at 8 QPs | `results/_ledger/reference/*/noise_floor.json`, `results/mixed/gate_*.json` |
| 19 components inventoried | 13,671,984 params, 48.75 kMAC/px | `artifacts/component_inventory.json` |
| 13/19 tolerate INT8 | 70.4% of params, 70.8% of kMAC/px | `results/decision/decision_summary.csv`, `decision_component_ranking.png` |
| Fragile set is small | H0, H3, D5 = 3.8% of params, 3.0% of kMAC/px | `results/decision/decision_summary.csv` |
| INT8 sensitivity is QP-dependent both ways | E0 x7.6 up, H3 x13.8 down | `results/decision/decision_qp_sensitivity.png`, `int8/results/summary.csv` |
| FP8 loses to INT8 at equal bits | E1 14.8% vs 1.35% | `fp8_e4m3/results/summary.csv`, `int8/results/summary.csv` |
| Additivity holds within ~5% | measured/predicted 0.96-1.05 Kodak, 1.02-1.11 CLIC | `results/mixed/additivity.csv` |
| Model is 60.4% smaller | 54.688 to 21.677 MB | `results/mixed/configs_*.csv`, `results/ilp/assignment_ilp.json` |
| Quality cost is small | BD-rate +0.477 / +0.552 / +0.486% | `results/mixed/bd_rate.csv`, `comparison_bd_rate.png` |
| Fewer bits at high QP | -0.42% at QP7 Kodak | `results/mixed/comparison_ilp_vs_fp32.csv`, `per_qp_tables.md` |
| Transfers to an unseen set | Set14 BD-rate +0.486%, penalty 0.935% | `results/mixed/*_set14.csv` |
| ILP beats greedy on size, loses on rate | 21.677 vs 24.565 MB; 0.48% vs 0.30% | `results/ilp/solve.csv`, `results/mixed/bd_rate.csv` |
| QAT is not triggered by rule | no red component >= 5% of MACs/params | `results/decision/decision_summary.csv`, `results/decision/README.md` |
| Latency, energy and memory | paired interleaved benchmark, see below | `results/metrics/interleaved_paired_kodak.csv` |
| Per-QP detail for every config | 7 configs x 8 QPs x 3 datasets | `results/mixed/per_qp_tables.md` and `per_qp_<metric>_<dataset>.csv` |

## 5. What is not backed, and must not be claimed

| Not supported | Why |
|---|---|
| Any speedup or energy saving from INT8, FP8, FP4 or any fake-quant configuration | those formats are simulated on `sm_86`; the weights stay FP32 and the kernels are unchanged |
| A real INT8 deployment | no TensorRT or ONNX Runtime path has been built |
| 4-bit quality as a verdict on 4-bit in general | INT4/FP4 are per-output-channel only; group and block scaling are not implemented |
| Calibrated activation quantization, W8A8 | not implemented |
| Generalisation beyond three small datasets | Kodak 24, CLIC 41, Set14 14; Set14 is a transfer check, not a peer benchmark |
| Perceptual quality | only MSE/PSNR was measured, no LPIPS or other perceptual metric was run |
| The `[R]` citations above | not machine-verified in this session |

## 6. Reproduction chain

```powershell
# 1. FP32 reference per dataset (writes results/_ledger/reference/<dataset>/)
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\pipeline.py reference --dataset kodak --qps 0 1 2 3 4 5 6 7

# 2. single-component sweep (writes the ledger)
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\pipeline.py sweep --format int8 --components all --dataset kodak

# 3. decision surface and candidate assignments
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\visualize_decision.py

# 4. mixed-precision solve
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\solve_ilp.py

# 5. whole-model evaluation on every dataset
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\eval_assignment.py --dataset kodak --configs fp32 fp16 bf16 int8 safe greedy ilp

# 6. analysis
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\analyze_additivity.py
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\analyze_comparison.py
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\report_per_qp.py
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\bench_configs.py --dataset kodak
```

Environment: Python 3.12.6, PyTorch 2.6.0+cu124, CUDA 12.4, RTX 3050 Laptop 4 GB (`sm_86`),
scipy 1.18.1, nvidia-ml-py 13.615.71, Pillow. No pandas or matplotlib is required; every table and
figure is stdlib plus Pillow.
