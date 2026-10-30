# PULSE-MSE Quantization Project - TODO and Progress (v3.0)

> Plan: `quantization.md` (v2.1). Model: `checkpoints/pulse-mse`. Hardware: RTX 3050 Laptop 4 GB, CUDA 12.4, PyTorch 2.6.0+cu124.
> Evaluation sets: `datasets/kodak_thumb` (24, smoke/G0a), `datasets/kodak` (24, full-res PNG), `datasets/clic_professional` (41, confirm).
> Dataset status: thumbnails verified 24 x 150x150 JPEG; full-resolution Kodak verified 24 PNGs (18 x 768x512, 6 x 512x768); CLIC 41 images.
> Sweep status: **complete** `2026-10-24T12:47Z`. 7 formats x 19 components x {`kodak`, `clic_professional`} x 8 QPs = 69,233 rows, 135 configs, 0 failures, `restored_ok` 135/135.
> Analysis status: rebuilt `2026-10-25T01:29+05:30` into `results/analysis/`.
> Minimum viable deliverable (MVD): WP1 + WP4 (FP16 and INT8 columns) + whole-model INT8 + time breakdown.
> Next phase is **v3**: visualizations -> QAT/greedy/ILP decision -> mixed-precision model -> 3-dataset evaluation against baseline.

## v3.0 amendments

- [x] Full-res Kodak PNGs present; the "pending download" blocker is closed.
- [x] Main weight-only sweep complete for all 7 formats on Kodak and CLIC.
- [x] Extra formats FP16, BF16, FP8-E4M3, FP8-E5M2, FP4-E2M1 implemented and swept.
- [x] Per-precision review surfaces for all 7 formats, 6 figures each in `{fmt}/results/figures/`.
- [x] Component x precision analysis rebuilt: table, 2 heatmaps, `recommendations.csv`.
- [x] INT8 verdict recorded: 13/19 green = 70.4% of listed params and 70.8% of kMAC/px; Red = H0, H3, D5 = 3.8% of params, 3.0% of kMAC/px.
- [x] QAT trigger evaluated and **not met**: no component >= 5% of MACs or params is Red at INT8, so QAT is skipped by rule with evidence.
- [x] Unit tests pass (4/4) via `python -m unittest discover -s quantization-pluse-mse/tests`.
- [ ] Whole-model multi-component config: **not implemented**. `run_sweep` (`scripts/pipeline.py:425`) builds exactly one config per component.
- [ ] Additivity: single-component penalties are **not** validated as summable, so no uniform-INT8 total may be quoted yet.
- [ ] BD-rate: complete 8-QP RD curves exist for 133 configs but BD-rate is not computed; `penalty_pct` is a slope proxy, not BD-rate.
- [ ] W8A8, calibration set, group/block scaling, clipping, group size, latency, energy: all pending.
- [ ] ILP: no solver code, no `coefficients.csv`, no OR-Tools installed.
- [ ] Report and slides: not started.

## Format coverage

| Format | Kodak | CLIC | Components | Quantization actually applied |
|---|---|---|---|---|
| INT8 | done | done | 19/19 | per-output-channel symmetric, `qmax=127` |
| INT4 | done | done | 19/19 | per-output-channel symmetric, `qmax=7` only |
| FP16 | done | done | 19/19 | native cast |
| BF16 | done | done | 19/19 | native cast |
| FP8-E4M3 | done | done | 19/19 | per-output-channel scaled cast to `float8_e4m3fn` |
| FP8-E5M2 | done | done | 19/19 | per-output-channel scaled cast to `float8_e5m2` |
| FP4-E2M1 | done | done | 19/19 | per-output-channel scaled grid cast, block 16 **not** implemented |

All main-sweep components are E0-E5, H0-H3, D0-D8 = **19**. Controls H4/H5/D9 are excluded by design.
INT4, FP8 and FP4 results are simulated (no native path on `sm_86`) and are quality-only.

## Progress at a glance

| WP | Work package | Status |
|---|---|---|
| WP0 | Environment and repo hygiene | Partial: `ortools` and `pytest` missing; `pandas`/`matplotlib` are not needed (analysis is stdlib + PIL) |
| WP1 | Phase 0: harness trust, noise floor, bitstream safety, time breakdown | Done except G0b (deferred) and the latency breakdown |
| WP2 | Component inventory from the live model | Done (Table 4 within 1% except hyper dec -1.36%) |
| WP3 | Fake-quant library | Done for all 7 formats; block/group rule not implemented; no activation quant |
| WP4 | Phase 1: weight-only screen | **Done**: 7 formats x 19 components x 2 datasets x 8 QPs, 0 failures |
| WP5 | Phase 2-3: follow-ups, whole-model, additivity | Not started (no W8A8, no whole-model, no additivity) |
| WP6 | Analysis, tables, figures | Partial: compact surfaces + analysis done; BD-rate, failure table, shortlists pending |
| WP7 | Conditional: QAT, ILP, deployment | QAT skipped by rule; ILP not started; no deployment build |
| WP8 | Report and presentation | Not started |
| WP9 | Decision visualizations (v3) | Planned |
| WP10 | QAT / greedy / ILP decision record (v3) | Planned |
| WP11 | Mixed-precision model + 3-dataset evaluation (v3) | Planned |

---

### WP0 - Environment and repo hygiene

**What to do**
- [ ] Install `ortools` when the ILP phase starts.
- [ ] Install `pytest` if CI-style runs are wanted; `unittest` works today.
- [ ] Freeze `requirements-quant.txt`.
- [ ] Add a manifest helper entry for git commit and dirty flag (currently only package and GPU facts are recorded).

**What we have done**
- [x] Python 3.12.6, PyTorch 2.6.0+cu124, CUDA 12.4, RTX 3050 Laptop (4 GB, `sm_86`) verified.
- [x] `datasets/kodak_thumb` (24), `datasets/kodak` (24 PNG), `datasets/clic_professional` (41) present.
- [x] `checkpoints/pulse-mse` has `config.json`, `model.pt`, `entropy_control_int.pt`.
- [x] `MLCodec_extensions_cpp` rebuilt and importing; one-image smoke test passed (`kodim01t.jpg`, QP 3, 0.6201 bpp, 30.556 dB).
- [x] `nvidia-ml-py` installed, so NVML energy sampling is available.
- [x] Repo made committable: large datasets, the raw ledger, and per-format sweep dumps are gitignored.

---

### WP1 - Phase 0: trust the harness

**What to do**
- [ ] **G0b (deferred until H1-H3 review is settled):** `scripts/test_bitstream_safety.py` (noise on `em_step1` weights; decode must still succeed).
- [ ] `scripts/bench_latency.py --format fp32`: per-stage time breakdown for one Kodak and one CLIC image.
- [ ] Copy the frozen baseline JSON into `results/baseline/` and record its SHA.

**What we have done**
- [x] `src/pulsequant/eval.py` reuses `codec/runtime.py` and the `evaluate.py` PSNR/bpp formula.
- [x] TF32/determinism flags pinned and recorded in the manifest.
- [x] `reference` run on `kodak_thumb`, `kodak`, and `clic_professional`: twice TF32 off + once default.
- [x] G0a passed; TF32-off noise floor is 0 dB / 0% / identical stream and recon hashes.
- [x] D1 INT8 controls passed; full D1 INT8 and E1 INT8 done on Kodak and CLIC.

---

### WP2 - Component inventory from the live model

**What to do**
- [ ] Generate `configs/components.yaml` from the inventory (no hand edits) and review once.

**What we have done**
- [x] `scripts/pipeline.py inventory` writes `artifacts/component_inventory.json` (+ `.md`).
- [x] Table 4 acceptance check passes within 1% except hyper dec at -1.36%.
- [x] H4/H5/D9 marked as controls; norms, biases and `position_bias` excluded.
- [x] 19 main-sweep components cover 13,671,984 of 14,179,432 neural parameters (96.4%).

---

### WP3 - Fake-quant library

**What to do**
- [ ] Block rule: block 16 along fan-in; per-output-channel fallback when `fan_in % 16 != 0`.
- [ ] `w8a8_dynamic` / `w8a8_static` via forward-pre-hooks; no 4-bit activations.
- [ ] Zero handling, clipping, scale search, idempotence tests.
- [ ] Optional formats (NF4, MXFP4/8, FP6, TF32 compute) only after the core passes.

**What we have done**
- [x] INT8, INT4, FP16, BF16, FP8-E4M3, FP8-E5M2, FP4-E2M1 fake quant in `src/pulsequant/core.py`.
- [x] Per-output-channel weight-only mode; biases, norms and QP gates untouched.
- [x] In-place overwrite from a cached FP32 clone, restore after, weight hashes.
- [x] Per-layer weight stats (SQNR, clip fraction, max abs error, effective bits).
- [x] Unit tests: identity, INT8 changes/restores, INT4 hurts more than INT8, output changes on a fixed input.
- [x] FP8 casts verified on `sm_86`.

---

### WP4 - Phase 1: weight-only screen

**What we have done** - the full screen is complete:

- [x] 7 formats x 19 components x 24 Kodak x 8 QPs and x 41 CLIC x 8 QPs.
- [x] 69,233 rows, 135 configs, 0 failures, `restored_ok` 135/135.
- [x] Controls present: `identity` (stream hashes match) and `int4` positive control.
- [x] Decoder-only components preserve `stream_hash` at 1.0, confirming the side invariant.
- [x] Every result records component, format, mode, granularity, `qmax`, weight hashes, size, model hash and config hash.

**What is still missing here**
- [ ] Whole-model run per format (all 19 at once): needs a multi-component config, 7 configs.
- [ ] Additivity: predicted sum of singles vs measured whole-model penalty.
- [ ] Group/block scaling so the 4-bit and FP8 columns stop being per-output-channel-lower-bound.

---

### WP5 - Phase 2-3: follow-ups, additivity

**What to do**
- [ ] 2a: W8A8 (INT8, FP8-E4M3) on the top-5 sensitive components + whole-model; dynamic first, then static with calibration shared across QPs vs per-QP.
- [ ] 2b: leaf split (pointwise / FFN / depthwise / CA) for the top-3 sensitive blocks.
- [ ] 2c: clipping comparison (min-max, percentile 99.9/99.99, MSE-optimal) on sensitive components.
- [ ] 2d: group size 16/32/64/128 for INT4/FP4 where it divides.
- [ ] 2e: CLIC confirmation of the shortlist (at most 10 configs, 41 images x 8 QPs).
- [ ] 3: "all green at INT8" and "all green + yellow"; predicted vs measured penalty.
- [ ] Calibration set: 64-128 random 512x512 crops, fixed seed, from DIV2K validation (801-900) or CLIC train; manifest with hashes; verify no overlap with Kodak/CLIC validation.

**What we have done**
- [x] Established that the repo has evaluation sets only; Kodak and CLIC validation must never be used for calibration.
- [ ] Calibration set not added; activation statistics not collected.

---

### WP6 - Analysis, tables, figures

**What to do**
- [ ] BD-rate from the existing 8-QP curves, per component per format. This is the standard codec metric and is computable now with no new runs.
- [ ] Bootstrap 95% CI, median, p90, worst image per component x format x QP.
- [ ] Failure table (decode failures, hash anomalies): currently zero failures, record that explicitly.
- [ ] Shortlists: safe (green) components; components needing higher precision.

**What we have done**
- [x] Three-table split (`sweep`, `configs`, `weight_stats`) plus `summary.csv`, `reference.json`, `manifests.json`, 6 figures per format.
- [x] Component x precision table, penalty and d_PSNR heatmaps, `recommendations.csv`.
- [x] `results/cross_precision/` summary for the completed formats.
- [x] Energy/memory feasibility note in `MISCELLANEOUS.md`.

---

### WP7 - Conditional: QAT, ILP, deployment

**What to do**
- [ ] ILP: coefficients table, CP-SAT model, greedy baseline, assignment validation. `x[c,p,q]` because INT8 sensitivity depends on QP in both directions.
- [ ] Deployment: FP16/BF16 PyTorch latency and energy first; TensorRT INT8 via ONNX; log fallbacks; never claim speed or energy for simulated formats.
- [ ] Energy protocol: loop at least 10 s, subtract idle, prefer the NVML total-energy counter, report J/image and scope (GPU only).

**What we have done**
- [x] QAT trigger evaluated: not met, so QAT is skipped by rule rather than by omission.
- [x] ILP formulation, variable set and constraints defined in `../ILP-mixed-precision/ilp.md`.
- [ ] No solver run and no TensorRT engine yet.

---

### WP8 - Report and presentation

**What to do**
- [ ] Baseline reproduction table plus time breakdown.
- [ ] Findings page: component ranking, QP dependence, FP8-vs-INT8 granularity result, 4-bit caveat.
- [ ] Whole-model INT8 results and RD curves.
- [ ] QAT/ILP decision with evidence, or why skipped.
- [ ] Explicit "simulated vs measured" section; note that thumbnails are smoke-only.
- [ ] Reproducibility appendix: commands, versions, hashes, noise floor.

**What we have done**
- [x] Report structure and figure list defined.

---

### WP9 - Decision visualizations (v3)

Done. `scripts/visualize_decision.py` writes six figures plus `decision_summary.csv` to
`results/decision/`, reading only the tracked per-precision surfaces.

- [x] `decision_component_ranking.png`: INT8 ranking, log scale, with parameter and MAC share.
- [x] `decision_precision_ladder.png`: the most aggressive precision still under 1% per component.
- [x] `decision_exposure.png`: parameter mass versus fragility, bubble = MAC share.
- [x] `decision_qp_sensitivity.png`: INT8 penalty per QP, shared log scale, split by family.
- [x] `decision_card.png`: level, penalty, driver QP, size, QAT trigger in one table.
- [x] `decision_budget_frontier.png`: greedy mixed versus uniform precision at equal cost.
- [x] Decision record in `results/decision/README.md`.

### WP10 - QAT / greedy / ILP decision (v3)

- [x] Decision recorded in `results/decision/README.md`: skip QAT by rule, keep greedy as the
  baseline, run CP-SAT ILP after additivity is validated.
- [ ] Implement the ILP once the additivity gate passes.

---

### WP11 - Mixed-precision model and evaluation (v3)

The multi-component capability was added as two additive scripts instead of a change to
`pipeline.py`, so the verified single-component path is untouched.

- [x] `src/pulsequant/core.py`: `apply_assignment` applies a per-component precision map in one
  pass, with `layer_fmt` carried on `QuantHandle` and used by `weight_stats`.
- [x] `scripts/eval_assignment.py`: whole-model measurement of uniform formats and mixed
  assignments against the FP32 reference.
- [x] **FP32 gate passed on Kodak**: the harness reproduces the reference stream and
  reconstruction hashes exactly at all 8 QPs.
- [x] Whole-model Kodak, all 8 QPs: FP32, FP16, BF16, INT8, safe, greedy.
- [ ] Whole-model CLIC, all 8 QPs: running.
- [x] `scripts/analyze_additivity.py`: additivity test. **Additivity holds**: measured over
  predicted penalty ratio is 0.96-1.05 for uniform INT8 across all 8 QPs, so the summed
  single-component coefficients are a valid predictor.
- [x] `scripts/solve_ilp.py`: MILP via HiGHS (`scipy.optimize.milp`, no new dependency).
  Variables `x[component, precision]` shared across QPs, per-QP quality constraints, size
  objective, greedy baseline for comparison.
- [ ] Evaluate the ILP assignment with the harness.
- [ ] Third dataset: not present in `datasets/`; the harness accepts any `datasets/<name>`.

**Measured on Kodak, all 8 QPs** (worst-QP mean penalty, analytic size):

| Config | Size MB | Saving | Worst-QP penalty % |
|---|---:|---:|---:|
| FP32 | 54.688 | 0.0 | 0.000 |
| FP16 | 27.416 | 49.87 | 0.038 |
| BF16 | 27.416 | 49.87 | 0.106 |
| INT8 | 13.904 | 74.57 | 26.184 |
| safe (per-component 1%) | 17.904 | 67.26 | 2.633 |
| greedy (1% total) | 24.565 | 55.08 | 0.386 |
| ILP (1% total, predicted) | 21.677 | 60.36 | 0.972 |

The ILP at the same 1% budget finds **21.677 MB where the earlier greedy found 24.565 MB**,
because the greedy compared here enforces the constraint per QP rather than on the sum of
per-component worst cases.

---

### Open questions

- [ ] Which third dataset to use for the held-out evaluation? Not present in `datasets/` yet.
- [ ] Is FP16/BF16 shown in the main heatmap or only as controls? (Recommended: shown, as the left columns.)
- [ ] Calibration source if W8A8 goes ahead: DIV2K validation or CLIC train?
- [ ] Is TensorRT on Windows acceptable, or switch to WSL2?
- [ ] Final deadline and the MVD cut-off.

---

### Immediate next actions

- [x] Sync the planning docs with the completed sweep.
- [x] Make the repo committable and commit it.
- [x] Build the WP9 decision visualizations.
- [x] Write the WP10 decision record.
- [ ] Implement the multi-component config and validate additivity (blocker for ILP and for the mixed model).
- [ ] Build the mixed-precision model from the assignment.
- [ ] Evaluate on Kodak, CLIC and a third dataset against the FP32 baseline.
- [ ] Compute BD-rate from the existing curves.
