# Milestones

Append-only proof of progress. Times are UTC.

## 2026-10-20T10:08Z - Inventory and Table 4 check

- `artifacts/component_inventory.json` and `.md` written.
- Table 4 kMAC/px check: analysis -0.03%, hyper enc +0.65%, hyper dec -1.36%,
  EM1 +0.09%, EM2 +0.07%, linear CDF 0.00%, synthesis -0.11%.
- All within 1% except hyper dec at -1.36%; record as a minor accounting difference.
- Kodak files are 150x150 JPEG thumbnails, not 768x512. Kodak is smoke only.

## 2026-10-20T10:09Z - FP32 reference and noise floor

- Kodak 24 x QP 0-7, three passes.
- TF32 off: max |d_PSNR| = 0.0 dB, max |d_bpp| = 0.0%, stream hashes identical.
- TF32 on: max |d_PSNR| = 0.0949 dB, max |d_bpp| = 1.04%, stream hashes differ.
- G0a vs frozen baseline: d_PSNR about 1e-7 dB, d_bpp 0.0% on all QPs. Pass.
- CLIC 41 x QP 0-7, three passes: TF32 off 0.0 dB / 0.0% / hashes identical;
  TF32 on 0.0189 dB / 0.25% / hashes differ. G0a pass.

## 2026-10-20T10:09Z - D1 INT8 smoke and controls

- D1 = `decoder.blocks.0`, `conv1`-`conv5` + `ca.1`, biases FP32.
- INT8 weight-only, per-output-channel symmetric, qmax=127.
- First 8 Kodak sorted, QP 0/3/7: 24/24 rows ok.
- Mean d_PSNR -0.00179 dB, worst -0.00873 dB, d_bpp 0.0%, mean penalty 0.032%.
- Identity control: 24/24 rows, d_PSNR 0.0, d_bpp 0.0, penalty 0.0, stream hashes match.
- INT4 control (simulated, qmax=7): 24/24 rows ok, mean d_PSNR -1.119 dB,
  worst -2.111 dB, mean penalty 20.26%. INT4 hurts more than INT8. Positive control passes.
- Weight hashes: original/quantized/restored recorded; quantized differs, restored equals original.
- Output changes on a fixed input; restore verified.
- No quantized checkpoint was written. The model is mutated in place and restored;
  the reproducible artifact is the base checkpoint + quant config + command + weight hashes.

## 2026-10-20T10:10Z - D1 INT8 full Kodak

- 24 images x QP 0-7, 192 rows, all ok.
- Mean d_PSNR -0.00182 dB, worst -0.01817 dB.
- d_bpp 0.0%, mean penalty 0.0299%, max 0.2775%.
- All weights restored; stream hashes match the FP32 reference (decoder-only).

## 2026-10-20T10:11Z - E1 INT8 full Kodak

- 24 images x QP 0-7, 192 rows, all ok.
- Mean d_PSNR -0.01977 dB, worst -0.18459 dB.
- d_bpp +0.0871%, mean penalty 0.4049%, max 2.5425%.
- Stream hashes differ from FP32, as expected for an encoder-side change.
- Worst-QP mean penalty is 0.89% (QP6), so E1 is green on Kodak by the 1% gate.
  Per-image tails reach 2.54% (kodim20 QP5/7); p90 is 1.32%.

## 2026-10-20T10:30Z - CLIC reference

- 41 images x QP 0-7, three passes.
- TF32 off: 0.0 dB, 0.0%, stream hashes identical.
- TF32 on: 0.0189 dB, 0.25%, hashes differ.
- G0a pass.

## 2026-10-20T10:30Z - CLIC D1 INT8

- 41 images x QP 0-7, 328 rows, all ok.
- Mean d_PSNR -0.00355 dB, worst -0.01348 dB.
- d_bpp 0.0%, mean penalty 0.0903%, max 0.3897%.
- Stream hashes match the FP32 reference.

## 2026-10-20T10:38Z - CLIC E1 INT8

- 41 images x QP 0-7, 328 rows, all ok.
- Mean d_PSNR -0.01940 dB, worst -0.08710 dB.
- d_bpp +0.0105%, mean penalty 0.5019%, max 3.6506%.
- Worst-QP mean penalty is 1.35% (QP7), so E1 is yellow on CLIC by the 1% gate.
- Worst E1 tails: Kodak `kodim20t` QP5/7 (-0.14/-0.18 dB, 2.54%);
  CLIC `casey-fyfe-999` QP6 (+0.049 dB but +4.86% bpp, 3.65%),
  `allef-vinicius-109434` QP7 (+0.023 dB but +3.59% bpp, 3.03%).
  The CLIC tails are rate increases, not PSNR drops.

## 2026-10-20T11:15Z - Per-precision review surfaces

- `pipeline.py export` written; source stays universal in `src/` and `scripts/`.
- `int8/results/` contains 1,064 rows, progress slice, 8 manifests, reference copies,
  summary CSVs, 27 PNG plots, and `component_vs_qp_*` tables for every metric.
- `int4/results/` contains the 24-row D1 positive control, reference copies, summary
  CSVs, 25 PNG plots, and `component_vs_qp_*` tables for every metric.
- Each precision README includes the FP32 baseline table (bpp, PSNR, encode/decode/total ms)
  and component x QP tables for d_PSNR, d_bpp, penalty, bpp, PSNR, encode, decode,
  total latency, and size reduction.
- Central raw ledger moved to `results/_ledger/`; review happens in the precision folders.
- Size reduction is analytic: D1 73.78%, E1 74.69% at INT8; D1 86.20% at the
  per-output-channel INT4 control.

## 2026-10-20T12:00Z - Compact per-precision analysis

- Deleted the previous large INT8/INT4 figure/table surface.
- Rebuilt a compact surface: 6 metric tables + 6 all-component QP line plots,
  plus one size table and one size bar plot.
- Metrics: d_PSNR, d_bpp, rate-equivalent penalty, encode latency change,
  decode latency change, total latency change.
- Baseline table included per dataset and QP.
- No baseline line in the plots; all components are colored lines across QP.
- Energy remains unmeasured and is marked N/A.

## 2026-10-20T19:00Z - Final output schema

- Switched to the three-table split:
  - `sweep_<fmt>.csv`: per-image/QP core metrics, `recon_hash`, `ref_recon_hash`, `status`, `error`
  - `configs_<fmt>.csv`: per-config assertions, weight hashes, output delta, analytic size
  - `weight_stats_<fmt>.csv`: per-layer SQNR, saturation fraction, max abs error, fan-in, effective bits
- Added `summary.csv` for derived mean/median/p90.
- Removed fake-quant latency, energy and memory from the sweep; those belong to the real-format benchmark.
- Removed per-image weight hashes and analytic size; now per config.
- Added `recon_hash` to the FP32 reference and quantized rows.
- Identity control now compares `recon_hash` to FP32; stream hash and `d_bpp_pct == 0` are the free integrity checks.
- INT4 control must differ in `recon_hash` and drop more PSNR than INT8.
- Old sample per-precision outputs deleted; full run will regenerate them.

## Current decision

- D1: green on Kodak and CLIC. Proceed.
- E1: green on Kodak by worst-QP mean (0.89%), yellow on CLIC (QP7 mean 1.35%).
  Per-image tails are 2.54% / 3.65%; p90 is 1.32% / 1.16%.
- Next: inspect E1 QP5-7 tails, then decide whether to keep E1 INT8 or test
  E1 FP16 / W8A8 before moving to E2.

## 2026-10-23 - Full format progress

- INT8, INT4, FP16, BF16, FP8-E4M3: complete on full-res Kodak + CLIC, all 19 components.
- FP8-E5M2: Kodak 12/19 components, CLIC pending.
- FP4-E2M1: pending.
- Ledger: 51,777 rows, 109 configs, 0 failures.
- Cross-precision analysis generated for completed formats in 
esults/cross_precision/.
- Main-sweep components: E0-E5, H0-H3, D0-D8 = 19.
- Controls H4/H5/D9 are excluded by design.
- Memory/energy deferred until after ILP/QAT.
