# Results workspace

Status: INT8, INT4, FP16, BF16, FP8-E4M3 and FP8-E5M2 complete on full-resolution Kodak + CLIC. FP4-E2M1 is running. Component x precision analysis is in `results/analysis/`. Thumbnails remain smoke/G0a only.

## Reference

| Dataset | Images | QPs | Rows | Noise floor |
|---|---:|---|---:|---|
| Kodak thumbnails | 24 | 0-7 | 576 | TF32 off: 0 dB, 0%, hashes identical |
| Kodak full-res | 24 | 0-7 | 576 | TF32 off: 0 dB, 0%, hashes identical |
| CLIC Professional | 41 | 0-7 | 984 | TF32 off: 0 dB, 0%, hashes identical |

G0a against the frozen baseline applies to `kodak_thumb` only. Full-resolution Kodak is the claim set. CLIC is the confirmation set.

## Folder layout

The central `_ledger/` is the append-only source of truth. The review surface is the
per-precision folder, e.g. `int8/results/` and `int4/results/`.

```text
results/
  _ledger/
    sweep.csv                  one row per (config, dataset, image, qp)
    progress.jsonl             append-only run/progress ledger
    manifests/{sweep_id}.json  config, hashes, git state, packages, flags
    reference/{dataset}/        pass1.csv, pass2.csv, tf32_on.csv, curve.json, g0a.json, noise_floor.json
    legacy_sample/              archived pre-schema sample run
  milestones.md                timestamped progress proof
  int8/results/                compact review surface
  int4/results/                compact review surface
  fp16/results/                populated when FP16 runs
  ...
```

Each `{precision}/results/` contains:

```text
README.md              baseline + component x QP tables + size + weight stats
sweep_<fmt>.csv        per-image/QP core metrics
configs_<fmt>.csv      per-config assertions and analytic size
weight_stats_<fmt>.csv  per-layer SQNR, clip fraction, max abs error
summary.csv            derived mean/median/p90 aggregates
reference.json         baseline, noise floor, G0a, slopes
manifests.json         relevant run manifests
figures/               5 metric line plots + 1 size bar
```

## Sweep schema

```text
sweep_id, config_id, dataset, image, h, w, qp,
bytes, pixels, bpp, mse, psnr, stream_hash, recon_hash,
ref_bpp, ref_psnr, ref_stream_hash, ref_recon_hash,
decode_ok, status, error
```

## Config schema

```text
config_id, sweep_id, component, format, mode, granularity, qmax, control,
orig_weight_sha256, quant_weight_sha256, restored_weight_sha256, restored_ok,
output_delta, fp32_bytes, quant_bytes, size_reduction_pct,
effective_bits_per_param, base_model_hash, quant_config_hash
```

## Weight stats schema

```text
config_id, layer, sqnr_db, clip_frac, max_abs_err, fan_in, effective_bits
```

## Progress schema

`_ledger/progress.jsonl` is append-only. Every row has `timestamp`, `sweep_id`, `run_id`,
`quant_config_hash`, component, format, control, dataset, image, qp, status, metrics,
weight hashes, git commit, model hash, manifest, and command.

## Previous sample (archived; will be replaced by the full run)

| Dataset | Component | Control | Rows | Mean d_PSNR | Worst d_PSNR | Mean penalty | Max penalty |
|---|---|---|---:|---:|---:|---:|---:|
| Kodak thumb | D1 | none | 192 | -0.00182 | -0.01817 | 0.0299% | 0.2775% |
| Kodak thumb | E1 | none | 192 | -0.01977 | -0.18459 | 0.4049% | 2.5425% |
| Kodak thumb | D1 | identity | 24 | 0.00000 | 0.00000 | 0.0000% | 0.0000% |
| Kodak thumb | D1 | int4 (simulated) | 24 | -1.11898 | -2.11059 | 20.2609% | 36.4326% |
| CLIC | D1 | none | 328 | -0.00355 | -0.01348 | 0.0903% | 0.3897% |
| CLIC | E1 | none | 328 | -0.01940 | -0.08710 | 0.5019% | 3.6506% |

D1 is green: reference passes, identity stream hashes match, INT4 hurts more than
INT8, weights change then restore, and output changes on a fixed input. E1 is green on
thumbnails by the worst-QP mean penalty (0.89%) and yellow on CLIC (QP7 mean 1.35%).
Per-image tails reach 2.54% on thumbnails and 3.65% on CLIC; p90 stays below 2%.
Per-precision results will be regenerated from the full run using the finalized
three-table surface: `sweep_<fmt>.csv`, `configs_<fmt>.csv`, `weight_stats_<fmt>.csv`,
`summary.csv`, `reference.json`, `manifests.json`, and 6 compact figures.

## Model lifecycle

No quantized checkpoint is saved. Weights are mutated in place and restored from a
cached FP32 clone. Reproducibility is proven by the base checkpoint hash, quant config
hash, weight hashes, command, and manifest. This avoids multiplying near-identical models.

## Reporting rules

1. Report per-image variance, not only the mean.
2. Mark simulated formats as simulated.
3. Report decode failures and hash anomalies.
4. State the model hash, git commit, manifest, and command.
5. Do not compare FP32 absolute numbers across datasets with different resolutions.
