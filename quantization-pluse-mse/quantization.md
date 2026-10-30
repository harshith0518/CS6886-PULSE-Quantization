# PULSE-MSE Quantization Plan (v2.1, simplified)

Status: active. Supersedes v1. Full-resolution Kodak is now present (24 PNGs). FP32 reference and D1 controls are run on full-res Kodak; INT8/INT4 all-component loop is running on Kodak + CLIC.

## 0.1 v2.1 amendments (controls, reference order, lifecycle)

These amendments override any conflicting v2 text below.

1. **FP32 reference comes first, all 8 QPs.** Run Kodak 24 x QP 0-7 twice with TF32 off and once with default flags. This produces the reference, the noise floor, and G0a. Do not run INT8 before this exists.
2. **D1 cannot be declared green without controls.** A 143k-parameter decoder block should cost about 0%. If INT8 also costs about 0%, the harness may be quantizing nothing.
3. **Positive control:** rerun D1 with `qmax=7` (INT4) on the same function. It must show a clearly larger penalty than INT8. If not, the quantized weights are not reaching the codec.
4. **Negative control:** apply and restore with no change. The stream hashes must be identical to the FP32 reference.
5. **Weight hashes, not `model_deleted`:** log SHA256 of D1's weights for the FP32 original, the quantized version (must differ), and after restore (must equal the original). Also assert that D1's output changes on a fixed input.
6. **Mutate in place and restore from a cached clone.** Do not build a second quantized model. No per-run `empty_cache()`.
7. **Resume key:** `(config_id, dataset, image, qp)`. Do not include `run_id`; it changes every invocation.
8. **Smoke images:** first 8 Kodak files in sorted order, identical for reference and quantized runs.
9. **D1 scope:** `conv1`-`conv5` plus `ca.1`, biases FP32. State this in `configs/components.yaml`.
10. **Kodak size check:** `datasets/kodak_thumb` holds the 24 **150x150 JPEG thumbnails**; use it for smoke and G0a only. `datasets/kodak` must hold the 24 full-resolution PNGs (768x512 or 512x768) before final claims. If missing, stop.
11. **G0b timing:** the bitstream-safety test is not needed for D1 (decoder-only changes cannot affect parsing). Run it before H1-H3.
12. **Time breakdown:** defer until just before the full sweep.
13. **D1 green rule:** D1 is green only if the reference reproduces the baseline, the identity run gives identical stream hashes, INT4 hurts more than INT8, and the weights restore. Otherwise "green" means nothing.
14. **Amended order:** inventory + Table 4 check + Kodak size check -> FP32 reference (3 runs) -> G0a + noise floor -> D1 INT8 smoke + identity + INT4 controls -> show data -> full D1 -> E1 -> CLIC across all QPs for the real loop.

---

## 0. Goal and minimum viable deliverable

Question: which parts of PULSE-MSE tolerate FP16 / BF16 / FP8 / INT8 / FP4 / INT4, at which QP,
and where does rate or quality actually break?

**Minimum viable deliverable (MVD)** = Phase 0 + Phase 1 (at least the FP16 and INT8 columns)
+ whole-model INT8 + the time breakdown. That already answers the proposal's headline question:
does one INT8 setup work across all 8 QPs, and how does sensitivity depend on QP and component.
Everything else is added value; do not let it block the MVD.

Rules (unchanged from v1):

1. One component at a time; every other component stays FP32.
2. Same images, padding, QP, seed, evaluation order as the baseline.
3. Quality (simulated formats) and deployment (real hardware formats) are separate result sets.
4. No speed, memory or energy claim for a simulated format. Ever.
5. No evaluation image (Kodak, CLIC validation) in calibration or QAT data.
6. Every run logs config, model hash, git commit (+dirty flag), package versions, command.
7. A failure is a result. Record it; never retry silently.

---

## 1. Facts that shape the design

### 1.1 Only one path must be bit-exact (paper Sec. 2-3, Table 6)

The CDF used to decode `y_q` is chosen by the integer path only:

```text
decoded z_hat -> INT8 linear projection (H4) -> CDF index   (+ Meta Prior / BitEstimator tables, H5)
```

The symbols `y_q` are coded with a zero-mean Gaussian. The float networks (hyper decoder, EM step 1,
EM step 2) only produce `mu` and `q`, which enter the reconstruction `y_hat = q * (y_q + mu)` and the
encoder's choice `y_q = round(y/q - mu)`. They never select a CDF. Table 6 agrees: float networks ran
on NPU / CPU / GPU with 0 CDF-index and 0 symbol mismatches.

Consequences:

- Quantizing any convolution component changes **rate and quality**. It cannot desynchronize the stream
  while encoder and decoder use the same quantized model.
- No "CDF recalibration" is needed for **decodability** unless H4 itself is changed (it is not, in the main sweep).
  Rate-optimality is separate: if H1-H3 change the y distribution, the fixed CDFs still decode but bpp may be suboptimal.
  Recalibration is optional for a rate-fair bpp comparison, not for stream validity.
- **One evaluation mode is enough: the real bitstream path** (`compress_image` + `decompress_image`).
  Drop v1's forward / encoder_only / decoder_only / entropy_only modes.

**Gate G0b (verify before trusting this):** add small Gaussian noise to the weights of `em_step1`,
run compress + decompress on 3 images. Expected: decode succeeds, PSNR drops mildly, no garbage output.
If decode breaks, read `codec/models/entropy/` to find what else feeds the CDF index and update this section.

### 1.2 Where each component runs

| Side | Components | Why it matters |
|---|---|---|
| Encode only | E0-E5, H0 | No sync constraint with the decoder. 43.7 of 45.35 kMAC/px of the sender. Safest place to be aggressive. Matches the "batch-encode photos on a PC" use case. |
| Encode and decode | H1, H2, H3 (and H4) | Must match on both sides, but a mismatch only costs quality, not the stream. |
| Decode only | D0-D8 | The latency-critical receiver (5.2 kMAC/px). |

### 1.3 Compute is not where the parameters are

Recomputed from the model report. These match paper Table 4 (analysis 43.21 vs 43.22, synthesis 3.53 vs 3.53,
hyper enc 0.45, hyper dec 0.24, EM1 0.58, EM2 0.74, linear CDF 0.12 kMAC/px). Use Table 4 as the
acceptance test for the inventory script.

- Encoder = 12.1M params (86%) and 43.2 kMAC/px (95% of sender compute).
- The channel-attention conv inside every DCB (9% of that DCB's params) runs on a pooled 1x1 vector: about 0 MACs/px.
- 68% of decoder MACs (2.40 of 3.53 kMAC/px) are in the renderer (D3-D8), which holds 53% of decoder params.
  D5-D8 alone are 2.4% of decoder params but 29% of decoder MACs (they run on the 1/4-resolution grid).
- So rank sensitivity against both parameters (memory) and kMAC/px (latency). They tell different stories.

### 1.4 The FP32 baseline is already mixed

```text
FP32 encoder + FP32 hyper/EM networks + INT8 integer scale decoder (H4) + integer CDF/rANS + FP32 decoder
```

H4 and H5 are controls, not PTQ targets. The paper reports about 0.04% BD-rate cost for the INT8 linear
CDF path (Table 1), which is a useful "INT8 can be nearly free here" anchor.
The entropy coder (rANS, Meta Prior search) runs on the CPU, so GPU quantization can only speed up the
neural-network share of the pipeline. Measure that share first (Phase 0).

### 1.5 Hardware reality (RTX 3050 Laptop, sm_86, 4 GB, CUDA 12.4, PyTorch 2.6.0+cu124)

- Real: FP32, FP16, BF16, INT8 (via TensorRT).
- Simulated only: FP8 (needs sm_89+), FP4, INT4 (no practical convolution path in PyTorch/TensorRT).
- **TF32:** PyTorch enables TF32 for cuDNN convolutions by default on Ampere. TF32 has a 10-bit mantissa, the
  same as FP16, so leaving it on would blur FP32-vs-FP16 differences. The reference and all quantized runs
  must pin `torch.backends.cudnn.allow_tf32 = False` and `torch.backends.cuda.matmul.allow_tf32 = False`
  (also `cudnn.deterministic = True`, `cudnn.benchmark = False`), and record the flags.

---

## 2. Formats

| Format | Bits | Simulation method | Effective bits/weight (incl. scales) | Real on this laptop? |
|---|---:|---|---|---|
| FP32 | 32 | reference | 32 | yes (TF32 off) |
| FP16 | 16 | `.half()` round trip | 16 | yes |
| BF16 | 16 | `.bfloat16()` round trip | 16 | yes |
| INT8 | 8 | symmetric, per-output-channel, range -127..127 | 8 + 32/fan_in (about 8.06 at fan_in 512) | yes (TensorRT) |
| FP8 E4M3 | 8 | `torch.float8_e4m3fn` round trip, scale so max maps to 448 | 8 | no (simulated) |
| FP8 E5M2 | 8 | `torch.float8_e5m2` round trip, scale so max maps to 57344 | 8 | no (simulated) |
| FP4 E2M1 | 4 | nearest of {0, 0.5, 1, 1.5, 2, 3, 4, 6} (+sign), block 16, FP16 block scale | 5.0 (4.5 with an FP8 scale) | no (simulated) |
| INT4 | 4 | symmetric -7..7, group 16, FP16 scale | 5.0 | no (simulated) |

Notes:

- FP8 casts through the PyTorch dtypes should work on sm_86 as elementwise conversions (check with a one-line test).
  FP4 E2M1 has no dtype in torch 2.6, so use a nearest-value lookup.
- E5M2 weights are expected to be worse than E4M3 (2 mantissa bits). Kept for completeness; E5M2 is mainly an activation format.
- Optional later (do not block): NF4, MXFP4/MXFP8 (block 32), FP6, TF32 as a compute mode.

---

## 3. How tensors are quantized

- **What is quantized:** Conv2d weights of the selected component. Biases, RMSNorm, GELU/HardGELU/SiLU,
  the CA sigmoid, PixelShuffle and QP gates stay FP32 (FP16 under TensorRT). "INT8" in this study means
  conv weights, and optionally conv input activations.
- **Modes:** `w_only` (default, Phase 1), `w8a8_dynamic`, `w8a8_static` (INT8 and FP8 only).
  **No 4-bit activations**: A4 post-training quantization on a conv codec would be uninformative.
- **Implementation:** `w_only` needs no model surgery. Cache the FP32 `state_dict`, overwrite the selected
  conv weights with the fake-quantized values, run, then restore. Activation quantization is a forward-pre-hook on the conv input.
- **Granularity defaults:**
  - INT8 weights: per-output-channel symmetric. INT8 activations: per-tensor **symmetric** (TensorRT has no zero point, so v1's "asymmetric" would not match deployment).
  - FP8 weights: per-tensor scale (per-channel as a follow-up). FP8 activations: per-tensor dynamic.
  - FP4 / INT4: blocks along the fan-in, **block size 16**.
- **Block-size rule (important):** `fan_in = (Cin / groups) * kH * kW`. Block 32/64/128 does not divide the fan-ins
  144 (decoder), 272 (decoder conv5), 48 (renderer, hyper conv5), 24 (hyper_dec grouped), or 9 (all depthwise 3x3).
  Block 16 divides everything except the depthwise convs (9) and hyper_dec grouped conv1/3/4 (24);
  for those two cases fall back to per-output-channel scales. Larger blocks are a follow-up only on layers where they divide.
- **Clipping:** min-max by default. Percentile (99.9 / 99.99) and MSE-optimal only on components the screen flags.
- **Rounding:** round-to-nearest-even in PTQ. Stochastic rounding only inside QAT.
- Run the same fake-quant core for every format so the formats are comparable.

Formulas (unchanged): symmetric `s = max|x| / q_max; q = clamp(round(x/s)); x_hat = q*s`.
FP8/FP4: scale so `max|x|` maps to the format's max finite value, round to the format grid, rescale.

---

## 4. Components

Quantize each row as a unit with all its conv descendants. Rows run in the order enc -> entropy -> dec.

| ID | Module prefix | Params | kMAC/px | Runs in | Main sweep |
|---|---|---:|---:|---|---|
| E0 | `encoder.patch_embed` | 393,728 | 1.54 | encode | yes |
| E1-E4 | `encoder.blocks.N` (each) | 2,893,824 | 10.26 each | encode | yes (4 rows) |
| E5 | `encoder.proj_out` (+`norm_out`) | 164,672 | 0.64 | encode | yes |
| H0 | `entropy_model.hyper_enc` | 303,456 | 0.45 | encode | yes |
| H1 | `entropy_model.hyper_dec` | 154,688 | 0.24 | both | yes |
| H2 | `entropy_model.em_step1` | 166,656 | 0.58 | both | yes |
| H3 | `entropy_model.em_step2` | 207,296 | 0.74 | both | yes |
| H4 | `entropy_model.scale_dec` | 496,648 | 0.12 | both | control (already INT8) |
| H5 | `entropy_bottleneck`, `meta_prior` | 549,120 | 0 | both | control (tables) |
| D0 | `decoder.up0` | 46,224 | 0.18 | decode | yes |
| D1-D2 | `decoder.blocks.N` (each) | 142,832 | 0.47 each | decode | yes (2 rows) |
| D3 | `decoder.renderer.latent_upsample` | 246,528 | 0.96 | decode | yes |
| D4 | `decoder.renderer.condition_proj` | 111,360 | 0.43 | decode | yes |
| D5 | `decoder.renderer.mlp` | 4,704 | 0.29 | decode | yes |
| D6 | `decoder.renderer.adaln` | 7,056 | 0.43 | decode | yes |
| D7 | `decoder.renderer.output` | 2,352 | 0.14 | decode | yes |
| D8 | `decoder.renderer.input_proj` (+`position_bias`) | 3,072 | 0.14 | decode | yes |
| D9 | QP gates (`q_scale_*`, `q_basic`, `qp_bias`) | 1,584 | 0 | both | keep FP32 |

Main sweep = 19 components (E0, E1-E4, E5, H0-H3, D0, D1-D2, D3-D8).
Inside a DCB the convs are conv1/conv3 (1x1), conv2 (depthwise 3x3), conv4/conv5 (FFN), and CA. If a DCB comes out sensitive,
split it by role (Phase 2b) rather than guessing. Depthwise convs (9 weights per channel) are the usual suspects.

---

## 5. Experiment phases

### Phase 0 - Trust the harness (do first, about 1 day)

**Amended execution order (v2.1):** inventory + Table 4 check + Kodak size check -> FP32 reference
(Kodak 24 x QP 0-7, twice TF32 off + once default flags) -> G0a + noise floor -> D1 INT8 smoke
plus identity and INT4 controls -> show data -> full D1 -> E1 -> CLIC across all QPs.

1. **eval.py:** reuse the compress / decompress code from `codec/runtime.py` and the exact PSNR / bpp
   calculation from `evaluate.py:78-89`. Do not reimplement it. `pulse_mse_baseline_benchmark.py` is only a
   wrapper around `evaluate.py`; it does not contain the codec or metric code. Add a single hook,
   `apply_quant(model, config)`.
2. **FP32 reference through the harness** (Kodak, 24 x 8 QPs): run twice with TF32 off, once with default flags.
   Compare stream hashes, PSNR, bpp. This gives the **noise floor** and the check against the frozen baseline JSON (**G0a**).
3. **G0b** bitstream-safety test (Sec. 1.1).
4. **Time breakdown** of the FP32 pipeline for one Kodak and one CLIC image: encode NN, Meta-Prior/rate search, rANS encode,
   rANS decode, decode NN, I/O. This bounds the best-case INT8 speedup and decides how much effort the latency angle deserves.
5. **Inventory from the live model:** params, shapes, fan-in, groups, kMAC/px per component, with the Table 4 acceptance test.
6. **Dataset sanity:** `datasets/kodak_thumb` contains the 24 `kodimNNt.jpg` **150x150 JPEG thumbnails**
   used by the frozen baseline JSON (`pixels=22,500`). `datasets/kodak` must contain the 24 full-resolution
   lossless PNGs. If the full-resolution set is missing, stop before running the real loop.

Exit: G0a and G0b pass; noise floor recorded; time breakdown in hand.

### Phase 1 - Weight-only screen (the main heatmap)

- Smoke test first: D1 x INT8 weight-only x QP {0, 3, 7} x first 8 Kodak images (sorted).
  Run the FP32 reference, identity control, and INT4 (`qmax=7`) control on the same 8 images.
  D1 is green only if: reference matches baseline, identity stream hashes match, INT4 penalty > INT8 penalty,
  weight hashes change then restore, and D1's output changes on a fixed input. Then repeat for E1.
- Time one full config (24 images x 8 QPs). Expect minutes, not hours (check the wall-clock in your baseline JSON); if it exceeds about 3 minutes, screen QPs {0, 3, 7}
  for everything and fill the other QPs only for yellow/red cells.
- Full screen: 19 components x 7 formats {FP16, BF16, INT8, FP8-E4M3, FP8-E5M2, FP4-E2M1, INT4}
  plus 7 whole-model configs (all 19 at once, per format) = 140 configs x 192 evaluations = 26,880 image evaluations, real bitstream mode.
- **Real quantization loop:** after the Kodak smoke and D1/E1 checks, run CLIC Professional across all 8 QPs
  for every component/format that survives the screen. Kodak is smoke only because the current files are 150x150 thumbnails.

### Phase 2 - Targeted follow-ups (only where Phase 1 points)

| Step | What | Scope |
|---|---|---|
| 2a | W8A8 (INT8; FP8-E4M3) dynamic, then static. Static calibration compared as **shared across QPs** vs **per-QP**. | top 5 sensitive + whole-model. Answers "does one INT8 config work for all 8 QPs?" |
| 2b | Leaf split by role: pointwise / FFN / depthwise / CA | top 3 sensitive blocks |
| 2c | Clipping: min-max vs percentile 99.9/99.99 vs MSE-optimal | INT8/INT4 on sensitive components |
| 2d | Group size 16/32/64/128 | INT4/FP4 on layers where it divides |
| 2e | CLIC confirmation (41 images x 8 QPs) | shortlist of at most 10 configs |
| 2f | Optional: encode FP32, decode quantized (cross-device tolerance) | H1-H3 only |

### Phase 3 - Whole-model and additivity (the part v1 was missing)

- Uniform whole-model per format (already in Phase 1), then "all green components at INT8" and "all green + yellow", W-only and W8A8.
- Compare **predicted** (sum of single-component penalties) with **measured** whole-model penalty. If damage is not additive,
  the ILP in Phase 4 needs interaction terms or must be validated more carefully.

### Phase 4 - Conditional: QAT and ILP (Sec. 9)

### Phase 5 - Conditional: real-format deployment (Sec. 9)

---

## 6. Metrics and how to collect them

### 6.1 What is measured, and how

| Metric | How | Notes |
|---|---|---|
| bpp | `8 * len(stream) / (H * W)` from real `compress_image`, including header/envelope | same function as the baseline |
| PSNR | decoded image vs original, using the baseline's exact function | same clamp/round as the baseline |
| Paired deltas | `d_psnr = psnr_q - psnr_ref`, `d_bpp_pct`, **per image and QP, same image against FP32** | cancels everything except the quantization effect |
| Rate-equivalent penalty (primary) | Sec. 6.3 | one number per cell: heatmaps, gates, ILP |
| BD-rate | Bjoentegaard over the 8 QP points, dataset-mean bpp and PSNR per QP, vs FP32 | aggregate per config; needs 8 valid points |
| Failures | `decode_ok`, exception text, stream hash | counted, never dropped |
| Weight error | SQNR (dB), clip fraction, max abs error per layer, computed once per config | static, not per image |
| Size | analytic: effective bits/weight x params | memory for simulated formats is analytic only |
| MACs | analytic from layer shapes, 1088x1920 equivalent | validate against paper Table 4 |

### 6.2 Reference and noise floor

- **Reference = FP32 re-run through this harness** (the entropy extension was rebuilt after the 2026-10-11 baseline).
  Paired deltas are taken against these rows.
- The frozen `benchmark_results/*.json` is a **sanity check (G0a)**, not the reference. If baseline numbers differ, the cause
  is most likely TF32 (default flags) or the extension rebuild; the TF32-on run in Phase 0 identifies which.
- **Noise floor** = largest per-image |d_psnr| and |d_bpp| between the repeated FP32 runs and the TF32 on/off runs.
  Gate thresholds must be at least 3x the noise floor, otherwise the cell is "inconclusive".

### 6.3 Primary metric: rate-equivalent penalty

ΔPSNR and Δbpp trade off, so two thresholds are ambiguous. Convert to one number using the local slope of the
FP32 curve at that QP:

```text
s_qp        = dB per bpp-doubling of the dataset-mean FP32 curve at that QP (central difference; one-sided at QP 0 and 7)
penalty_pct = 100 * [ (1 + d_bpp_pct/100) * 2^(-d_psnr / s_qp) - 1 ]       # positive = worse
```

Slopes from the existing baseline: Kodak about 4.1 dB per doubling (0.10 dB is roughly 1.7% rate), CLIC about 2.7 dB per doubling
(0.10 dB is roughly 2.6% rate). A fixed dB threshold therefore means different things on the two datasets;
the penalty does not. BD-rate is the rigorous aggregate; the penalty is its per-QP version.

### 6.4 Statistics

- Report per cell: mean (with 95% bootstrap CI across images), median, p90, worst image (name it), and failure count.
- 24 images is small. Differences below the CI are not findings.

### 6.5 Timing, energy, memory (real formats only: FP32, FP16, BF16, TensorRT INT8)

Fake-quant runs add quantize/dequantize ops and still store FP32 weights, so **sweep rows record quality only**.
Timing/energy live in a separate script and file.

- **Latency:** one fixed Kodak and one fixed CLIC image; 5 warm-ups, then at least 30 timed repeats; report median and IQR.
  `torch.cuda.synchronize()` around every timed region. Time each stage separately (encode NN, rate search, rANS encode,
  rANS decode, decode NN). Plugged in, high-performance power mode, other apps closed; log GPU clocks and temperature
  before and after (thermal throttling is the biggest noise source on a laptop).
- **Energy:** the baseline's 200 ms `nvidia-smi` sampling cannot resolve a roughly 100 ms operation. Instead loop the same operation
  for at least 10 s and divide by the repeat count. Prefer NVML `nvmlDeviceGetTotalEnergyConsumption` (before/after, no sampling);
  if unsupported on this laptop GPU, sample power at 20 Hz or faster. Measure idle power for 10 s first and subtract it. Report J per
  image and J per megapixel. State the scope: GPU board only; the CPU entropy coder is not included.
- **Memory:** `torch.cuda.reset_peak_memory_stats()` then `max_memory_allocated()` and `max_memory_reserved()` per run (process level).
  GPU-wide `nvidia-smi` numbers (the baseline's 383 MiB / 1,554 MiB) include the CUDA context and other apps, so label them as such.

---

## 7. Output files

One append-only ledger instead of thousands of JSON files. Resumable: skip any row whose key
`(config_id, dataset, image, qp)` already exists.

```text
results/_ledger/sweep.csv                  per-image/QP core metrics
results/_ledger/configs.csv                per-config assertions and analytic size
results/_ledger/weight_stats.csv           per-layer SQNR, clip fraction, max abs error
results/_ledger/manifests/{sweep_id}.json  git commit + dirty flag, package versions, GPU/driver, TF32/determinism flags,
                                           seed, command, model hash, dataset hashes, full config
{precision}/results/sweep_<fmt>.csv         precision slice of the core metrics
{precision}/results/configs_<fmt>.csv       precision slice of the config assertions
{precision}/results/weight_stats_<fmt>.csv  precision slice of the per-layer stats
{precision}/results/summary.csv             derived mean/median/p90 aggregates
{precision}/results/README.md               baseline + component x QP tables + size + weight stats
{precision}/results/reference.json          baseline, noise floor, G0a, slopes
{precision}/results/manifests.json          relevant manifests
{precision}/results/figures/                5 metric line plots + 1 size bar
```

`sweep.csv` columns:

```text
sweep_id, config_id, dataset, image, h, w, qp,
bytes, pixels, bpp, mse, psnr, stream_hash, recon_hash,
ref_bpp, ref_psnr, ref_stream_hash, ref_recon_hash,
decode_ok, status, error
```

`configs.csv` columns:

```text
config_id, sweep_id, component, format, mode, granularity, qmax, control,
orig_weight_sha256, quant_weight_sha256, restored_weight_sha256, restored_ok,
output_delta, fp32_bytes, quant_bytes, size_reduction_pct,
effective_bits_per_param, base_model_hash, quant_config_hash
```

`weight_stats.csv` columns:

```text
config_id, layer, sqnr_db, clip_frac, max_abs_err, fan_in, effective_bits
```

FP32 reference rows use `component = FP32`.

Code layout: one universal library and one entry point per task. The per-precision folders
(`fp16/`, `bf16/`, `fp8_e4m3/`, `fp8_e5m2/`, `int8/`, `fp4_e2m1/`, `int4/`, `fp32/`)
hold that format's README and its `results/` review surface. They contain no separate code;
`pipeline.py export` writes the slice, metrics, compact tables, plots, reference summary, and manifests.

```text
src/pulsequant/   core.py  eval.py  plots.py
scripts/          pipeline.py inventory | reference | sweep | export
```

---

## 8. Decision gates

Thresholds are starting points. Set them relative to the measured noise floor.

| Gate | Condition | Action |
|---|---|---|
| G0a | FP32 harness matches the frozen baseline within the noise floor | proceed |
| G0b | Bitstream-safety test passes (Sec. 1.1) | proceed with the single bitstream mode |
| Green | worst-QP penalty <= 1% (about 0.06 dB Kodak, 0.04 dB CLIC) | safe at this format |
| Yellow | 1% to 3% | test W8A8, clipping, group size |
| Red | > 3%, or any decode failure | keep FP16/FP32, or QAT if the component matters |
| Low-bit interesting | within 0.5 percentage points of the same component at INT8 | candidate for ILP |
| "Matters" (QAT trigger) | component holds at least 5% of sender or receiver MACs, or at least 5% of params | defined here because v1 left it open |
| QAT justified | a component that matters is Red at INT8 and a 500-step trial recovers at least half the penalty | run full QAT |
| ILP justified | at least 2 components have different safe precisions | run ILP |

Classify on the **worst QP** (that is the "one setup for all 8 QPs" question) and also report the mean over QPs.
Flag a cell if its p90 image penalty exceeds twice the threshold.

---

## 9. Conditional work

### QAT (Phase 4)

- Loss: `R_y + R_z + lambda_qp * D_MSE`, the same family as training. Sample a QP at random each step (lambda range 24-672).
- Fake quant with straight-through estimator; learn weight scales/clipping; freeze everything outside the target component.
- 512x512 crops, batch 1-2, AMP. PULSE works at 1/16 resolution, so activations at 512 crops are tens of MB; the 4 GB card is
  not the bottleneck (optimizer state for 14M params is on the order of 100-200 MB, less if only one component trains).
- 500-1000 steps, LR 1e-5 to 1e-4, cosine; validate every 250 steps on a held-out set that is neither Kodak nor CLIC validation.
- Evaluate with the identical PTQ protocol and compare PTQ vs QAT vs FP32.

### ILP (Phase 4)

- Variables: `x[c, p]` in {0, 1}, one precision per component, **one assignment shared by all QPs** (the model is "one model, eight QPs").
  Solve a second, per-QP variant only to show whether per-QP assignment buys anything.
- Constraint: `sum_c sum_p x[c,p] * penalty[c,p,q] <= epsilon` for every QP q (additivity assumption; Phase 3 tests it), and
  `sum_p x[c,p] = 1`.
- Two objectives, two solves:
  1. **Size** (effective bits x params): works for all formats because size is analytic. Label the result "size-optimal, quality simulated".
  2. **Latency/energy:** only for real formats (FP32/FP16/BF16/INT8) using measured per-component timings.
  If a coefficient is missing, exclude that format. Never invent one.
- Solver: `scipy.optimize.milp` (HiGHS) or OR-Tools CP-SAT. Gurobi is unnecessary at about 19 x 8 variables.
- Validate the chosen assignment with the full protocol against uniform FP32, FP16 and INT8.

### Deployment (Phase 5)

- FP16/BF16: PyTorch autocast, measured with the Sec. 6.5 protocol. This is easy and should come first.
- TensorRT INT8: export via the existing ONNX path, build FP16 and INT8 engines, record every layer that falls back, keep the entropy path integer and CPU-side.
  TensorRT does run on Windows but the install is fiddly; budget time, and keep WSL2 as the fallback. Do not let this block the quality study.
- Do not compare simulated formats on speed, memory (measured) or energy.

---

## 10. Figures (all generated by `make_figures.py` from the CSVs)

| # | Figure | Message |
|---|---|---|
| F1 | Heatmap: component x format, cell = worst-QP penalty (green/yellow/red, value annotated) | which parts tolerate which precision |
| F2 | Heatmap: component x QP for INT8 (and the worst low-bit format) | does QP change sensitivity |
| F3 | RD curves (PSNR vs bpp, 8 QPs): FP32 vs uniform FP16 vs uniform INT8 vs best mixed, BD-rate in the legend | does it still look like PULSE |
| F4 | Pareto: model size (MB) vs BD-rate, marker shape = real vs simulated | the sweet spot |
| F5 | Sensitivity vs params and vs kMAC/px (log-log), colour = encode / both / decode | why: compute and params disagree |
| F6 | Per-image penalty distributions for the 4 worst components, worst image named | tail risk hidden by means |
| F7 | Predicted (sum of singles) vs measured whole-model penalty | additivity |
| F8 | Stacked latency per stage and J/image, real formats only | where the time goes |

Style rules: same colour per format in every figure; every figure states dataset, image count and QPs; simulated formats hatched;
no figure without a CSV behind it.

Suggested 4-slide story for the professor: (1) baseline reproduced plus time breakdown, (2) F1 and F2,
(3) F3 and F4 for whole-model INT8, (4) what QAT/ILP added, or why they were not needed.

---

## 11. Data

- **Evaluation:** Kodak (24, currently 150x150 thumbnails) is smoke only. CLIC Professional validation (41)
  is the real quantization loop, across all 8 QPs. These are enough for evaluation.
- **Calibration** (needed only for static activation scales, Phase 2a; Phase 1 does not need it): 64-128 random 512x512 crops
  with a fixed seed. Source: DIV2K validation images 801-900 (disjoint from the paper's DIV2K training split and from our evaluation sets),
  or CLIC training images. A manifest with image hashes, crop coordinates and preprocessing.
- **QAT** (only if triggered): DIV2K train plus CLIC train (585 images) is plenty for 500-2000 steps.
  Flickr2K is not needed.
- Proposal wording to fix: "add extra testable image sets to bias the model toward the test image sets" reads as training on
  test images. Say "extra images from the same distribution; the evaluation images are never used".

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| TF32 silently on | pin and record flags (Sec. 1.5); measure TF32 on/off in Phase 0 |
| Harness protocol drift | reuse the baseline code, paired per-image deltas, G0a |
| 24 images is small | bootstrap CIs; confirm on CLIC before claiming |
| Kodak files are JPEG | note in report; relative deltas unaffected |
| Fake-quant block size does not divide fan-in | block-16 rule with per-channel fallback (Sec. 3) |
| Depthwise convs dominate damage | split by role in Phase 2b |
| TensorRT on Windows | quality study does not depend on it; WSL2 fallback |
| Thermal throttling | timing protocol (Sec. 6.5); report clocks and temperature |
| Entropy path assumption wrong | G0b before the sweep |
| Mean hides failures | p90, worst image, failure counts |

## 13. Out of scope

Full PULSE retraining, 2:4 sparsity, perceptual and PULSE-S/L variants, cross-device bitstream portability,
real FP4/INT4 kernels, production TensorRT before the PTQ map exists, 4-bit activation quantization.

---

## 14. Changes from v1 (for the record)

- One evaluation mode (real bitstream) instead of five; no CDF recalibration step (Sec. 1.1).
- FP32 re-run through the harness is the reference; frozen JSON is a sanity check (Sec. 6.2).
- One primary metric (rate-equivalent penalty) instead of two thresholds (Sec. 6.3); thresholds anchored to the noise floor.
- Added whole-model and additivity runs (Phase 3) and a definition of "component that matters".
- Block size 16 default with a divisibility rule; INT8 activations symmetric to match TensorRT.
- No timing/energy/memory columns on simulated runs; protocols fixed (Sec. 6.5).
- Single ledger CSV plus manifests instead of one JSON per run; one codebase instead of per-precision scripts.
- 4-bit activations dropped; ILP uses one assignment shared across QPs; stages A-H collapsed into Phases 0-5.

---

## 15. Next step: prompt for the implementer (Phase 0 complete)

```text
Read quantization.md (v2.1) and TODO.md. Phase 0 is complete; do not repeat it.

Amended order: inventory.py with Table 4 check and Kodak size print -> FP32 reference (Kodak 24 x QP 0-7,
twice TF32 off + once default flags) -> G0a + noise floor -> D1 INT8 weight-only smoke (first 8 Kodak sorted,
QP 0/3/7) plus identity and INT4 qmax=7 controls -> show data -> full D1 -> E1 -> CLIC all QPs for the real loop.

Controls for D1: identity control stream hashes must match FP32; INT4 qmax=7 must hurt more than INT8;
log sha256 of D1 weights original/quantized/restored (quantized differs, restored equals original); assert D1
output changes on a fixed input; mutate in place and restore from a cached clone; no second quantized model;
no per-run empty_cache(). Resume key is (config_id, dataset, image, qp), not run_id.
```

1. src/pulsequant/eval.py: import and reuse the compress/decompress/PSNR/bpp code from
   pulse_mse_baseline_benchmark.py (do not reimplement). Add a hook apply_quant(model, config) that is a no-op for FP32.
   Pin torch.backends.cudnn.allow_tf32=False, torch.backends.cuda.matmul.allow_tf32=False,
   cudnn.deterministic=True, cudnn.benchmark=False, and write them to the run manifest.
2. scripts/pipeline.py reference: Kodak, QP 0-7, three passes; G0a and noise floor.
   (resumable). Run it twice with TF32 off and once with default flags (--tf32 on). Report max |d_psnr|, |d_bpp|, and whether
   stream hashes are identical.
3. scripts/test_bitstream_safety.py: add small Gaussian noise to em_step1 weights, compress+decompress 3 Kodak images at QP 3,
   report decode_ok, PSNR, and any mismatch. (Gate G0b.)
4. scripts/pipeline.py inventory: params, shapes, groups, fan_in, kMAC/px (1088x1920 equivalent) per component in Sec. 4, written to
   artifacts/component_inventory.json. Assert it matches paper Table 4 within 1%
   (analysis 43.22, hyper enc 0.45, hyper dec 0.24, EM1 0.58, EM2 0.74, linear CDF 0.12, synthesis 3.53 kMAC/px).
   Treat the channel-attention conv as ~0 MACs and the renderer grid convs (D5-D8) as running at 1/16 of the pixel count.
5. scripts/bench_latency.py --format fp32: per-stage timing (Sec. 6.5) for one Kodak and one CLIC image.
6. Print Image.open(...).size for every Kodak file.

Stop after these six items and show me the outputs and any surprises.
```
