# The mixed-precision model against the FP32 base model

Every number here is measured through the real encode/decode path at all 8 QPs on three
independent image sets. Nothing is extrapolated from single-component experiments, and the FP32
gate confirms the harness reproduces the frozen reference bit-exactly (`gate_*.json`).

Configurations compared:

| Name | What it is |
|---|---|
| FP32 base | unmodified `pulse-mse`, the reference |
| FP16 / BF16 uniform | every one of the 19 components cast, the naive baselines |
| INT8 uniform | every component per-output-channel INT8, the naive aggressive baseline |
| safe mixed | each component pushed to its own 1% limit, 17.904 MB |
| greedy mixed | greedy selection, 24.565 MB |
| **ILP mixed** | **MILP optimum at a 1% worst-QP budget, 21.677 MB** |

## 1. Model size: the headline gain

| Model | Weights | Saving |
|---|---:|---:|
| FP32 base | 54.688 MB | — |
| FP16 / BF16 uniform | 27.416 MB | 49.9% |
| greedy mixed | 24.565 MB | 55.1% |
| **ILP mixed** | **21.677 MB** | **60.4%** |
| safe mixed | 17.904 MB | 67.3% |
| INT8 uniform | 13.904 MB | 74.6% (but see BD-rate) |

Where the 21.677 MB actually sits, in the ILP model:

| Part | Weight | Share |
|---|---:|---:|
| E1 + E4 at FP16 | 11.60 MB | 53.5% |
| E2 + E3 at INT8 | 5.86 MB | 27.0% |
| the other 15 components | 4.22 MB | 19.5% |

Two blocks carry over half the remaining cost because they are the only large components still at
16-bit. That is the single biggest lever left, and it is what QAT would attack.

## 2. Bitstream quality: BD-rate against FP32

Negative would mean fewer bits than FP32 at equal PSNR. No configuration achieves that.

| Config | Kodak | CLIC | Set14 | BD-PSNR (dB), Kodak / CLIC / Set14 |
|---|---:|---:|---:|---|
| FP16 uniform | +0.016% | +0.004% | +0.007% | -0.0007 / -0.0002 / -0.0003 |
| BF16 uniform | +0.084% | +0.076% | +0.071% | -0.0038 / -0.0030 / -0.0028 |
| INT8 uniform | +10.205% | +18.707% | +9.389% | -0.4426 / -0.6826 / -0.3577 |
| safe mixed | +1.457% | +2.318% | +1.511% | -0.0651 / -0.0886 / -0.0591 |
| greedy mixed | +0.299% | +0.225% | +0.278% | -0.0134 / -0.0087 / -0.0109 |
| **ILP mixed** | **+0.477%** | **+0.552%** | **+0.486%** | **-0.0214 / -0.0213 / -0.0191** |

![BD-rate](comparison_bd_rate.png)

Uniform INT8 is 20-40x worse than the mixed assignments. FP16 is effectively free.

## 3. Where the ILP model is better

**Weight footprint.** -60.4% with no retraining, versus -49.9% for uniform FP16. At equal size the
choice is stark: 21.677 MB at +0.48% BD-rate beats 27.416 MB at +0.02%, or you can read it the
other way and say FP16 buys the first half of the compression for almost nothing.

**Bitrate at high QP.** This is the non-obvious win. At QP4-QP7 the quantized encoder produces
*fewer* bits than FP32:

| QP | Kodak d_bpp | CLIC d_bpp | Set14 d_bpp |
|---:|---:|---:|---:|
| 0 | +0.335% | +0.478% | +0.319% |
| 3 | -0.036% | +0.028% | +0.044% |
| 4 | -0.167% | -0.081% | -0.146% |
| 5 | -0.274% | -0.211% | -0.215% |
| 7 | -0.419% | -0.268% | -0.250% |

Perturbing the encoder marginally changes the latents in a way that makes them slightly cheaper to
code. So at the quality settings that matter in practice, the model is simultaneously 60% smaller
and produces a smaller file.

**Transfer.** The assignment was fitted on Kodak and CLIC only, yet Set14 lands within 0.05% of the
other two on BD-rate. The result is not overfitted to the fitting sets.

## 4. Where it is worse

**PSNR, everywhere, and worsening with QP.** The loss grows monotonically from -0.004 dB at QP0 to
-0.063 dB at QP7 on Kodak. Peak measured loss across all three sets is **-0.0634 dB**.

![per-QP deltas](comparison_ilp_dpsnr.png)

**ILP is not the best BD-rate.** Greedy has a better BD-rate (+0.225 to +0.299%) than ILP
(+0.477 to +0.552%) because greedy spends 2.9 MB more on weights. This is a genuine trade-off, not
a bug:

| If you care about | Pick | Why |
|---|---|---|
| Weight footprint, memory budget | ILP, 21.677 MB | smallest model inside 1% |
| Bitstream efficiency | greedy, 24.565 MB | about half the BD-rate penalty |

Both are inside the 1% penalty budget on every dataset. The earlier claim that ILP is simply "the
recommended operating point" needs that qualifier.

**The fragile components stay expensive.** H0, H3 and D5 cannot go below FP16, so the entropy
network and renderer MLP contribute fixed cost with no compression available inside the 1% budget.

## 5. Speed and energy: measured, and nothing got faster

These numbers did not exist when this document was first written. They now do, and they were
produced with a paired design, because a naive sequential benchmark turned out to be unusable: on
this laptop the *same* FP32 configuration measured 248 ms at the start of a run and 308 ms at the
end, a 25% drift from throttling. See `results/metrics/README.md` for that control and for the
interleaved protocol.

Median paired deltas against FP32, 10 Kodak images, 5 rounds, 50 pairs per cell, bootstrap 95% CI:

| Config | QP | Latency | Energy |
|---|---:|---|---|
| ILP mixed | 0 | **+5.66%** (+1.12 to +12.91) | -0.74% (-2.61 to +7.71) |
| ILP mixed | 3 | **+4.79%** (+1.27 to +9.80) | -0.39% (-2.24 to +7.76) |
| ILP mixed | 7 | +4.35% (-0.60 to +7.92) | +0.43% (-2.30 to +7.85) |
| FP16 autocast | 3 | +4.29% (+1.18 to +9.72) | **-9.99%** (-15.14 to -5.28) |
| FP16 autocast | 7 | +4.51% (+0.46 to +8.77) | **-9.81%** (-12.75 to -4.68) |
| BF16 autocast | 3 | -1.55% (-2.69 to +4.02) | **-8.61%** (-13.43 to -4.64) |
| BF16 autocast | 7 | +1.82% (-0.33 to +7.37) | **-8.86%** (-10.38 to -2.20) |

Readings:

- **No configuration is faster.** Not one latency delta has a confidence interval entirely below
  zero. The mixed model is 4-6% slower; genuine half precision is neutral to slightly slower.
- **FP16 and BF16 cut energy by about 9-12%**, with intervals excluding zero at QP3 and QP7. This is
  the only measured efficiency gain in the study, and it comes from real half-precision kernels,
  not from quantization.
- **The ILP model does not change energy** (intervals span zero at every QP), exactly as the
  mechanism predicts: the weights are still FP32 and the kernels are unchanged, so the 60% size
  saving buys zero runtime benefit as built.
- The ILP latency penalty is **observed but not explained**. Fake quantization should be neutral;
  the plausible mechanism is that changed weight values alter latent statistics and therefore the
  work done in rate search and entropy coding. It was not isolated experimentally.
- Peak VRAM is higher for fake-quant configurations (about 281 MB vs 228 MB), but that is a harness
  artifact: the quantizer keeps an FP32 clone so it can restore weights. Not a memory regression.

The honest one-line answer to "how much faster": **it is not faster at all.** It is 60.4% smaller
in bytes, 9-12% more energy-efficient only if you additionally run real FP16/BF16 kernels, and
marginally slower in wall-clock time as shipped.

What would change that:

| Path | Status | Expected |
|---|---|---|
| INT8 kernels (TensorRT or ONNX Runtime on the E2/E3 blocks) | not built; needs ONNX export | the only route to a real INT8 speedup |
| Simulated formats (FP8, FP4, INT4) | by definition no speed claim | forbidden |

## 6. Summary

| Dimension | Verdict |
|---|---|
| Model size | **60.4% smaller**, 54.688 to 21.677 MB |
| Bitstream at equal PSNR | **+0.48% worse** (Kodak), +0.55% (CLIC), +0.49% (Set14) |
| Bitrate at QP4-QP7 | **0.08% to 0.42% smaller files** |
| PSNR | -0.004 to -0.063 dB, worst at QP7 |
| Robustness across datasets | stable within 1.4x, vs 2.9x for uniform INT8 |
| Latency | **4-6% slower** as built; no configuration is faster |
| Energy | **unchanged** for the mixed model; -9 to -12% only with real FP16/BF16 kernels |
| Retraining required | none |

## Files

- `bd_rate.csv`, `comparison_ilp_vs_fp32.csv`: the numbers above
- `comparison_rd_curves.png`, `comparison_bd_rate.png`, `comparison_ilp_dpsnr.png`: three figures
- `sweep_<dataset>.csv`: raw per-image, per-QP rows behind all of it
