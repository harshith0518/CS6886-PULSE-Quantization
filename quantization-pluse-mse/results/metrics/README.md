# Latency, energy and memory measurements

Two benchmarks live here. Only the second is valid for comparing configurations; the first is kept
because it documents why.

## What was measured

| Metric | Source | Notes |
|---|---|---|
| round-trip latency | `time.perf_counter` around `compress_image` + `decompress_image` with `torch.cuda.synchronize()` on both sides | single thread, `configure_runtime(device, 1)` |
| encode / decode split | separate timestamps around each call | reported per configuration |
| GPU energy | `nvmlDeviceGetTotalEnergyConsumption` delta across the measured block | whole-device counter, so any other GPU work is included |
| power | `nvmlDeviceGetPowerUsage` sample | one sample per block, indicative only |
| peak VRAM | `torch.cuda.max_memory_allocated()` after resetting stats | includes the harness's FP32 weight clone |
| SM clock, temperature | NVML | recorded to show thermal state |

## Benchmark 1: `bench_kodak.csv` — invalid for comparison, kept as evidence

`scripts/bench_configs.py` runs each configuration as one long block, in sequence. Repeating the
FP32 baseline as a drift control at the start and the end of the same run gives:

| FP32 block | round-trip | energy per image |
|---|---:|---:|
| first | 248.5 - 262.0 ms | 3.52 - 3.65 J |
| last | 308.6 - 323.3 ms | 4.17 - 4.33 J |

The same model, same data, same code moved by **+25% latency and +20% energy** purely from position
in the run. So every cross-configuration difference in that file is confounded with throttling.
It must not be used to compare anything. It is kept because that control is the reason the second
benchmark exists.

## Benchmark 2: `interleaved_kodak.csv` — the valid comparison

`scripts/bench_interleaved.py` measures all configurations **per image and per round**, shuffling
the order within each cell, so drift hits every candidate equally. Each candidate is paired with the
FP32 measurement from the same image/round/QP cell, and the median of paired percentage deltas is
reported with a 2000-sample bootstrap 95% CI.

10 Kodak images, 5 rounds, QPs 0/3/7, FP32 as the reference. 50 pairs per cell.

| Config | QP | latency vs FP32 | 95% CI | energy vs FP32 | 95% CI |
|---|---:|---:|---|---:|---|
| ILP mixed | 0 | +5.66% | +1.12 to +12.91% | -0.74% | -2.61 to +7.71% |
| ILP mixed | 3 | +4.79% | +1.27 to +9.80% | -0.39% | -2.24 to +7.76% |
| ILP mixed | 7 | +4.35% | -0.60 to +7.92% | +0.43% | -2.30 to +7.85% |
| FP16 autocast | 0 | +5.58% | +2.46 to +13.81% | -12.00% | -10.46 to +0.09% |
| FP16 autocast | 3 | +4.29% | +1.18 to +9.72% | -9.99% | -15.14 to -5.28% |
| FP16 autocast | 7 | +4.51% | +0.46 to +8.77% | -9.81% | -12.75 to -4.68% |
| BF16 autocast | 0 | -1.20% | -1.16 to +11.49% | -10.70% | -14.34 to -1.02% |
| BF16 autocast | 3 | -1.55% | -2.69 to +4.02% | -8.61% | -13.43 to -4.64% |
| BF16 autocast | 7 | +1.82% | -0.33 to +7.37% | -8.86% | -10.38 to -2.20% |

### Reading it

**Nothing got faster.** There is no configuration with a latency reduction whose confidence interval
excludes zero. The mixed model is 4-6% *slower*, and even genuine half precision is neutral to
slightly slower on this card.

**FP16 and BF16 reduce energy by about 9-12%** with intervals that exclude zero at QP3 and QP7. That
is the one measured efficiency gain in the whole study. It comes from real half-precision kernels,
not from quantization.

**The ILP model does not change energy.** Its energy delta interval spans zero at every QP, which is
exactly what the mechanism predicts: the weights are still FP32 and the kernels are unchanged.

**Why ILP shows a small latency penalty is not fully explained.** Fake quantization should be
latency-neutral by construction, since it only rewrites weight values. The likely mechanism is that
different weight values change the encoder's latent statistics and therefore the amount of work in
the rate search and entropy coding, but this was not isolated experimentally. Treat the +5% as
observed, not explained.

### Caveats

- One laptop GPU, one driver, one thermal state. The drift in benchmark 1 shows how sensitive this
  hardware is.
- Whole-device energy counter: any other GPU activity on the machine is included.
- Peak VRAM for the fake-quant configurations (about 281 MB vs 228 MB for FP32) is a **harness
  artifact**: `apply_assignment` holds an FP32 clone of the original weights so it can restore them.
  A deployed quantized model would not carry that clone. Do not read it as a memory regression.
- 10 images out of 24, three QPs, single thread. Not a throughput characterisation.

## Reproduce

```powershell
# invalid on its own, but shows the drift
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\bench_configs.py --dataset kodak `
  --configs fp32 fp16 bf16 int8 safe greedy ilp fp16_autocast bf16_autocast fp32 --qps 0 3 7

# the paired benchmark
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\bench_interleaved.py `
  --images 10 --rounds 5 --qps 0 3 7
```

Make sure no stray `python.exe` is running before either: an orphaned benchmark process was found
holding roughly three quarters of a core during the first attempt, which is one of the reasons that
run was unusable.
