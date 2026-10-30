# MISCELLANEOUS - Energy and Memory Feasibility

Status: research and feasibility note. No energy/memory benchmark code has been
implemented in the quantization pipeline yet.

## 1. Why this note exists

The INT8/INT4/FP16/BF16/FP8/FP4 sweep currently records:

- bpp, PSNR, MSE
- d_PSNR, d_bpp, rate-equivalent penalty
- stream hash, recon hash
- analytic weight size and per-layer weight statistics

It does **not** record:

- energy
- CPU RAM
- GPU VRAM
- per-component latency or memory

This note tracks whether those are worth adding, and how.

## 2. What already exists

`pulse_mse_baseline_benchmark.py` already measures whole-run GPU energy and
memory:

- samples `nvidia-smi --query-gpu=power.draw,memory.used,memory.total`
- integrates power over time with the trapezoidal rule
- writes `gpu_energy_estimate_j_whole_run`
- records peak/average GPU memory

Existing baseline numbers:

| Dataset | Whole-run GPU energy | Peak GPU memory |
|---|---:|---:|
| Kodak thumbnails | 309.38 J | 383 MiB |
| CLIC | 5159.68 J | 1554 MiB |

Limitations:

- whole-run, not per-image
- 200 ms sampling is too coarse for ~100 ms operations
- total GPU power, so other processes can contaminate it
- no CPU entropy-coding energy
- not per-component

## 3. Energy: what is doable

### NVML total-energy counter

Tested on this laptop:

```text
nvidia-smi power.draw: 15.04 W
clocks.sm: 855 MHz
clocks.mem: 5500 MHz
temperature.gpu: 60 C
NVML total energy counter: supported
```

The NVML counter is the best option:

```text
energy_j = (counter_after_mJ - counter_before_mJ) / 1000
```

It avoids 200 ms sampling error. It is still whole-GPU, so:

- close other GPU apps
- measure idle counter drift
- repeat each run
- report the counter scope

### What is meaningful

| Format | Energy measurement meaningful? |
|---|---|
| FP32 | yes |
| FP16 | yes |
| BF16 | yes |
| TensorRT INT8 | yes |
| Fake INT8/INT4/FP8/FP4 | **no** |
| Simulated INT4/FP4 | **no** |

Fake quantization keeps FP32 weights and adds quantize/dequantize ops, so its
runtime and energy are not deployment energy.

### Per-component energy

Doable only as a separate isolated run:

1. Load model.
2. Warm up.
3. Read NVML counter.
4. Run only the target component repeatedly for at least 10 s.
5. Read counter again.
6. Subtract idle drift.
7. Divide by repeats and images/pixels.

This measures whole-GPU energy for that component run, not the component's isolated
energy. Report the scope explicitly.

## 4. Memory: what is doable

### Model size

Already available analytically:

- `fp32_bytes`
- `quant_bytes`
- `size_reduction_pct`
- `effective_bits_per_param`

These are theoretical sizes. For fake quantization the real stored weights remain
FP32, so the analytic size is not resident memory.

For real formats:

- FP16/BF16: actual state_dict bytes are half of FP32.
- TensorRT INT8: engine size on disk and device memory can be measured.
- INT4/FP4: only analytic unless a real backend exists.

### Process RAM

Doable with `psutil`:

```text
rss_mb = psutil.Process().memory_info().rss / 1024**2
```

Measure:

1. clean interpreter RSS
2. after model load
3. after one run
4. after `gc.collect()`
5. after `torch.cuda.empty_cache()`

Windows may not return freed memory to the OS, so RSS can stay high. A fresh
subprocess per measurement is the cleanest.

### GPU VRAM

Doable with PyTorch:

```text
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

Also record `nvidia-smi memory.used` as whole-GPU context.

### Per-component memory

Partially doable:

- weights: analytic per component
- activations: forward hooks with `max_memory_allocated` around the component
- exact per-component resident memory: not clean because buffers are shared and the
  allocator caches

For fake quantization, per-component memory change is zero because the FP32 weights
are still resident.

## 5. Verdict

| Question | Doable? | Notes |
|---|---|---|
| Energy for FP16/BF16/INT8 real formats | **Yes** | NVML total-energy counter, repeat >=10 s, idle baseline |
| Energy for fake INT8/INT4/FP8/FP4 | **No** | Runtime is not representative |
| Energy per component | **Yes, approximate** | Isolated component run; whole-GPU counter |
| Energy per image | **Yes** | Divide by image count after loop |
| Model size after quantization | **Yes, analytic** | Existing `component_size` |
| Real model size after FP16/BF16 | **Yes** | `state_dict` bytes |
| TensorRT engine size | **Yes** | Engine file bytes |
| Process RAM | **Yes** | `psutil` RSS; fresh process preferred |
| GPU VRAM | **Yes** | `torch.cuda.max_memory_allocated/reserved` |
| Per-component memory | **Approximate** | Analytic weights + activation hooks |
| Per-component latency | **Approximate** | Separate hooks/profiler run |

## 6. Proposed benchmark

Create `scripts/bench_energy_memory.py` later, not in the current sweep.

```text
--format fp32|fp16|bf16|int8
--component all|D1|E1|...
--dataset kodak|clic_professional
--qps 0 3 7
--images 1 8
--warmup 3
--repeat 10
--idle-seconds 10
--device cuda
```

For each run:

1. Fresh subprocess.
2. Record clean RSS and GPU VRAM.
3. Load model, record RSS/VRAM.
4. Warm up.
5. Read NVML energy counter.
6. Run the selected component/format/images/QPs.
7. Read NVML energy counter.
8. Record peak RSS/VRAM.
9. Restore weights, `gc.collect()`, `torch.cuda.empty_cache()`.
10. Record post-clean RSS/VRAM.

Output:

```text
format, component, dataset, qp, images,
energy_j, energy_j_per_image, energy_j_per_megapixel,
rss_clean_mb, rss_loaded_mb, rss_peak_mb, rss_after_mb,
vram_clean_mb, vram_loaded_mb, vram_peak_alloc_mb, vram_peak_reserved_mb,
model_fp32_bytes, model_quant_bytes, model_size_reduction_pct,
idle_power_w, sm_clock_mhz, mem_clock_mhz, temp_c,
status, error
```


## 8. Implemented measurement

scripts/bench_memory_energy.py measures actual data, not formula projections.

For each format/component/QP:

- actual packed weight bytes from the quantized tensors
- actual model bytes after replacing the component with its packed form
- process RSS before load, after load, after run, after cleanup
- GPU VRAM before load, after load, peak allocated, peak reserved, after cleanup
- NVML total-energy counter before/after a repeated run
- idle energy over a fixed interval, then idle power subtraction
- GPU power, SM clock, memory clock, temperature
- repeats and elapsed seconds
- status and error

Initial memory check on D1:

| Format | Component bytes | Model bytes | Reduction |
|---|---:|---:|---:|
| FP32 | 567,360 | 57,574,542 | 0.01% |
| FP16 | 283,680 | 57,290,862 | 49.96% |
| INT8 | 145,808 | 57,152,990 | 73.90% |
| INT4 | 74,888 | 57,082,070 | 86.22% |

These are actual packed bytes, not formula projections.

Energy test while the extra-format loop was running produced a negative
idle-subtracted value. That is expected: the idle baseline was contaminated by
the concurrent loop. Energy must be measured alone, after the extra loop stops.


un_memory_energy.ps1 waits for the extra-format loop to finish, then runs the
memory/energy benchmark across all formats and components.

TensorRT engine size is not implemented yet; record it as N/A until TensorRT is
installed.

## 9. What is not meaningful

- Energy for fake INT8/INT4/FP8/FP4.
- Memory reduction for fake quantization.
- Per-component energy claims from simulated formats.
- Per-image energy from a 200 ms sampler.
- Per-component memory as an exact number.

## 10. Recommendation

Do energy and memory as a **separate deployment benchmark**, after the quality
sweep. Start with:

1. FP32
2. FP16
3. BF16
4. TensorRT INT8

Report energy and memory only for those. Keep fake INT8/INT4/FP8/FP4 as quality
and analytic-size results.

## 11. Current decision

- Deferred for now. The current priority is the quality/rate/size/correctness analysis across precisions.
- After ILP or ILP+QAT, run the memory/energy benchmark on the selected real-format configurations.
- Compare those results against the FP32 baseline.
