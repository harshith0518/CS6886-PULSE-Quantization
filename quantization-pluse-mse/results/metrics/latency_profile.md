# Where the latency actually is

Stage-level breakdown, because "the model is slow" is useless without knowing which stage.

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\profile_latency.py --images 3 --repeats 3
```

Measured on 2 Kodak images at QP3, median of 2 runs each. Small sample: the *structure* it shows is
far larger than any plausible noise, but exact milliseconds should be re-measured with more repeats
before being quoted.

| Device | Threads | Encode | ... NN | ... entropy | Decode | ... entropy | ... NN |
|---|---|---:|---:|---:|---:|---:|---:|
| CPU | 1 | 839 ms | 701 | 138 | 211 | 105 | 107 |
| CPU | 4 | 367 ms | 308 | 59 | 157 | 106 | 49 |
| **CUDA** | **1** | **131 ms** | **51** | **80** | **88 ms** | **80** | **8** |
| CUDA | 4 | 123 ms | 42 | 81 | 98 | 90 | 9 |

## The finding

On the GPU, which is how every number in this project was measured:

- **Decode is 91% entropy coding.** 80 ms of 88 ms. The synthesis network is **8 ms**.
- **Encode is 61% entropy coding.** 80 ms of 131 ms; the analysis network is 51 ms.
- On the GPU, **73% of a full round trip is in the entropy stage** and 27% in the neural networks.

**Scope warning.** That 73% figure is GPU-only. On one CPU thread the split is about 50/50
(network 100 ms, entropy 98 ms at 512x512), so any ranking of optimisation levers must state which
device it applies to. The entropy stage also absorbs per-call CDF and index setup, which
`decode_decomposition.md` separates out: of the "entropy" time, roughly 55 ms per call on CPU is
table building rather than coding.

The entropy stage is the integer CDF decoder in the C++ extension, and its cost is essentially
**independent of device and thread count** (80 ms on GPU, 105 ms on one CPU thread, 106 ms on four).
It does not scale with threads because it runs single-threaded and only the surrounding torch work
responds to `torch.set_num_threads`.

## Why this settles the quantization question

Weight quantization, FP16, BF16, INT8 and TensorRT all accelerate **convolutions**. On the decode
path those convolutions are 8 ms of 88 ms. Even a hypothetically perfect 4x faster kernel saves
6 ms, about 7% of decode, and nothing at all in the entropy stage.

That is the structural reason the mixed-precision model showed **no latency gain** — it is not a
measurement failure, it is arithmetic. Quantization in this codec is a *footprint* technique. It
cannot be a latency technique.

## Levers, ranked by measured headroom

| Lever | Headroom | Effort | Notes |
|---|---|---|---|
| Parallelise the entropy decoder | up to ~80 ms of an 88 ms decode | high | single largest lever by far; the stream format already carries an `entropy_lanes` field that defaults to 1 |
| Fewer coded symbols | proportional to bits | high | requires rate-model or architecture changes, i.e. retraining |
| FP16/BF16 for the encode NN | ~51 ms of encode is NN | low | measured: energy -9 to -12%, latency neutral to worse |
| INT8 kernels (TensorRT/ONNX) | ~51 ms encode NN, ~8 ms decode NN | high | the ceiling is 27% of round trip, not 27% of decode |
| Quantization of the decoder | ~8 ms | none | pointless: the target is already 9% of decode |
| More CPU threads | 839 to 367 ms encode | none | helps the NN only, and only on CPU |

The single highest-value experiment is not a quantization experiment at all: it is whether the
entropy decoder can use more than one lane. `decompress_image` reads
`metadata.get("entropy_lanes", 1)`, so the format anticipates it, but `compress_image` never emits
more than one lane and the C++ side was not exercised with more. That changes the bitstream
contract, so it needs an explicit decision and a compatibility check, not a silent edit.

## Caveats

- Two images, QP3 only, medians of two runs. Re-measure with more repeats before quoting milliseconds.
- The encode NN figure for CPU/4-threads (308 ms) exceeds the encode total for one of the two
  images, because the two were timed separately and the machine drifted; treat CPU numbers as
  indicative.
- Entropy is derived by subtraction for encode (total minus NN), so it includes any overhead shared
  between the two, such as packing the bitstream.
