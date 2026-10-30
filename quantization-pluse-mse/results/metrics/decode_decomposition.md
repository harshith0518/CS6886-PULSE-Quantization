# Decode decomposition: what the entropy bucket actually contains

The earlier stage profile reported "decode entropy 80 ms" by timing `model.decompress` directly.
That call is not pure rANS: it also runs the entropy-model network and rebuilds coding tables. This
document separates those, because the difference decides what can be optimised.

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\profile_decode_stages.py --repeats 5
```

Raw rows: `profile_decode_stages.csv`. Kodak `kodim01.png`, QP3, crops of 256, 384 and 512 square.
Symbols are `y_shape[1]*y_shape[2]*y_shape[3]`, so 81,920 / 184,320 / 327,680.

## Stage decomposition

| Device | Crop | MP | Symbols | Decode total | `model.decompress` | Synthesis | Unaccounted |
|---|---|---:|---:|---:|---:|---:|---:|
| CPU/1 | 256 | 0.066 | 81,920 | 71.7 | 47.3 | 21.3 | +3.1 |
| CPU/1 | 384 | 0.148 | 184,320 | 115.2 | 74.8 | 44.0 | -3.7 |
| CPU/1 | 512 | 0.262 | 327,680 | 200.1 | 105.2 | 105.0 | -10.3 |
| CUDA/1 | 256 | 0.066 | 81,920 | 69.9 | 49.1 | 9.3 | +11.3 |
| CUDA/1 | 384 | 0.148 | 184,320 | 86.2 | 59.7 | 21.7 | +4.6 |
| CUDA/1 | 512 | 0.262 | 327,680 | 88.8 | 87.0 | 18.2 | -16.5 |

Two-term fit of `model.decompress` against symbol count:

| Device | Fixed cost | Marginal cost |
|---|---:|---:|
| CPU/1 | **29.4 ms** | 234 ns/symbol |
| CUDA/1 | **34.3 ms** | 157 ns/symbol |

Unaccounted time stays inside ±17 ms and flips sign, so there is **no hidden I/O or Python glue
reservoir**. The earlier "entropy" number was real work, not a residual.

## What the profile attributes on CPU/1

`cProfile` of three `decompress_image` calls, per call:

| Function | Per call | What it is |
|---|---:|---|
| `torch.conv2d` (153 calls) | 100 ms | synthesis network, the decoder convolutions |
| `int8scale.packed_indexes` | **37 ms** | per-call index packing in the entropy path |
| `set_y_cdf` | **9 ms** | per-call CDF table build |
| `set_z_cdf` | **9 ms** | per-call CDF table build |
| `torch.pixel_shuffle` (15 calls) | 13 ms | synthesis upsampling |

So roughly **55 ms per call is per-call setup** in the entropy path, against a synthesis network
that costs 100 ms on one CPU thread and only about 8-18 ms on the GPU.

## Conclusions, with their confidence

**Confirmed: a large fixed per-call cost.** 29-34 ms that does not depend on image size, plus about
55 ms of attributed table building and index packing on CPU. This is the single most actionable
finding, and it is invisible in the earlier profile because the cost sits inside `model.decompress`.

**Confirmed: the marginal coding rate is far too slow.** 157-600 ns per symbol across every
measured configuration. A competently written scalar rANS decoder runs at single-digit
nanoseconds per symbol, so this is one to two orders of magnitude off. The extension is present and
exports `RansDecoder` / `RansEncoder`, so this is not a missing-extension fallback; it looks like a
slow build or an unoptimised inner loop.

**Not confirmed: that the two-term model is reliable.** The fit is poor at the largest crop, and the
full 768x512 decode measured 89.8 ms where the fit predicts 111 ms. With three points and no error
bars, the fixed/marginal split is indicative, not established. Re-measure with more sizes and
repeats before quoting it.

**Not established: how much is rANS versus the entropy network.** `model.decompress` was not
instrumented internally; the split between the C++ decode and the entropy-model forward inside it
is still unknown. `packed_indexes`, `set_y_cdf` and `set_z_cdf` are attributed, but the rANS loop
itself is not isolated.

## What this changes

The earlier statement "73% of round trip is entropy coding" must be scoped: it holds for the GPU
configuration. On CPU/1 the network is about 50% of decode and entropy about 50%, so the ranking of
levers is device-dependent.

The priority is not multi-lane rANS yet. It is:

1. Hoist the per-call CDF and index setup out of the decode call, if it is genuinely recomputed per
   call. Expected saving up to ~55 ms on CPU and an unknown but probably similar share on GPU.
   Bitstream unchanged.
2. Check the extension build: Release versus Debug, optimisation flags, and whether the inner loop
   is the fast path. Confirmed cheap to check, because the exports are present.
3. Only then consider multi-lane rANS, which changes the bitstream format.

## Reproduce

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\profile_decode_stages.py --repeats 5
```
