# Prompt for an external review of the latency problem

Copy the block below into a fresh Claude conversation. Attach these files if the interface allows,
because they are the evidence the prompt refers to:

- `results/metrics/latency_profile.md`
- `results/metrics/profile_kodak.csv`
- `results/metrics/interleaved_paired_kodak.csv`
- `codec/runtime.py`
- `codec/images.py`

Everything the prompt needs is restated inside it, so it works without the attachments too.

---

```text
I need a rigorous, evidence-backed plan for reducing decode latency in a learned image codec.
Do not give generic optimisation advice. Every recommendation must state: the mechanism, the
expected magnitude on my hardware, the exact measurement that would confirm it, and the evidence
that would falsify it. Where you are uncertain, say so and say what would resolve it. Do not
invent benchmark numbers.

## The system

PULSE (Jia et al., arXiv 2605.04563), a learned image codec explicitly designed for low-latency
single-thread CPU decoding. One weight file serves 8 quality levels (QP 0-7).

- Encoder: patch embedding + 4 DCB blocks + latent projection. 12.13M params (85.6%).
- Entropy model: hyperprior encoder/decoder, two entropy-model steps, a fitted meta-prior lookup
  table, and an integer linear CDF decoder. ~1.34M params (9.4%). Entropy coding is bit-exact and
  runs through a compiled C++ pybind11 extension.
- Decoder: 2 DCB blocks + a renderer MLP (conditioned adaptive). 0.71M params (5.0%).
- Total 14.18M neural params. Published complexity of the receiver: ~5.2 kMAC/pixel.
- The paper claims 1080p decode in 126 ms on a single CPU thread.

## My measurements

Setup: RTX 3050 Laptop 4GB, CUDA 12.4, PyTorch 2.6.0, Windows. Kodak images at 768x512
(0.393 MP), QP3, batch 1, `torch.set_num_threads` set per row. Stage split obtained by timing
`model.encode` / `model.decode` separately from the full `compress` / `decompress` calls, medians of
2 runs on 2 images. I trust the structure of these ratios; I do not fully trust the absolute ms.

| device | threads | encode total | encode NN | encode entropy | decode total | decode entropy | decode NN |
|---|---|---|---|---|---|---|---|
| CPU | 1 | 839 ms | 701 | 138 | 211 | 105 | 107 |
| CPU | 4 | 367 ms | 308 | 59 | 157 | 106 | 49 |
| CUDA | 1 | 131 ms | 51 | 80 | 88 | 80 | 8 |
| CUDA | 4 | 123 ms | 42 | 81 | 98 | 90 | 9 |

Key observations:

1. On GPU, decode is 80 ms entropy + 8 ms neural network: 91% entropy.
2. The entropy stage is nearly invariant to device and thread count (80 ms GPU, 105 ms CPU/1,
   106 ms CPU/4), while the neural stages scale strongly with threads (701 -> 308 ms on CPU).
3. Round trip is ~73% entropy, ~27% neural network.

## Already ruled out, with the reason

- **Weight quantization / mixed precision.** I solved a per-component precision assignment with an
  MILP (INT8/FP16/BF16 mix, 60.4% smaller weights, validated on 3 datasets). Measured with a paired
  interleaved benchmark, 50 pairs per cell, bootstrap CI: latency 4-6% WORSE, energy unchanged.
  Structural reason: convolutions are 9% of decode.
- **FP16/BF16 autocast.** Energy improved 9-12% (CI excludes zero), latency neutral to worse
  (cast overhead exceeds tensor-core gain at these shapes).
- **Quantizing the decoder specifically.** The target is 8 ms of an 88 ms decode.

## Open facts

- `decompress_image` reads `metadata.get("entropy_lanes", 1)`. The encoder never emits more than one
  lane. The bitstream format anticipates multi-lane entropy coding.
- The C++ entropy decoder appears to run single-threaded; it does not respond to
  `torch.set_num_threads`.
- Existing `.pulse` bitstreams exist and must stay decodable.
- No TensorRT, no ONNX runtime, no GPU inference server. Everything is PyTorch + the C++ extension.

## Questions

1. **Is my operating point simply wrong?** My CPU single-thread decode is 536 ms per megapixel
   (211 ms for 0.393 MP) versus the paper's 61 ms per megapixel (126 ms for 2.074 MP) - 8.8x worse.
   Rank the candidate explanations and, for each, give the specific check that confirms or eliminates
   it. Consider at least: measurement methodology and whether the published figure excludes entropy
   coding; thread configuration and BLAS/OpenMP behaviour; padding overhead at my image size;
   first-call warmup or thread-pool effects; the extension's fast path not being exercised
   (`entropy_lanes`, `skip_index_cutoff`, CDF mode); image content and bitrate; and the possibility
   that the published number is decoder-network-only.

2. **Rank concrete approaches to cut decode latency** on this hardware. For each: mechanism,
   expected ms saved out of an 88 ms decode, implementation effort, whether it changes the bitstream
   format, and what would falsify the expected gain. Include, but do not limit yourself to,
   multi-lane or parallel entropy decoding, SIMD or lookup-table CDF approaches, reducing coded
   symbol count, operator fusion, CUDA graphs, batching, and moving the entropy stage to a different
   execution model.

3. **Design the single best next experiment.** Give the exact protocol: sample size, how to control
   for the thermal drift I observed (the same FP32 configuration varied 248 -> 308 ms across one
   run), what to hold fixed, what to vary, the primary metric, and the decision rule for abandoning
   the approach.

4. **Which of my numbers are most likely to be wrong**, and in what order should I re-measure them?

5. **What may I claim in a report** about latency and energy given this evidence, and what must I
   explicitly not claim?

Constraints: 4 GB VRAM laptop, Windows, prefer existing dependencies (PyTorch, scipy, a C++
extension already in the repo). Any change that breaks bitstream compatibility must be flagged as
such. I am not asking for a rewrite of the codec; I am asking where the next 20 ms can credibly
come from, and how to prove it.
```

---

## How to get the most out of it

1. **Do not trim the measured table.** Generic answers appear when the model has no numbers to
   anchor on.
2. **Ask a follow-up specifically on question 1.** The 8.8x gap against the published figure is the
   most informative anomaly in this project: either our setup is wrong, or the published number
   measures something narrower than we assume. Both outcomes change the report.
3. **Push back on any recommendation without a magnitude.** "Use CUDA graphs" is worthless here
   unless it says how many milliseconds of an 88 ms decode and why.
4. If it proposes multi-lane entropy coding, ask explicitly what breaks for existing `.pulse` files
   and whether the decoder can auto-detect the lane count from the metadata.
