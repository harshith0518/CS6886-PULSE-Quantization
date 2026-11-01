# STATUS - project checkpoint

Last updated 2026-11-01. HEAD `c253418`, 10/10 unit tests passing. All experiments are complete; the
work that remains is the report, the slides and the submission package.

Read this first after any break. `RESOURCES.md` maps every claim to the file that proves it.

## 1. What this project is

Component-wise quantization study of the released `pulse-mse` model from PULSE (Jia et al.,
arXiv:2609.18602), plus an optimization stage and a my fine-tuning stage. The question: which
components tolerate which precision, at which QP, what a per-component assignment buys, and whether
quantization-aware fine-tuning recovers what the assignment costs.

## 2. Finished and committed

| Area | State | Where |
|---|---|---|
| Weight-only sweep | complete, 7 formats x 19 components x {kodak, clic} x 8 QPs = 69,233 rows, 0 failures | `results/_ledger/` |
| FP32 reference | 4 datasets x 3 passes, noise floor recorded | `results/_ledger/reference/` |
| Decision surface | 6 figures + decision card + `decision_summary.csv` | `results/decision/` |
| ILP solver | HiGHS, per-QP constraints, shared-across-QP variables | `results/ilp/` |
| Whole-model evaluation | 7 configs x 8 QPs x 3 datasets, FP32 gate passes on all three | `results/mixed/` |
| Additivity test | holds: 0.96-1.05 Kodak, 1.02-1.11 CLIC | `results/mixed/additivity.csv` |
| BD-rate comparison | 3 datasets, per-QP tables, 3 figures | `results/mixed/COMPARISON.md` |
| Latency / energy | paired interleaved benchmark + stage profile | `results/metrics/` |
| Provenance | source segregation with verification tags | `RESOURCES.md` |
| Group-scaled 4-bit | `int4g32` implemented; 5.0 effective bits per weight | `src/pulsequant/core.py` |
| QAT pipeline | AL controller, STE convolutions, scoped freezing, bundle export | `scripts/qat_run.py` |
| QAT results | three tiers at 2,000 and 12,000 steps, three datasets | `results/final/`, `results/night/` |
| Fine ILP sweep | 1.2 / 1.4 / 1.6 / 1.8 percent solved; 1.6 and 1.8 measured | `results/night/ilp_fine_measured.json` |
| Unattended pipeline | four stages, resumable, sleep-guarded | `scripts/run_all_stages.py` |

## 3. The numbers

**The frontier.** Three measured operating points, each on all three datasets with the same harness,
the same eight QPs and the same FP32 reference:

| Point | Size | Saving | Kodak | CLIC | Set14 |
|---|---|---|---|---|---|
| 1% budget champion | 21.677 MB | -60.36% | 0.672% | 0.851% | 0.935% |
| 1.8% budget variant | 18.664 MB | -65.87% | 1.268% | 1.595% | 1.601% |
| INT8 tier + QAT 12k | 14.401 MB | -73.67% | 1.848% | 2.874% | 3.363% |

BD-rate for the 1 percent champion: **+0.477 / +0.552 / +0.486 percent**. It was fitted on Kodak and
CLIC only; Set14 is a genuine transfer test and it holds.

**Why 1.8 percent replaced 2 percent.** The knee sits between 1.6 (21.151 MB) and 1.8 (18.664 MB),
where 2.5 MB falls away. The 1.8 assignment dominates the 2 percent one that was previously chosen:
0.275 MB larger but better on all three datasets (1.27/1.60/1.60 versus 1.40/1.94/1.71).

**Reference points for the same harness:**

| Config | Size | Kodak | CLIC | Set14 |
|---|---|---|---|---|
| FP16 uniform | 27.416 MB | 0.038% | 0.047% | 0.056% |
| greedy mixed | 24.565 MB | 0.386% | 0.358% | 0.393% |
| safe mixed (PTQ only) | 17.904 MB | 2.633% | 4.301% | 2.810% |
| uniform INT8 | 13.904 MB | 26.184% | 56.701% | 19.374% |

Mixed precision at 21.677 MB costs 0.94 percent where uniform INT8 at a *smaller* size costs 26.18
percent on Kodak: a **28x quality gap at comparable or better compression**. That single comparison is
the argument for the whole project.

**QAT: what it recovered.** Augmented-Lagrangian controller, damage removed relative to each tier's
own pre-QAT state:

| Tier | Size | 2,000 steps | 12,000 steps |
|---|---|---|---|
| More INT8 | 14.401 MB | 8.61% -> 8.61% (0.0%) | 8.61% -> **3.36%** (61.0%) |
| Moderate INT4 | 13.572 MB | 119.18% -> 25.27% (78.8%) | 119.18% -> **20.53%** (82.8%) |
| More INT4 | 11.369 MB | 1771.71% -> 84.22% (95.2%) | 1771.71% -> **41.84%** (97.6%) |

**QAT on the ILP 2 percent assignment recovered exactly nothing**: 1.936% -> 1.936% over 8,000 steps.
A solver-optimised assignment leaves nothing for fine-tuning to repair, because its damage is already
spread optimally. This is the cleanest statement of the project's finding: **QAT recovers what the
assignment leaves on the table.**

Note the correction: an earlier reading claimed QAT only helps severe damage, based on the INT8 tier
gaining nothing at 2,000 steps. The 12,000-step run falsified that. The 2,000-step negative was a
step-budget artifact.

**QAT cost.** 12,000 steps at 512 crops took 98.7-120.5 minutes per tier on the RTX 3050, about
0.49 s per step with batch 1. Against pretraining that is **2.2 percent of the optimizer steps and
0.2 percent of the sample budget** (paper: 500,000 steps at 512 with batch 8, then 50,000 at 1024 with
batch 8, about 5.6M 512-equivalent crops).

**Latency and energy** (paired, interleaved, 50 pairs per cell):

- ILP mixed is **4-6% slower** than FP32, not faster. Energy unchanged.
- Real FP16/BF16 autocast: energy **-9 to -12%**, latency neutral to worse.
- Stage profile at QP3 on GPU: decode 88 ms = 80 ms entropy + 8 ms network; encode 131 ms = 80 entropy
  + 51 network. **73% of round trip is entropy coding.** On CPU/1 it is closer to 50/50.
- Why quantization cannot fix latency: it accelerates convolutions, and convolutions are 27% of round
  trip and only 9% of decode. Arithmetic, not a measurement artefact.

## 4. Open items

| Item | Status | Notes |
|---|---|---|
| **Report, slides, submission package** | **in progress** | the graded deliverable |
| Checkpoint selection under-reports QAT gains | open | the validation probe scored the INT8 12k model as equal to its start (-0.08%) while the full evaluation shows 8.61 -> 3.36. Selection uses 6 pool crops at 512; the real metric uses native resolution |
| Entropy multi-lane decision | open | largest latency lever (~80 ms of an 88 ms decode). Changes the bitstream contract, so needs an explicit decision |
| W8A8 activations | not implemented | only weights were quantized |
| INT8 kernels (TensorRT/ONNX) | not built | the only route to a real INT8 speedup |
| QAT for tiers 2/3 beyond 12,000 steps | open | tier 3 was still improving at its last checkpoint |

## 5. Known measurement traps (do not repeat)

1. **Never compare configurations from a sequential benchmark.** The same FP32 config drifted from
   248 ms to 308 ms (+25%) and 3.5 to 4.3 J (+20%) within one run. Use `bench_interleaved.py`.
2. **Check for orphaned `python.exe` before benchmarking.** A leftover process held ~75% of a core.
3. **Never sum each component's own worst QP.** That gave 65.3% where the true value was 26.7%.
4. **Peak VRAM for fake-quant configs is a harness artefact** (+52 MB from the restore clone).
5. **Windows schannel is broken** (`SEC_E_CREDENTIALS`); Python `urllib` works.
6. **Never floor a metric at zero.** The QAT checkpoint selector accumulated `max(0.0, penalty)`, so a
   negative penalty (a genuine improvement) could never be selected. Every run before the fix saved
   its step-0 weights and discarded its own gains, which is why QAT looked like it failed five times.
7. **The training surrogate rate is not the stream bitrate.** `rates["sq_loss"]` runs about 0.8 while
   measured stream bpp is about 0.32 on the same model. Comparing the two makes an augmented-Lagrangian
   multiplier diverge on a phantom error.
8. **`compress_image` is decorated with `torch.inference_mode()`.** Calling it between training steps
   caches inference tensors on the entropy model and breaks the next backward pass with "inference
   tensors cannot be saved for backward". Call the model's own methods under `no_grad` instead.
9. **A QAT bundle needs three things done correctly**: `model.pt` must be a `{"state_dict": ...}`
   mapping in the *native* schema (the released file stores the float `scale_dec`, while a model loaded
   with the integer control has different keys there); and `entropy_control_int.pt` embeds the SHA-256
   of the `model.pt` it was built from, so the hash must be rebound — which is only sound because the
   entropy components are frozen. Verify that no `entropy_model.*` tensor moved before rebinding.
10. **Module-name prefix matching with a trailing dot silently matches nothing.** `startswith(p + ".")`
    against a prefix that already ends in `"."` marked zero parameters trainable.
11. **Validate at the training resolution.** Full pool images are 2040x1356, well outside the resolution
    this model was trained at, and give 4 dB nonsense. Crop to 512 and quantise to uint8 first, because
    the codec emits [-1,1] while the reference metric compares against [0,1].
12. **Update a Lagrange multiplier on a fixed validation set, not per training batch.** Per-batch
    updates respond to content variation rather than to the constraint error.

## 6. Environment

Python 3.12.6, PyTorch 2.6.0+cu124, CUDA 12.4, RTX 3050 Laptop 4 GB (`sm_86`), scipy 1.18.1,
nvidia-ml-py 13.615.71, Pillow. No pandas, matplotlib, ortools, onnx or TensorRT, and none is needed:
every table and figure is stdlib plus Pillow, and the solver is scipy's HiGHS wrapper.

## 7. Suggested next action

Write the report and the deck. Every number they need exists and is committed; `RESOURCES.md` section 4
is already a claim-to-proof map. Figures to reuse: `results/decision/figures/` (6), `results/mixed/`
(BD-rate, 3 datasets), `results/metrics/` (latency, decode decomposition).
