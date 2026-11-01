# Which Bits Can We Drop?

**Component-wise quantization, mixed-precision selection and quantization-aware fine-tuning for the
PULSE learned image codec.** CS6886 final project.

**Report (2 pages):** [`submission/report.pdf`](submission/report.pdf) ·
**Slides:** [`submission/PULSE-quantization-deck.pdf`](submission/PULSE-quantization-deck.pdf) ·
**Full runbook:** [`RUNBOOK.md`](RUNBOOK.md)

---

## What this repository is

This is a **fork of the PULSE reference implementation** (from
[`microsoft/GenCodec`](https://github.com/microsoft/GenCodec), paper
[arXiv:2609.18602](https://arxiv.org/abs/2609.18602), weights on
[HuggingFace](https://huggingface.co/zhaoyangjia/PULSE)) with a quantization study built on top of it.

The upstream code in `codec/`, `deployment/`, `configs/` and the training scripts is **unchanged and
not ours**. Everything under `quantization-pluse-mse/` and `submission/` is our work.

## The problem

PULSE ships **one weight file (54.688 MB, 14.18M parameters)** that must serve **eight quality
settings**. The standard way to shrink it is to pick one precision for the whole model — but the model
is wildly unbalanced:

| Group | Parameters | Share | Why it matters |
|---|---|---|---|
| Encoder E0–E5 | 12.13M | 85.6% | four blocks alone are 11.6 MB |
| Entropy H0–H3 | 1.34M | 9.4% | dominates decode time (73%) |
| Decoder D0–D8 | 708k | 5.0% | small, but holds a 4.7k block that loses 8% at 8 bits |

**Our goal:** find the smallest per-component precision assignment that stays inside a quality budget,
then test whether fine-tuning can recover what it costs.

## What we did

1. **Measured sensitivity.** Quantized **one component at a time** (everything else FP32), across
   **7 formats × 19 components × 8 quality settings × 2 datasets** = 69,233 measurements, with an FP32
   gate that reproduces the reference bitstream exactly at every setting.
2. **Verified the measurements are additive** (0.96–1.05 Kodak, 1.02–1.11 CLIC), so single-component
   costs can be summed and used as optimizer coefficients.
3. **Solved the allocation as a mixed-integer program.** One precision per component, *shared across
   all eight quality settings* because one file must serve all eight, with one quality constraint per
   setting. Solved with HiGHS.
4. **Fine-tuned the result** with straight-through quantized convolutions and an augmented-Lagrangian
   rate constraint, holding bitrate fixed while pushing distortion down. Entropy weights stay frozen
   so the integer CDF tables in the bundle remain valid.

## Results

Worst-quality-setting rate-equivalent penalty against FP32. Lower is better.

| Configuration | Size | Saving | Kodak | CLIC | Set14 |
|---|---|---|---|---|---|
| **ILP mixed, 1% budget** | **21.677 MB** | **−60.4%** | **0.672%** | **0.851%** | **0.935%** |
| ILP mixed, 1.8% budget | 18.664 MB | −65.9% | 1.268% | 1.595% | 1.601% |
| INT8 tier + QAT (12k steps) | 14.401 MB | −73.7% | 1.848% | 2.874% | 3.363% |
| uniform INT8 (the naive baseline) | 13.904 MB | −74.6% | 26.184% | 56.701% | 19.374% |

BD-rate for the champion: **+0.48 / +0.55 / +0.49%** (Kodak / CLIC / Set14). Set14 was never used for
fitting, so it is a genuine transfer check.

**The headline:** uniform INT8 produces a *smaller* file than our champion and is **39× worse on Kodak
and 67× worse on CLIC**. Granularity, not precision, is what matters.

## What worked, and what did not

**Worked**

- Per-component sensitivity plus an ILP: 60% of the bytes gone for under 1% quality cost.
- The mixed-versus-uniform gap (39–67×), the single strongest result in the project.
- **QAT as a repair mechanism.** It removed **61%**, **83%** and **98%** of the damage on three
  hand-chosen assignments, taking a 14.401 MB INT8 model from 8.61% to 3.36%.
- Group-32 scaling for 4-bit: cut INT4 weight error 40% and damage 5–22× versus per-channel.

**Did not work**

- **No latency win.** Quantization accelerates convolutions, which are 9% of a decode; 73% is entropy
  coding. Measured +4.79% latency (95% CI +1.27 to +9.80), energy change −0.39% with an interval
  spanning zero. We report this rather than hide it.
- **4-bit alone.** Even with group scaling, INT4 costs 26–157 penalty points per MB saved versus
  0.14–0.46 for INT8 — about two orders of magnitude worse per byte.
- **QAT on the optimizer's own assignment: exactly 0.0% gain.** Fine-tuning repairs what an assignment
  leaves on the table, and an optimal assignment leaves nothing. We consider this our most useful
  finding, not a failure.
- **Reproducing the full pretraining corpus** (Open Images + SAM-1B is terabytes). We fine-tuned on 100
  DIV2K images instead: 2.2% of the pretraining optimizer steps and 0.2% of its sample budget.

## Tools and resources

| Purpose | Tool |
|---|---|
| Model, quantization simulation, fine-tuning | PyTorch 2.6.0 + CUDA 12.4, one RTX 3050 Laptop (4 GB) |
| Entropy coding | PULSE native extension (`MLCodec_extensions_cpp`) |
| Mixed-integer optimization | HiGHS via `scipy.optimize.milp` |
| Energy and power | NVIDIA NVML total-energy counters |
| Rate-distortion comparison | Bjøntegaard delta-rate with PCHIP interpolation |
| Documents | LaTeX (Tectonic), `python-docx`, `python-pptx`, LibreOffice for PDF export |
| Data | Kodak, CLIC Professional, Set14, DIV2K (fine-tuning pool) |

## Quick start

```bash
git clone https://github.com/harshith0518/CS6886-PULSE-Quantization.git
cd CS6886-PULSE-Quantization
python -m venv venv && .\venv\Scripts\activate      # source venv/bin/activate on POSIX
pip install -r requirements.txt
pip install -e . --no-build-isolation                # builds the entropy extension
```

Download the released checkpoint from [HuggingFace](https://huggingface.co/zhaoyangjia/PULSE) into
`checkpoints/pulse-mse/` (three files: `config.json`, `model.pt`, `entropy_control_int.pt` — the third
is mandatory, the loader refuses a bundle without it).

Then reproduce the headline number:

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\eval_assignment.py `
  --model checkpoints/pulse-mse `
  --configs quantization-pluse-mse\results\ilp\assignment_ilp.json `
  --dataset kodak --output results\check
```

`RUNBOOK.md` has every command, expected runtime, and troubleshooting step.

## Repository map

| Path | What is in it |
|---|---|
| `codec/`, `deployment/`, `configs/`, `train_*.py` | **Upstream PULSE** — not our work |
| `quantization-pluse-mse/` | Our study: sweep, analysis, ILP, QAT, all result tables |
| `quantization-pluse-mse/STATUS.md` | Project checkpoint — every number, every trap we hit |
| `quantization-pluse-mse/RESOURCES.md` | Claim-to-proof map with verified citations |
| `submission/` | Report (`.tex` + `.pdf`), slides, and a longer supplementary write-up |
| `datasets/` | Evaluation sets and the fine-tuning pool (images are not tracked) |
| `docs/archive/` | Earlier working notes, kept for provenance |

## Conclusion

Mixed precision buys 60% of this model for under 1% quality cost, and it beats uniform quantization by
39× at comparable size. Fine-tuning is a *substitute* for good allocation rather than a supplement: it
recovered 61–98% of the damage on hand-chosen precision, and nothing on the assignment the solver had
already optimized. Both halves of that statement are useful, and the second is only visible because we
measured the control case.

Quantization did not make the codec faster, and §4.3 of the report explains why in arithmetic rather
than assertion.

---

## Upstream PULSE (original README)

<h2 align="center">PULSE: Unlocking Practical Image Compression on Single-Thread CPU</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2609.18602"><img src="https://img.shields.io/badge/arXiv-2609.18602-b31b1b.svg" alt="arXiv: 2609.18602"></a>
  <a href="https://github.com/microsoft/GenCodec/tree/main/PULSE"><img src="https://img.shields.io/badge/Code-GitHub-blue.svg" alt="GitHub"></a>
  <a href="https://huggingface.co/zhaoyangjia/PULSE"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue.svg" alt="Hugging Face"></a>
</p>

---

## 📖 Introduction

**PULSE** is an asymmetric, variable-rate neural image codec that combines
**ultra-low-complexity decoding**, **bit-exact entropy coding**, and
**MSE or perceptual optimization** for practical compression on
resource-constrained hardware.

[Paper](https://arxiv.org/abs/2609.18602) |
[Training](docs/training.md) |
[Deployment](docs/deployment.md)

![PULSE compression performance and cross-platform deployment overview](assets/overview.png)

## ✨ Highlight

PULSE pairs low-resolution feature processing with an **AdaLN-modulated pixel
renderer** and a **two-step entropy model**. A content-adaptive **Meta Prior**
compensates for the reduced capacity of the linear CDF decoder, especially in
low-content-complexity regions. It features:

- ⚡ **Single-thread CPU decoding** — The base receiver requires just
  **5.2 kMAC/pixel**. The paper reports **126 ms** decoding for a **1080p** image
  on one AMD EPYC 9V84 CPU thread.
- 🌐 **Cross-platform entropy coding** — **Integer Linear CDF prediction + CPU
  rANS** keep entropy coding bit-exact while neural transforms can run on
  accelerators; floating-point reconstructed pixels need not be bit-identical.
- 🎨 **Flexible rate and quality** — **Eight quality levels per model**, with
  MSE models for S/base and perceptual models for base/L, including
  face/text-aware fine-tuning.
- 🗜️ **Train, compress, and deploy.** This release includes training, Meta Prior
post-training, integer Linear CDF export, y CDF calibration, and independent `.pulse` compression
and decompression. Optimized AMD CPU and H100 benchmarks include entropy coding;
the mobile path provides whole-image QNN/ARM compression and decompression.

<p align="center">
  <img src="assets/architecture.png" width="95%" alt="PULSE architecture">
</p>

## 🧩 Models

Each model supports eight QPs (`0`–`7`), from lower to higher rate/quality.

| Model | Receiver complexity | Weights | Training config |
|---|---:|---|---|
| PULSE-S (MSE) | ~2.7 kMAC/pixel | `pulse-s-mse` | [Config](configs/mse/s.yaml) |
| PULSE (MSE) | ~5.2 kMAC/pixel | `pulse-mse` | [Config](configs/mse/base.yaml) |
| PULSE (Perceptual) | ~5.2 kMAC/pixel | `pulse-perceptual` | [Config](configs/perceptual/base.yaml) |
| PULSE-L (Perceptual) | ~20 kMAC/pixel | `pulse-l-perceptual` | [Config](configs/perceptual/l.yaml) |

Pretrained bundles and training recipes are provided for these four models.

## 🛠️ Installation

Use Python 3.12 and a C++17 compiler for the pinned environment below.
Install PyTorch 2.5.0 / torchvision 0.20.0 wheels appropriate for your CPU/CUDA
environment, then install PULSE:

```bash
git clone https://github.com/microsoft/GenCodec.git
cd GenCodec/PULSE
pip install -r requirements.txt
pip install -e .
```

This builds the CPU rANS and integer-CDF extension; a CUDA compiler is not
required. Tested with Python 3.12, PyTorch 2.5.0 and torchvision 0.20.0 on Linux.
`requirements.txt` pins the direct dependencies for training, evaluation,
inference and model downloads. Optional ONNX/OpenVINO dependencies are in
`requirements-deploy.txt`. These files are not transitive lockfiles.
For a minimal inference-only installation on Python 3.10+, use
`pip install -e ".[download]"` instead.
ONNX export explicitly uses the TorchScript exporter rather than relying on
version-dependent defaults.

Non-editable installs also include recipes and `pulse-compress`,
`pulse-decompress`, `pulse-train-mse`, `pulse-train-perceptual` and the other
`pulse-*` command-line entry points.

## 📥 Download checkpoints

```bash
python download_models.py --models pulse-mse pulse-perceptual
```

Bundles contain `config.json`, `model.pt`, and `entropy_control_int.pt`.
Keep these files together. Downloads are checked against SHA256 hashes.
Use `--revision <commit>` to pin a Hugging Face model revision.

## 🗜️ Compress and decompress

```bash
python compress.py --model checkpoints/pulse-mse \
  --input image.png --output outputs/image.pulse --qp 3 --device cpu --threads 1

python decompress.py --model checkpoints/pulse-mse \
  --input outputs/image.pulse --output outputs/reconstruction.png --device cpu --threads 1
```

Both commands also accept input/output directories. Decompression does not
require the original image. Use the **same model bundle** for both operations;
`.pulse` files do not embed the neural network. `--device cuda:0` enables CUDA.

New files include a 32-byte envelope with a model identifier, payload length
and checksum. H100 and mobile streams record three lane counts and the skip
partitioning profile for independent entropy sections. See the
[deployment guide](docs/deployment.md) for decoder compatibility and builds.
Wrong bundles and damaged files are rejected before entropy
decoding. Unchecked native files require explicit `--allow-legacy` on decompression;
`compress.py --legacy` writes that unchecked format for compatibility only.

The integer entropy-control path is deterministic across supported devices.
Floating-point reconstructed pixels are not promised to be bit-identical.

Encoding defaults are calibrated part-specific y CDFs, skip-index cutoff **2**,
and **Meta Prior Index Merge** on every platform. Older bundles without a
calibrated y table (including the initial S bundle) must first be updated with
`python -m codec.training.y_cdf`; Gaussian coding is not silently substituted.
Index Merge uses fitted probabilities when present, otherwise its deterministic
P=0.5 initialization. Decoders follow the file header for legacy syntax.

## 📊 Evaluate and benchmark

Build the optimized platform extensions using the guides below before benchmarking.

```bash
python evaluate.py --model checkpoints/pulse-mse --input data/kodak \
  --qps 0 1 2 3 4 5 6 7 --device cuda:0 --output outputs/kodak.json

python -m deployment.cpu_amd_epyc_9v84_single_thread.benchmark --model checkpoints/pulse-mse \
  --height 1088 --width 1920 --output outputs/cpu_amd_epyc_9v84_single_thread.json

python -m deployment.gpu_nvidia_h100.benchmark --model checkpoints/pulse-mse \
  --device cuda:0 --height 1088 --width 1920 --output outputs/gpu_nvidia_h100.json
```

Rates include actual file headers. Timings exclude disk IO, model loading,
graph conversion and warmup. Benchmark results depend on hardware, image content
and runtime versions.
Platform guides: [AMD CPU](deployment/cpu_amd_epyc_9v84_single_thread/README.md),
[H100 GPU](deployment/gpu_nvidia_h100/README.md), and
[REDMI NPU](deployment/npu_qualcomm_snapdragon_8_elite/README.md).
See [Deployment](docs/deployment.md) for the shared timing scope.

Perceptual metric dependencies are included in `requirements.txt`; add
`--perceptual-metrics --dists --fid` to `evaluate.py`.
FID uses 256-pixel patches by default (`--fid-patch 64` selects 64-pixel patches).
Use `evaluate.py --legacy` only when evaluating the native header-rate
convention; the default includes the additional 32-byte integrity envelope.

## 🚀 Training

```bash
python train_mse.py --size base --train-data data/train.txt \
  --hr-data data/train_hr.txt --output runs/pulse-mse

python postprocess.py --model runs/pulse-mse/model --data data/train_hr.txt \
  --output checkpoints/my-pulse-mse
```

MSE supports `--size s` or `base`; perceptual supports `--size base` or `l`.
The perceptual/ROI recipe, teacher preparation,
multi-GPU training and resume commands are in [Training](docs/training.md).
`postprocess.py` also fits the part-specific y CDFs for all four models
and enables Index Merge fitting by default. It creates a new, validated checkpoint at
`--output`, including all three matching model files and `SHA256SUMS`;
the training checkpoint is not overwritten.

## 📖 Citation

```bibtex
@article{jia2026pulse,
    title     = {{PULSE}: Unlocking Practical Image Compression on Single-Thread {CPU}},
    author    = {Jia, Zhaoyang and Zhang, Tianyu and Zheng, Zihan and Xie, Wenxuan and Li, Jiahao and Li, Bin and Li, Houqiang and Lu, Yan},
    journal   = {arXiv preprint arXiv:2609.18602},
    year      = {2026},
    url       = {https://arxiv.org/abs/2609.18602}
}
```

## 🙏 Acknowledgments and license

The codec and entropy kernels build on [DCVC](https://github.com/microsoft/DCVC).
We also thank the authors of LPIPS, DINOv2, FaceNet and CRNN.
See [LICENSE](LICENSE); original Microsoft copyright and MIT license notices are retained.
External dependencies and resources remain under their respective licenses.
Third-party teacher weights and vendor SDK/runtime binaries are not included.

