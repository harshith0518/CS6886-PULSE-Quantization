Yes. Now that we have the **actual `pulse-mse` checkpoint audit**, I would finalize the project around the model we actually have rather than keeping the earlier generic task list.

The key change is that we now know PULSE-MSE is **highly asymmetric**: the encoder contains ~85.6% of the parameters, while the entire decoder is only ~5%. That should drive the experimental hierarchy.

---

# Final Project Structure — PULSE-MSE INT8 Study

## 0. Overall Research Question

### **Main question**

> **Can INT8 quantization make PULSE more efficient without significantly degrading rate–distortion performance, and can the sensitivity of different PULSE components be predicted across quality settings?**

The project therefore has **four connected layers**:

```text
                    PULSE-MSE
                       │
          ┌────────────┴────────────┐
          │                         │
     Understand                  Establish
     the model                   baseline
          │                         │
          ▼                         ▼
    Architecture                Rate-Distortion
    + Runtime                   Performance
          │                         │
          └────────────┬────────────┘
                       │
                       ▼
              INT8 Quantization
                       │
          ┌────────────┴────────────┐
          │                         │
      Where is it               How much does
      sensitive?                it actually help?
          │                         │
          ▼                         ▼
    Fragility Map              Deployment
    + Hessian                  + Energy
```

This gives the project a coherent story rather than six independent experiments.

---

# SECTION I — MODEL UNDERSTANDING & BASELINE

This section establishes **what PULSE-MSE actually is and how it behaves before we modify it.**

---

## M0 — PULSE-MSE Architecture & Model Audit

### Objective

Build an exact computational description of the released `pulse-mse` model before performing quantization experiments.

### Verified model facts

The checkpoint is:

> **PULSE-MSE, Base, MSE objective**

with:

- \(M=320\)
- \(z_{ch}=96\)
- hidden size = 144
- renderer hidden size = 48
- 2 conditional blocks
- entropy channels = 128
- CA DCB variant

These values come directly from the checkpoint configuration. :chatgpt-content-reference{index="0"}

### Model size

| Component | Parameters | Share |
|---|---:|---:|
| Encoder | 12,133,696 | **85.6%** |
| Entropy model | 1,337,224 | **9.4%** |
| Decoder | 708,504 | **5.0%** |
| **Total** | **14,179,432** | **100%** |

The model is 54.09 MB in FP32, with a theoretical 13.52 MB weight footprint if all weights are represented in INT8. :chatgpt-content-reference{index="1"}

### Architecture

```text
Input Image
     │
     ▼
Patch Embedding
3 → 512 channels
     │
     ▼
4 × Encoder DCB
512 channels
     │
     ▼
Latent Projection
512 → 320
     │
     ▼
Entropy / Hyperprior
     │
     ▼
Compressed Representation
     │
     ▼
Decoder
320 → 144
     │
     ▼
2 × Decoder DCB
     │
     ▼
AdaLN Renderer
     │
     ▼
PixelShuffle ×4
     │
     ▼
PixelShuffle ×4
     │
     ▼
Reconstructed Image
```

The actual model structure confirms the four encoder DCBs, two decoder DCBs and AdaLN renderer. :chatgpt-content-reference{index="2"}

### Why M0 matters

This tells us immediately that PULSE is **not a uniformly sized network**.

The project therefore should not ask simply:

> "Does INT8 work?"

It should ask:

> **"Which parts of this highly asymmetric architecture benefit from INT8, and which parts are sensitive to it?"**

### Deliverable

**Architecture + parameter/MAC/activation profile**

Eventually:

```text
Component | Params | MACs | Activation range | Runtime
```

---

# M1 — Baseline Rate–Distortion Reproduction

This is our **experimental control**.

### Objective

Reproduce the published PULSE-MSE rate–distortion behaviour before changing precision.

### Dataset

**Primary:**
- CLIC Professional validation — 41 images

**Secondary sanity check:**
- Kodak

### Quality settings

Run all **8 QPs / quality parameters**.

### Measurements

For every QP:

- bitrate / bpp
- PSNR
- reconstruction
- encode time
- decode time

### Main output

```text
PSNR
  │
  │             ●
  │          ●
  │       ●
  │    ●
  │ ●
  └────────────────────
          bpp
```

Overlay:

> **Published PULSE curve vs reproduced curve**

Because the paper provides curves rather than a complete numerical table, the comparison should use a digitized/reference curve and a clearly stated tolerance rather than inventing a fixed numerical criterion.

### Deliverable

**Baseline RD curve + reproduction table**

---

# SECTION II — WHERE DOES THE TIME GO?

This section answers the practical systems question before optimization.

---

# M2 — Runtime / Amdahl Analysis

### Objective

Determine where the actual PULSE runtime is spent.

Instead of assuming convolution is the bottleneck, measure it.

### Break the pipeline into:

```text
Encoding
│
├── Neural encoder
├── Probability / entropy model
├── CPU entropy coding
├── GPU → CPU copies
└── Python/framework overhead

Decoding
│
├── Entropy decoding
├── Probability model
├── Neural decoder
├── CPU → GPU copies
└── Python/framework overhead
```

### Measurements

Use CUDA events for GPU portions and synchronized wall-clock timing for end-to-end measurements.

Measure at multiple QPs.

### Main output

A stacked runtime chart:

```text
            Runtime
QP 1   ███████████████████
QP 4   █████████████████
QP 8   █████████████████████
```

with components such as:

```text
Neural network
Entropy
CPU coding
Memory transfer
Framework overhead
```

### Why this is critical

M0 tells us the encoder has ~86% of the **parameters**.

M2 tells us whether it also dominates **actual runtime**.

Those are not necessarily the same thing.

### Deliverable

**Runtime/Amdahl breakdown**

---

# SECTION III — INT8 SENSITIVITY

This is where the project becomes a research study rather than just a deployment benchmark.

---

# M4 — QP-Dependent INT8 Fragility Map

I would actually place this **before the full deployment experiment**.

### Research question

> **Does INT8 quantization affect different PULSE components differently at different quality settings?**

### Core idea

Quantize **one architectural component at a time**.

Keep everything else FP32.

For each component:

```text
FP32 component
       ↓
INT8 component
       ↓
same latent / same symbols
       ↓
reconstruction
       ↓
ΔPSNR
```

This is important because it isolates **decoder/neural-transform quantization damage** from changes in entropy coding.

---

## Groups to test

### Encoder

```text
E0 — Patch embedding
E1 — Encoder DCB 0
E2 — Encoder DCB 1
E3 — Encoder DCB 2
E4 — Encoder DCB 3
E5 — Latent projection
```

### Entropy model

```text
H0 — Hyper encoder
H1 — Hyper decoder
H2 — EM Step 1
H3 — EM Step 2
H4 — Scale decoder
```

### Decoder

```text
D0 — Initial projection
D1 — Decoder DCB 0
D2 — Decoder DCB 1
D3 — Latent upsampling
D4 — Condition projection
D5 — Renderer MLP
D6 — AdaLN
D7 — Output projection
```

These groups are based on the actual architecture rather than arbitrary layer numbering. The checkpoint confirms the two decoder DCBs and renderer components. :chatgpt-content-reference{index="3"}

### Output

A heatmap:

| Component | QP1 | QP2 | ... | QP8 |
|---|---:|---:|---:|---:|
| Encoder DCB0 | Δ | Δ | ... | Δ |
| Encoder DCB1 | Δ | Δ | ... | Δ |
| Decoder DCB0 | Δ | Δ | ... | Δ |
| Renderer | Δ | Δ | ... | Δ |

where each value is:

> **ΔPSNR relative to FP32**

### This becomes one of the project's main figures.

---

# E1 — Hessian-Based Sensitivity Prediction

This should be the **main research extension**, not just another random extra.

### Research question

> **Can second-order sensitivity predict which PULSE components will be most vulnerable to INT8 quantization?**

Instead of experimentally quantizing every component first and then stopping there, calculate an approximate sensitivity score.

### Method

Use a Hutchinson-style approximation of the Hessian trace:

\[
S_i \approx \mathrm{Tr}(H_i)
\]

or an equivalent normalized second-order sensitivity measure.

If double-backpropagation becomes impractical, use an empirical-Fisher / squared-gradient proxy as the fallback.

### Compare

```text
Predicted sensitivity
        VS
Actual ΔPSNR
```

### Main figure

Scatter plot:

```text
Actual INT8 damage
│
│             ●
│        ●
│     ●
│  ●
│________________________
       Hessian score
```

and calculate:

> **Spearman rank correlation**

### Why this is strong

M4 says:

> "This layer is fragile."

E1 asks:

> **"Could we have predicted that before performing the expensive quantization experiment?"**

That is a much more interesting research contribution.

---

# SECTION IV — ACTUAL INT8 DEPLOYMENT

Once we understand sensitivity, we move to actual deployment.

---

# M3 — Receiver FP32 → FP16 → INT8

### Objective

Determine whether the lightweight receiver actually benefits from lower precision.

### Configurations

```text
FP32
  ↓
FP16
  ↓
INT8
```

### INT8 calibration experiment

Compare:

### A. Shared calibration

One calibration scale/range for all QPs.

versus

### B. QP-specific calibration

Separate calibration for each quality parameter.

This directly tests the hypothesis:

> **Quantization behaviour may depend on the operating quality setting.**

### Metrics

For each QP:

- PSNR
- bpp
- latency
- memory
- BD-rate
- quality degradation

### Main output

```text
            Shared INT8    QP-specific INT8
QP1             ...              ...
QP2             ...              ...
...
QP8             ...              ...
```

and:

> **FP32 vs FP16 vs INT8 rate–distortion curves**

### Important caveat

TensorRT should be treated as the **deployment backend if feasible**, not as a project assumption.

If some operators remain FP16 or TensorRT support is problematic, document that rather than pretending the model is fully INT8.

---

# SECTION V — HEAVY ENCODER OPTIMIZATION

This deserves its own section because the architecture tells us it may be the biggest practical opportunity.

---

# M5 — Encoder INT8 Feasibility

### Motivation

The encoder contains:

> **12.13M / 14.18M parameters ≈ 85.6% of the model.** :chatgpt-content-reference{index="4"}

Each of the four encoder DCBs is approximately 2.89M parameters, with the large 512→2048→512 1×1 convolutions dominating the parameter count. :chatgpt-content-reference{index="5"}

Therefore:

> **The encoder is the primary candidate for meaningful INT8 acceleration.**

### Compare

```text
Encoder FP16
      VS
Encoder INT8
```

### Metrics

- latency
- throughput
- GPU memory
- PSNR
- bpp
- BD-rate

### Target

Earlier we considered:

> ≥1.3× speedup

I would **not make 1.3× a pass/fail criterion**.

Instead:

> **Measure the achieved speedup and explain it using the layer-level runtime/roofline profile.**

A 1.15× result with a convincing explanation is still a valid result.

### Special 4 GB VRAM constraint

Your RTX 3050 has only **4 GB VRAM**, so 1080p experiments may require careful memory management or tiling.

This should be documented as an experimental constraint, not hidden.

---

# SECTION VI — ENERGY / PRACTICAL EFFICIENCY

---

# M6 — Energy per Image

This converts the optimization from:

> "It runs faster"

into:

> **"It is more computationally efficient."**

### Compare

```text
FP32
FP16
INT8
```

for the feasible configurations.

### Measure

- power
- runtime
- energy/image
- PSNR
- bpp

If direct GPU energy measurement isn't available, sample power over time and integrate it.

### Main result

A Pareto plot:

```text
Energy / image
│
│  FP32 ●
│
│       ● FP16
│
│          ● INT8
│________________________
             PSNR
```

### Deliverable

**Quality–energy–latency trade-off**

---

# SECTION VII — OPTIONAL RECOVERY / ROBUSTNESS EXPERIMENTS

These are useful, but they should **not threaten the core project**.

---

# E2 — QAT Recovery

Only do this if:

> INT8 produces significant quality degradation.

Pipeline:

```text
PTQ
 ↓
measure degradation
 ↓
if unacceptable
 ↓
short QAT fine-tuning
 ↓
re-evaluate
```

Do **not** plan full PULSE retraining on the RTX 3050.

This is a recovery experiment, not a core deliverable.

---

# E3 — Cross-Device Decode Compatibility

If time permits:

```text
Encode
   ↓
bitstream
   ↓
Device A
Device B
Device C
   ↓
decode
```

The important criterion is:

> **Successful decoding + comparable rate/quality**

not necessarily bit-for-bit identical reconstructed images across different hardware backends.

Different INT8 implementations can produce small numerical differences.

---

# E4 — PULSE-S-MSE Comparison

Only after the main `pulse-mse` study is complete.

Why?

Because the S model gives us a useful question:

> Does the INT8 behaviour change when the receiver is made even smaller?

This could provide a compact comparison:

```text
                 PULSE-MSE     PULSE-S-MSE
Params              ...             ...
Decoder             ...             ...
INT8 speedup        ...             ...
ΔPSNR               ...             ...
```

But **do not let this become another full eight-QP project** unless you have substantial time remaining.

---

# SECTION VIII — FUTURE WORK / EXPLICITLY OUT OF SCOPE

These should be shown separately in your slides so that reviewers know we considered them but deliberately excluded them.

---

## F1 — 2:4 Structured Sparsity

Potential future direction:

```text
INT8
 +
2:4 sparsity
```

Particularly interesting for the large encoder 1×1 convolutions.

But it requires substantially more work and hardware-specific validation.

**Not core.**

---

## F2 — ILP-Based Mixed Precision

Potential future formulation:

\[
\min_{\text{precision assignment}}
\quad
\text{Latency}
\]

subject to:

\[
\Delta PSNR < \epsilon
\]

This could automatically choose:

```text
Layer A → INT8
Layer B → INT8
Layer C → FP16
Layer D → FP16
```

But it depends on M3/M4/M5 data first.

**Future work.**

---

## F3 — Other PULSE Models

The repository contains:

```text
pulse-s-mse
pulse-mse
pulse-perceptual
pulse-l-perceptual
```

but our primary study is deliberately:

> **pulse-mse**

The other models are comparison/future work rather than four parallel projects.

This keeps the project manageable.

---

## F4 — T4 / Additional Hardware Replication

Useful for generality:

```text
RTX 3050
     VS
T4
     VS
other accelerators
```

but unnecessary for the core research question.

---

# FINAL TASK HIERARCHY

This is the version I would actually put into your proposal/presentation.

## **CORE — Must Complete**

### **WP1 — PULSE-MSE Characterization**

**M0 + M1**

- Architecture audit
- Parameter distribution
- Model execution structure
- Baseline RD reproduction
- CLIC Professional + Kodak
- All 8 QPs

**Deliverable:**

> Baseline architecture + rate–distortion characterization

---

### **WP2 — Runtime Characterization**

**M2**

- Encoder/decoder timing
- Entropy coding
- Data transfers
- Framework overhead
- QP-dependent runtime

**Deliverable:**

> Amdahl/runtime breakdown identifying the actual bottlenecks

---

### **WP3 — INT8 Sensitivity Analysis**

**M4**

- Component-wise quantization
- All 8 QPs
- ΔPSNR measurement
- Fragility heatmap

**Deliverable:**

> QP-dependent INT8 fragility map

---

### **WP4 — Sensitivity Prediction**

**E1**

- Hessian/Hutchinson sensitivity
- Compare predicted vs observed degradation
- Spearman correlation

**Deliverable:**

> Predictive relationship between second-order sensitivity and INT8 damage

**This is the main research novelty.**

---

### **WP5 — Practical INT8 Deployment**

**M3 + M5**

Receiver:

```text
FP32 → FP16 → INT8
```

Encoder:

```text
FP16 → INT8
```

Measure:

- latency
- PSNR
- bpp
- BD-rate
- memory
- speedup

**Deliverable:**

> Practical precision-performance trade-off

---

### **WP6 — Energy Efficiency**

**M6**

- Power
- Runtime
- Joules/image
- Quality

**Deliverable:**

> Energy–quality–latency Pareto analysis

---

# OPTIONAL EXTENSIONS

Only after WP1–WP6 are secure:

### E2 — QAT Recovery
Recover quality after INT8.

### E3 — Cross-device decoding
Check practical bitstream portability.

### E4 — PULSE-S-MSE
Small-model comparison.

---

# FUTURE WORK

Explicitly defer:

- 2:4 structured sparsity
- ILP mixed precision
- Full mixed-precision optimization
- T4 replication
- Perceptual models
- Large-scale QAT/retraining

---

# THE FINAL EXPERIMENTAL FLOW

This is probably the **most useful slide** to include:

```text
                 PULSE-MSE
                     │
                     ▼
        ┌─────────────────────────┐
        │ WP1: Model + Baseline   │
        │ M0 + M1                  │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │ WP2: Runtime Analysis   │
        │ M2                      │
        └────────────┬────────────┘
                     │
              Where is the
              computation?
                     │
                     ▼
        ┌─────────────────────────┐
        │ WP3: INT8 Fragility     │
        │ M4                      │
        └────────────┬────────────┘
                     │
              Which parts are
              sensitive?
                     │
                     ▼
        ┌─────────────────────────┐
        │ WP4: Hessian Prediction │
        │ E1                      │
        └────────────┬────────────┘
                     │
             Can sensitivity
             be predicted?
                     │
                     ▼
        ┌─────────────────────────┐
        │ WP5: INT8 Deployment    │
        │ M3 + M5                 │
        └────────────┬────────────┘
                     │
          Does INT8 actually help?
                     │
                     ▼
        ┌─────────────────────────┐
        │ WP6: Energy Efficiency  │
        │ M6                      │
        └─────────────────────────┘
```

---

# THE STORY YOUR PRESENTATION SHOULD TELL

I would **not** present this as:

> "We are going to try INT8 on PULSE."

That's too basic.

Instead:

### **1. PULSE is asymmetric**

85.6% of parameters are in the encoder, while only 5% are in the decoder. :chatgpt-content-reference{index="6"}

↓

### **2. Therefore, we first need to understand the runtime**

Parameter count does not necessarily equal runtime.

↓

### **3. INT8 sensitivity may not be uniform**

Different architectural components may respond differently to quantization.

↓

### **4. Quality setting may matter**

The eight QPs may not have identical INT8 sensitivity.

↓

### **5. We map that sensitivity**

Component × QP → ΔPSNR.

↓

### **6. We ask whether it can be predicted**

Hessian sensitivity → actual quantization damage.

↓

### **7. We deploy the useful configurations**

FP32 / FP16 / INT8.

↓

### **8. Finally, we quantify the practical benefit**

Latency + bitrate + PSNR + memory + energy.

That gives you a clean research narrative:

> **Characterize → Localize → Predict → Quantize → Deploy → Evaluate**

---

# THE 6 MAIN DELIVERABLES

For your final proposal slide, I would reduce everything to these six:

| Deliverable | Question | Main Output |
|---|---|---|
| **D1. PULSE Characterization** | What exactly are we optimizing? | Architecture + parameter/MAC profile |
| **D2. Baseline Reproduction** | Can we reproduce PULSE? | RD curves |
| **D3. Runtime Analysis** | Where does time go? | Amdahl breakdown |
| **D4. INT8 Fragility Map** | Where does INT8 hurt? | Component × QP heatmap |
| **D5. Sensitivity Prediction** | Can we predict INT8 damage? | Hessian vs ΔPSNR + Spearman |
| **D6. Deployment & Efficiency** | Does INT8 actually help? | Latency/BD-rate/energy Pareto |

And then underneath:

> **Optional:** QAT recovery, PULSE-S comparison, cross-device decoding  
> **Future:** 2:4 sparsity, ILP mixed precision, additional hardware, perceptual models

---

## One-sentence project definition

> **This project characterizes the computational and quantization behaviour of the PULSE-MSE neural image codec, investigates whether INT8 sensitivity varies across architectural components and quality settings, evaluates whether Hessian-based sensitivity can predict quantization damage, and determines the resulting latency, rate–distortion, memory, and energy trade-offs on a consumer NVIDIA GPU.**

That is the version I would use as the **central project statement** in the slides.

I can also turn this exact structure into **slide-by-slide content with titles, bullet limits, figures to show, and a final Gantt aligned to these deliverables**.

Would you like the next version organized as a slide-by-slide deck outline or as a compact project-plan table?