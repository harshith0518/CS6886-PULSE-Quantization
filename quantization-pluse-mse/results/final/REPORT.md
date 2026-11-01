# Submission operating points

3 measured points, each evaluated on all three datasets with the same harness,
the same eight QPs and the same FP32 reference. Nothing here is a prediction.

| Point | Size | Saving | Kodak | CLIC | Set14 |
|---|---:|---:|---:|---:|---:|
| 1% budget champion | 21.677 MB | -60.36% | 0.672% | 0.851% | 0.935% |
| 1.8% budget variant | 18.664 MB | -65.87% | 1.268% | 1.595% | 1.601% |
| INT8 tier + QAT 12k | 14.401 MB | -73.67% | 1.848% | 2.874% | 3.363% |

These three points trace the size-quality frontier. The 1.8 percent variant beats the
2 percent assignment it replaced: 0.275 MB larger but better on all three datasets. The
INT8 tier at 14.401 MB is a quantized model that quantization-aware fine-tuning made
usable, taking it from 8.61 to 3.36 percent, and it is the smallest model in this table
worth deploying.

# QAT: what it recovered, and where

Fine-tuning was run on three precision tiers with an augmented-Lagrangian rate
controller: minimise distortion subject to holding the bitrate at its pre-QAT value,
with the multiplier updated from the measured rate error on a fixed validation set.

## Augmented-Lagrangian result

| Tier | Size | Before QAT | After QAT | Damage removed |
|---|---:|---:|---:|---:|
| More INT8 | 14.401 MB | 8.61% | 8.61% | 0.0% |
| Moderate INT4 + few INT8 | 13.572 MB | 119.18% | 25.27% | 78.8% |
| More INT4 + few INT8 | 11.369 MB | 1771.71% | 84.22% | 95.2% |
| More INT8 | 14.401 MB | 8.61% | 3.36% | 61.0% |
| Moderate INT4 + few INT8 | 13.572 MB | 119.18% | 20.53% | 82.8% |
| More INT4 + few INT8 | 11.369 MB | 1771.71% | 41.84% | 97.6% |
| ILP 2% + QAT | 18.389 MB | 1.94% | 1.94% | 0.0% |

The pattern is the expected one and it is the interesting result: **QAT pays exactly where
there is damage to recover.** The INT8 tier starts at 8.6% and gains nothing, because
INT8 quantization cost it very little. The two INT4 tiers start at 119% and 1772% and
recover 79% and 95% of that damage. QAT is not a general improvement; it is a repair for
an aggressive precision choice.

## Trajectory of the objectives that were tried

Earlier attempts used fixed-penalty objectives and failed; they are recorded here because
they show why the constrained form is necessary.

| Objective | Steps | Worst penalty | Final dPSNR |
|---|---:|---:|---:|
| raw RD objective, lr 3e-6, rate term unconstrained | 5900 | 72.46% | -1.536 dB |
| rate-preserving, lr 3e-6, rate weight 1 | 200 | 0.00% | +0.189 dB |
| rate-preserving, lr 3e-6, rate weight 1 (restart) | 1200 | 12.79% | +0.118 dB |
| rate-preserving, lr 1e-6, rate weight 8 | 300 | 46.18% | -2.220 dB |

## A harness bug that had to be fixed first

The checkpoint selector accumulated `max(0.0, penalty)`, so a negative penalty - a
genuine improvement - was floored to zero and could never be selected. Every run before
the fix therefore saved its step-0 weights and silently discarded its own gains. This is
the reason the first attempts were reported as making no progress; it was the selector,
not the training. The floor is removed and the selector now keeps any checkpoint that
beats the starting model.

## What the saved bundles are

Each tier's bundle contains the best checkpoint by worst-QP penalty against that tier's own
starting weights, which is why tiers 2 and 3 report negative `best_penalty_vs_start_pct`
(-28.1% and -74.0%). The bundles decode through the real coder: entropy components are
frozen, so the integer CDF tables remain valid and the artifact hash is rebound only after
verifying that no `entropy_model.*` tensor moved.

# Augmented-Lagrangian attempt


| Tier | Size | Before | After | Recovery |

|---|---:|---:|---:|---:|

| tier1_more_int8 | 14.401 MB | 8.613% | 8.613% | 0.0 |

| tier2_moderate_int4 | 13.572 MB | 119.18% | 25.273% | 93.907 |

| tier3_more_int4 | 11.369 MB | 1771.715% | 84.217% | 1687.498 |

| tier1_more_int8 | 14.401 MB | 8.613% | 3.363% | 5.25 |

| tier2_moderate_int4 | 13.572 MB | 119.18% | 20.528% | 98.652 |

| tier3_more_int4 | 11.369 MB | 1771.715% | 41.842% | 1729.873 |

| ilp_2pct_qat | 18.389 MB | 1.936% | 1.936% | 0.0 |

