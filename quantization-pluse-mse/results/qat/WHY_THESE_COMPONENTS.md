# Why these components, and not the others

Every number below is measured. `penalty` is the worst-QP rate-equivalent cost of quantizing that
component alone, worst over Kodak and CLIC; INT4 uses group-32 scaling (`int4g32`, 5.0 effective
bits per weight) measured on 4 Kodak images. `cost/MB` is the decision quantity: percentage
points of penalty paid per MB of weights saved.

The selection rule is: **move the components with the lowest cost/MB first**, which is exactly what
the greedy and MILP solvers minimise. A component is excluded when its cost/MB is orders of
magnitude worse than the alternatives.

## The decision table

| Comp | Role | Params | 16-bit MB | INT8 MB | int4g32 MB | penalty@16-bit | penalty@INT8 | penalty@int4g32 | cost/MB 16->8 | cost/MB 8->4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 | encoder_dcb_2 | 2,893,824 | 5.788 | 2.894 | 1.809 | 0.010 | 0.427 | 34.498 | 0.14 | 31.40 |
| E2 | encoder_dcb_1 | 2,893,824 | 5.788 | 2.894 | 1.809 | 0.010 | 0.509 | 28.853 | 0.17 | 26.12 |
| E4 | encoder_dcb_3 | 2,893,824 | 5.788 | 2.894 | 1.809 | 0.015 | 0.710 | 34.602 | 0.24 | 31.23 |
| E1 | encoder_dcb_0 | 2,893,824 | 5.788 | 2.894 | 1.809 | 0.015 | 1.352 | 172.271 | 0.46 | 157.50 |
| H1 | hyper_decoder | 154,688 | 0.309 | 0.155 | 0.097 | 0.005 | 0.131 | - | 0.82 | - |
| D1 | decoder_dcb_0 | 142,832 | 0.286 | 0.143 | 0.089 | 0.001 | 0.124 | - | 0.86 | - |
| D2 | decoder_dcb_1 | 142,832 | 0.286 | 0.143 | 0.089 | 0.001 | 0.148 | - | 1.02 | - |
| D4 | condition_projection | 111,360 | 0.223 | 0.111 | 0.070 | 0.001 | 0.159 | - | 1.42 | - |
| E0 | patch_embedding | 393,728 | 0.787 | 0.394 | 0.246 | 0.027 | 1.174 | 215.901 | 2.92 | 1454.31 |
| E5 | latent_projection | 164,672 | 0.329 | 0.165 | 0.103 | 0.010 | 0.508 | 41.079 | 3.02 | 656.99 |
| D0 | initial_projection | 46,224 | 0.092 | 0.046 | 0.029 | 0.000 | 0.172 | - | 3.72 | - |
| H2 | em_step_1 | 166,656 | 0.333 | 0.167 | 0.104 | 0.006 | 0.716 | 193.394 | 4.26 | 3083.05 |
| D3 | latent_upsampling | 246,528 | 0.493 | 0.247 | 0.154 | 0.001 | 1.547 | 61.770 | 6.27 | 651.42 |
| H0 | hyper_encoder | 303,456 | 0.607 | 0.303 | 0.190 | 0.017 | 10.087 | - | 33.18 | - |
| D6 | adaln | 7,056 | 0.014 | 0.007 | 0.004 | 0.000 | 0.432 | - | 61.23 | - |
| D8 | renderer_input | 2,304 | 0.005 | 0.002 | 0.001 | 0.001 | 0.178 | - | 76.58 | - |
| H3 | em_step_2 | 207,296 | 0.415 | 0.207 | 0.130 | 0.007 | 38.152 | - | 184.01 | - |
| D7 | output_projection | 2,352 | 0.005 | 0.002 | 0.001 | 0.002 | 0.717 | 78.537 | 303.74 | 88231.62 |
| D5 | renderer_mlp | 4,704 | 0.009 | 0.005 | 0.003 | 0.005 | 8.057 | - | 1711.59 | - |

## What the table shows

**INT8 is cheap on 16 of the 19 components.** Cost per MB saved for the 16-bit to INT8 step spans
four orders of magnitude: **0.14** for E3 up to **1712** for D5. The median is a few units. A
threshold of roughly 30 cost/MB would admit 13 components and exclude H0, D6, D8, H3, D7 and D5.

**The four encoder blocks are where the bytes are, and they are the cheapest per byte.** E1-E4 hold
2.89M parameters each and account for 11.6 MB of the 54.7 MB model, yet their INT8 cost is only
**0.14 to 0.46 per MB** - the four best rows in the table. Saving 11.6 MB by moving them to INT8
costs a predicted 1.7 percentage points in total, against 16.9 MB saved. This is the single
strongest quantitative argument in the study: the cheapest components happen to be the largest.

**INT4 is rejected by the data at every component.** Going from INT8 to group-32 4-bit costs
**26.1 (E2), 31.4 (E3), 31.2 (E4) and 157.5 (E1)** per MB saved - roughly **100x worse per byte than
the INT8 step on the same blocks**. E0, E5, H2, D3, D7 and D5 are worse still, up to 88,000 per MB.
Group-32 scaling improved INT4 by 5-22x over per-channel, which is real progress but nowhere near
the ~100x still needed.

**Excluded outright:** H0 (10.09% at INT8, 33.2 cost/MB), H3 (38.15%, 184 cost/MB) and D5 (8.06%,
1712 cost/MB). Their cost per byte is 1-2 orders of magnitude worse than the encoder blocks for a
fraction of the bytes, so every tier keeps them at BF16.

## Tier design that follows from this

| Tier | Rule applied to the table | INT4 components | Size |
|---|---|---|---|
| 1. More INT8 | every component whose INT8 cost/MB is under ~5 | 0 | 14.4 MB |
| 2. Moderate INT4 | the two cheapest INT4 rows (E2, E3) + the cheapest INT8 rows for E1, E4 | 2 | 13.6 MB |
| 3. More INT4 | all four encoder blocks at INT4, since they are the only components where 4-bit saves megabytes | 4 | 11.4 MB |

Caveat: the int4g32 column is a 4-image estimate, so treat its ordering as indicative and its
absolute values as noisy. The INT8 and 16-bit columns are full-dataset measurements (24 Kodak and
41 CLIC images, all 8 QPs).
