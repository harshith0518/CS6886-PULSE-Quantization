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

