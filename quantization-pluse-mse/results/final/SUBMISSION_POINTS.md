# Submission operating points

Two measured points, both evaluated on all three datasets with the same harness, the same
eight QPs and the same FP32 reference. Nothing here is a prediction.

| Point | Size | Saving | Kodak | CLIC | Set14 |
|---|---:|---:|---:|---:|---:|
| 1% budget champion | 21.677 MB | -60.36% | 0.672% | 0.851% | 0.935% |
| 2% budget variant | 18.389 MB | -66.38% | 1.395% | 1.936% | 1.707% |

The 2% variant is 3.288 MB smaller than the champion and costs about one extra point of
worst-QP penalty on every dataset. Both are inside a 2% budget, so the pair reads as a
size-quality trade rather than a single number.

