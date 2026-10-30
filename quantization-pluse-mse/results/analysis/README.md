# Component x precision analysis

Formats included: FP16, BF16, FP8-E4M3, FP8-E5M2, INT8, INT4, FP4

| Component | FP16 | BF16 | FP8-E4M3 | FP8-E5M2 | INT8 | INT4 | FP4 | Safe to | QAT | Do not touch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 | 0.000 | 0.002 | 0.414 | 1.591 | 0.172 | 51.276 | 16.117 | INT8 | no | no |
| D1 | 0.001 | 0.009 | 1.085 | 3.985 | 0.124 | 26.738 | 20.261 | INT8 | no | no |
| D2 | 0.001 | 0.001 | 0.986 | 5.061 | 0.148 | 85.668 | 37.780 | INT8 | no | no |
| D3 | 0.001 | 0.005 | 2.041 | 9.159 | 1.547 | 479.394 | 215.237 | BF16 | yes | no |
| D4 | 0.001 | -0.001 | 3.041 | 13.840 | 0.159 | 62.235 | 53.939 | INT8 | no | no |
| D5 | 0.005 | 0.051 | 26.918 | 210.367 | 8.057 | 1170.481 | 877.627 | BF16 | yes | no |
| D6 | -0.001 | 0.036 | 3.661 | 13.521 | 0.432 | 95.757 | 79.348 | INT8 | no | no |
| D7 | 0.002 | 0.018 | 12.028 | 49.373 | 0.717 | 290.083 | 284.686 | INT8 | no | no |
| D8 | 0.001 | 0.082 | 4.769 | 8.736 | 0.178 | 51.517 | 40.234 | INT8 | no | no |
| E0 | 0.027 | 0.000 | 15.801 | 229.971 | 1.174 | 5016.665 | 3698.625 | BF16 | yes | no |
| E1 | 0.015 | 0.060 | 14.819 | 30.156 | 1.352 | 3767.782 | 529.945 | BF16 | yes | no |
| E2 | 0.010 | 0.047 | 1.653 | 9.227 | 0.509 | 166.860 | 128.176 | INT8 | no | no |
| E3 | 0.010 | 0.070 | 2.815 | 8.052 | 0.427 | 148.173 | 62.968 | INT8 | no | no |
| E4 | 0.015 | 0.033 | 3.970 | 12.547 | 0.710 | 374.508 | 100.341 | INT8 | no | no |
| E5 | 0.010 | 0.045 | 2.552 | 12.101 | 0.508 | 326.794 | 113.807 | INT8 | no | no |
| H0 | 0.017 | 0.110 | 11.572 | 17.216 | 10.087 | 220.016 | 95.569 | BF16 | yes | no |
| H1 | 0.005 | 0.014 | 0.553 | 5.490 | 0.131 | 13.429 | 11.449 | INT8 | no | no |
| H2 | 0.006 | 0.016 | 4.779 | 29.490 | 0.716 | 532.955 | 195.567 | INT8 | no | no |
| H3 | 0.007 | 0.024 | 46.506 | 439.642 | 38.152 | 163482.324 | 177781.038 | BF16 | yes | no |

## Recommendation logic

- `safe_to` = most aggressive format whose worst-QP mean penalty is <= 1%.
- `needs_qat` = INT8 is yellow/red; QAT may recover it.
- `do_not_quantize` = even FP16/BF16 is not green; keep FP32.
- Values are worst-QP mean penalty percentages.

## Files

- `component_precision_metrics.csv`
- `component_precision_penalty.csv`
- `component_precision_d_psnr.csv`
- `component_precision_size.csv`
- `recommendations.csv`
- `component_precision_table.png`
- `component_precision_penalty_heatmap.png`
- `component_precision_d_psnr_heatmap.png`
