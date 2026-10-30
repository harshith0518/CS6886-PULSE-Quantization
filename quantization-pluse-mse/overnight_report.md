# Overnight INT8/INT4 run

Datasets: clic_professional
Formats: fp4_e2m1
Sweep rows: 69233
Configs: 135
Weight-stat rows: 698
Failures: 0

## Status counts

| Status | Count |
|---|---:|
| ok | 69233 |

## Notes

- No quantized checkpoints are saved; weights are restored after every run.
- Fake-quant latency, memory and energy are excluded from this sweep.
- CDF/symbol mismatch counts are not implemented; correctness is decode/stream/recon/weight-hash based.
- Full-resolution Kodak is skipped if `datasets/kodak` has fewer than 24 files.
