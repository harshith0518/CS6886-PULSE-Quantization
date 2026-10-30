# Library layout

Three universal modules; no per-precision code.

| Module | Responsibility |
|---|---|
| `pulsequant/core.py` | Component map, INT8/INT4 fake quant, in-place apply/restore, weight hashes |
| `pulsequant/eval.py` | Codec evaluation, metrics, manifests, progress JSONL |
| `pulsequant/plots.py` | Pillow line/bar/size plots |

## Rules

1. One component at a time; everything else FP32.
2. Mutate in place and restore from a cached FP32 clone.
3. Weight hashes prove change and restore.
4. Keep the entropy artifact and CDF tables read-only.
5. Simulated formats are never given speed or energy claims.
