# Pipeline

One entry point: `pipeline.py`.

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\pipeline.py inventory
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\pipeline.py reference --dataset kodak_thumb
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\pipeline.py reference --dataset kodak
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\pipeline.py sweep --format int8 --components all --dataset clic_professional
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\pipeline.py export --format int8
```

## Output per precision

```text
int8/results/
  README.md              baseline + component x QP tables + size + weight stats
  sweep_int8.csv         per-image/QP core metrics
  configs_int8.csv       per-config assertions and analytic size
  weight_stats_int8.csv  per-layer SQNR, clip fraction, max abs error
  summary.csv            derived mean/median/p90 aggregates
  reference.json         baseline, noise floor, G0a, slopes
  manifests.json         relevant run manifests
  figures/               5 metric line plots + 1 size bar
```

## Rules

- Resume key: `(config_id, dataset, image, qp)`, where `config_id` is the quant-config hash prefix.
- No quantized checkpoint is written.
- No per-component or per-dataset plot families.
- Energy is N/A in the quality sweep.
