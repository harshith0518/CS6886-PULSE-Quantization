$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$python = Join-Path $root "venv\Scripts\python.exe"
& $python "quantization-pluse-mse\scripts\pipeline_extra.py" `
  --datasets kodak clic_professional `
  --formats fp16 bf16 fp8_e4m3 fp8_e5m2 fp4_e2m1 `
  --components all `
  --qps 0 1 2 3 4 5 6 7 `
  --skip-reference `
  --device cuda `
  --threads 1
