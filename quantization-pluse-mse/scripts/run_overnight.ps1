$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$python = Join-Path $root "venv\Scripts\python.exe"
$logDir = Join-Path $root "quantization-pluse-mse\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
& $python "quantization-pluse-mse\scripts\pipeline.py" overnight `
  --datasets kodak clic_professional `
  --formats int8 int4 `
  --components all `
  --qps 0 1 2 3 4 5 6 7 `
  --skip-reference `
  --device cuda `
  --threads 1
