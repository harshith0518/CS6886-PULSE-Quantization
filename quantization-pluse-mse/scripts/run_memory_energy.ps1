$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$python = Join-Path $root "venv\Scripts\python.exe"

while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "pipeline_extra.py" }) {
    Start-Sleep -Seconds 60
}

& $python "quantization-pluse-mse\scripts\bench_memory_energy.py" `
  --formats fp32 fp16 bf16 int8 int4 fp8_e4m3 fp8_e5m2 fp4_e2m1 `
  --components all `
  --qps 0 1 2 3 4 5 6 7 `
  --image datasets/kodak/kodim01.png `
  --repeat-seconds 10 `
  --idle-seconds 10 `
  --device cuda `
  --output "quantization-pluse-mse/results/memory_energy.csv"
