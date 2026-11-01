# Container

One Linux container that can run the study. CPU is enough for evaluation, the solver and the tests.
Fine-tuning wants a GPU.

**Status: written but not built here.** The machine this project was developed on has Windows and no
Docker daemon, so the image has never been built. Treat the first build as a test of the recipe, not a
formality, and expect to adjust one or two lines. The commands below are what the recipe intends.

## What is in the image

| Included | Excluded (mounted at run time) |
|---|---|
| Python 3.12, CPU PyTorch 2.6, the pinned dependencies | The 84 MB checkpoint |
| The compiled entropy extension (`MLCodec_extensions_cpp`) | The evaluation images |
| The codec, the study scripts and the result tables | Output results |
| The documents are not needed | PDFs, decks, the TeX binary |

## Where the models come from

| Model | In the image? | How you get it |
|---|---|---|
| The FP32 base, `pulse-mse` (84 MB) | **No** | The entrypoint downloads it once into a named volume on first run |
| The three quantized operating points | **Yes, as precision maps** | They are JSON files in `quantization-pluse-mse/results/ilp/` and need no download |
| A fine-tuned bundle (56 MB per tier) | No | Regenerate with `qat_overnight.py`; not shipped |

This is the useful split. A "quantized model" here is **a precision map applied to the base
checkpoint at load time**, not a separate weight file. So the repository already carries all three
operating points, and the only download is the 84 MB base model.

The first `docker compose run` downloads the base model into the `pulse-checkpoints` volume. Later
runs start instantly. To use a copy you already have, comment the named volume in
`docker-compose.yml` and uncomment the bind mount beside it.

## Prerequisites

- Docker 24 or newer, with the Compose plugin (`docker compose`).
- About 1 GB of free disk for the image and the model.
- A local `datasets/kodak/` with the 24 Kodak images, mounted read-only. Add `clic_professional/` and
  `set14/` for the full comparison.

The dataset mount is read-only, so the container cannot damage your data.

## Build

From the repository root:

```bash
docker build -f docker/Dockerfile -t pulse-quantization:cpu .
```

Expect 10 to 20 minutes on a first build: the CPU PyTorch wheel is large and the native extension
compiles once.

## Run

```bash
# 1. prove the install: ten unit tests over the quantization core
docker compose -f docker/docker-compose.yml run --rm app

# 2. reproduce the headline number: expect 21.677 MB and 0.672% on Kodak
docker compose -f docker/docker-compose.yml run --rm verify

# 3. solve a different budget: expect 21.677 MB at 1.0% and 18.664 MB at 1.8%
docker compose -f docker/docker-compose.yml run --rm solve
```

Results from `verify` land in `results-docker/check/` on the host.

## Interactive shell

```bash
docker run --rm -it \
  -v "$PWD/checkpoints:/app/checkpoints:ro" \
  -v "$PWD/datasets:/app/datasets:ro" \
  -v "$PWD/results-docker:/app/results" \
  pulse-quantization:cpu bash
```

## GPU variant

The CPU image is deliberate: it builds everywhere and covers evaluation, the solver and the tests.
For fine-tuning, change the first two lines of the Dockerfile:

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.12 python3.12-venv python3-pip \
    build-essential git && rm -rf /var/lib/apt/lists/*
```

and install the CUDA wheel instead of the CPU one:

```dockerfile
RUN pip install torch==2.6.0
```

Then run with `--gpus all`. Fine-tuning needs about 4 GB of VRAM at 512-pixel crops.

## Why this is not the default path

`RUNBOOK.md` describes a native install and is the tested route: every number in the report was
produced that way. The container exists so that a reader does not have to reproduce a Windows plus
CUDA setup, and so that the evaluation path has a fixed environment. If the container and the runbook
disagree, trust the runbook, then fix the container.
