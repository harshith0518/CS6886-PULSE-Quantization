# Single Linux container for the PULSE quantization study.
#
# CPU by default, which is enough to evaluate an assignment and to run the tests.
# Fine-tuning wants a GPU; see docker/README.md for the CUDA variant.
#
# Build from the repository root:
#   docker build -f docker/Dockerfile -t pulse-quantization:cpu .
#
# NOT YET BUILT OR TESTED BY US. Treat the first build as a check, not a formality.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TECTONIC_CACHE_DIR=/app/.scratch/tectonic-cache

# build-essential and pybind11 are required to compile the CPU rANS and
# integer-CDF kernels in codec/models/entropy/extensions/cpp.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The project documents that PyTorch must be installed first, so the CPU wheel is
# pinned here and the remaining pins follow. torch is deliberately absent from
# requirements.txt for exactly this reason.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0

COPY requirements.txt ./
RUN pip install -r requirements.txt pybind11

COPY . ./
RUN pip install -e . --no-build-isolation \
    && python -c "import MLCodec_extensions_cpp; print('entropy extension ok')"

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# The 84 MB checkpoint and the evaluation images are mounted, never baked in.
VOLUME ["/app/checkpoints", "/app/datasets"]

# The entrypoint fetches the checkpoint on first run, then runs the command.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command proves the install: ten unit tests over the quantization core.
CMD ["python", "-m", "unittest", "discover", "-s", "quantization-pluse-mse/tests"]
