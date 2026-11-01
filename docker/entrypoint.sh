#!/bin/sh
# Fetch the released checkpoint on first use, then run whatever was asked for.
#
# The bundle is 84 MB, so it is not baked into the image. It lands in a named volume, which means the
# download happens once and later runs start instantly.
set -e

BUNDLE=/app/checkpoints/pulse-mse

if [ ! -f "$BUNDLE/model.pt" ]; then
    echo "checkpoint not found in $BUNDLE"
    echo "trying to fetch zhaoyangjia/PULSE from Hugging Face ..."
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download zhaoyangjia/PULSE --local-dir "$BUNDLE" || true
    else
        python -c "
from huggingface_hub import snapshot_download
snapshot_download('zhaoyangjia/PULSE', local_dir='$BUNDLE')
" || true
    fi
fi

if [ ! -f "$BUNDLE/model.pt" ]; then
    cat <<'EOF'

The automatic download did not produce a usable bundle.

A runnable bundle needs exactly these three files:
    checkpoints/pulse-mse/config.json
    checkpoints/pulse-mse/model.pt
    checkpoints/pulse-mse/entropy_control_int.pt

Download the pulse-mse release from https://huggingface.co/zhaoyangjia/PULSE
and mount the directory instead of using the named volume:

    docker run --rm \
      -v /path/to/pulse-mse:/app/checkpoints/pulse-mse:ro \
      -v /path/to/datasets:/app/datasets:ro \
      pulse-quantization:cpu \
      python quantization-pluse-mse/scripts/eval_assignment.py \
        --model checkpoints/pulse-mse \
        --configs quantization-pluse-mse/results/ilp/assignment_ilp.json \
        --dataset kodak --output /tmp/check

EOF
    exit 1
fi

echo "checkpoint ready: $(ls -1 "$BUNDLE" | tr '\n' ' ')"
exec "$@"
