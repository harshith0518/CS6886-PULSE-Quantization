import os
import json
import torch
import torch.nn as nn

# ============================================================
# SETTINGS
# ============================================================

MODEL_DIR = os.path.join("checkpoints", "pulse-mse")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# ============================================================
# BASIC SYSTEM INFO
# ============================================================

print("=" * 80)
print("SYSTEM / PYTORCH INFO")
print("=" * 80)

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version    :", torch.version.cuda)
    print("GPU             :", torch.cuda.get_device_name(0))
    print(
        "VRAM            :",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )

print()

# ============================================================
# CHECK FILES
# ============================================================

print("=" * 80)
print("CHECKPOINT FILES")
print("=" * 80)

print("Model path  :", MODEL_PATH)
print("Config path :", CONFIG_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Could not find {MODEL_PATH}")

if not os.path.exists(CONFIG_PATH):
    print("WARNING: config.json not found")

print()

# ============================================================
# CONFIG
# ============================================================

if os.path.exists(CONFIG_PATH):

    print("=" * 80)
    print("CONFIG.JSON")
    print("=" * 80)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    print(json.dumps(config, indent=2))

    print()

# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("=" * 80)
print("CHECKPOINT INSPECTION")
print("=" * 80)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

print("Checkpoint type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("Checkpoint keys:")
    for k in checkpoint.keys():
        print("  ", k)

print()

# ============================================================
# FIND STATE DICT
# ============================================================

state_dict = None

if isinstance(checkpoint, dict):

    possible_keys = [
        "state_dict",
        "model",
        "model_state_dict",
        "net",
        "network"
    ]

    for key in possible_keys:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state_dict = checkpoint[key]
            print("Using state dict key:", key)
            break

    if state_dict is None:
        # Sometimes checkpoint itself is state_dict
        if all(torch.is_tensor(v) for v in checkpoint.values()):
            state_dict = checkpoint

if state_dict is None:
    raise RuntimeError(
        "Could not identify a state_dict inside model.pt. "
        "Print the checkpoint structure above and send it to me."
    )

print("Number of tensors:", len(state_dict))
print()

# ============================================================
# STATE DICT SUMMARY
# ============================================================

print("=" * 80)
print("STATE DICT TENSORS")
print("=" * 80)

total_params = 0
dtype_counts = {}

largest_tensors = []

for name, tensor in state_dict.items():

    if not torch.is_tensor(tensor):
        continue

    numel = tensor.numel()
    total_params += numel

    dtype_counts[str(tensor.dtype)] = (
        dtype_counts.get(str(tensor.dtype), 0) + 1
    )

    largest_tensors.append(
        (numel, name, tuple(tensor.shape), str(tensor.dtype))
    )

largest_tensors.sort(reverse=True)

print("Total tensor elements:", f"{total_params:,}")
print()

print("Dtype counts:")
for dtype, count in dtype_counts.items():
    print(f"  {dtype}: {count}")

print()

print("Largest tensors:")
for numel, name, shape, dtype in largest_tensors[:30]:

    size_mb = numel * 4 / 1024**2

    print(
        f"{name:70s} "
        f"shape={str(shape):25s} "
        f"elements={numel:12,d} "
        f"dtype={dtype:15s} "
        f"FP32_size={size_mb:8.2f} MB"
    )

print()

# ============================================================
# TRY TO IMPORT PULSE MODEL
# ============================================================

print("=" * 80)
print("IMPORTING PULSE MODEL")
print("=" * 80)

try:

    from codec.models.pulse import PULSE

    print("Successfully imported codec.models.pulse.PULSE")

except Exception as e:

    print("Could not import PULSE directly.")
    print("ERROR:", repr(e))
    print()
    print(
        "This is important. Send me the complete output "
        "up to this point."
    )
    raise

print()

# ============================================================
# INSPECT CONSTRUCTOR
# ============================================================

import inspect

print("=" * 80)
print("PULSE CLASS")
print("=" * 80)

print(inspect.signature(PULSE))

print()

# ============================================================
# TRY TO CONSTRUCT MODEL
# ============================================================

print("=" * 80)
print("CONSTRUCTING MODEL")
print("=" * 80)

model = None

construction_attempts = []

# Try config-based construction first
try:
    model = PULSE(config)
    construction_attempts.append("PULSE(config)")
except Exception as e:
    construction_attempts.append(
        f"PULSE(config) FAILED: {repr(e)}"
    )

# Try no-argument construction
if model is None:
    try:
        model = PULSE()
        construction_attempts.append("PULSE()")
    except Exception as e:
        construction_attempts.append(
            f"PULSE() FAILED: {repr(e)}"
        )

print("Construction attempts:")
for x in construction_attempts:
    print("  ", x)

if model is None:

    print()
    print(
        "Could not automatically construct the model."
    )
    print(
        "Send me this output and I will give you the "
        "exact loading script for this PULSE version."
    )
    raise SystemExit

print()

# ============================================================
# MODEL STRUCTURE
# ============================================================

print("=" * 80)
print("MODEL STRUCTURE")
print("=" * 80)

print(model)

print()

# ============================================================
# PARAMETER SUMMARY
# ============================================================

print("=" * 80)
print("PARAMETER SUMMARY")
print("=" * 80)

total = 0
trainable = 0

for name, param in model.named_parameters():

    n = param.numel()

    total += n

    if param.requires_grad:
        trainable += n

print("Total parameters     :", f"{total:,}")
print("Trainable parameters :", f"{trainable:,}")
print(
    "FP32 parameter size  :",
    round(total * 4 / 1024**2, 2),
    "MB"
)
print(
    "INT8 parameter size  :",
    round(total / 1024**2, 2),
    "MB"
)

print()

# ============================================================
# PARAMETER COUNT BY TOP-LEVEL MODULE
# ============================================================

print("=" * 80)
print("PARAMETERS BY TOP-LEVEL MODULE")
print("=" * 80)

for name, module in model.named_children():

    params = sum(p.numel() for p in module.parameters())

    print(
        f"{name:40s} "
        f"{params:15,d} params "
        f"{params * 4 / 1024**2:10.2f} MB FP32"
    )

print()

# ============================================================
# ALL MODULES
# ============================================================

print("=" * 80)
print("ALL MODULES")
print("=" * 80)

for name, module in model.named_modules():

    if name == "":
        name = "<ROOT>"

    params = sum(p.numel() for p in module.parameters())

    if isinstance(module, nn.Conv2d):

        print(
            f"{name:70s} "
            f"{type(module).__name__:20s} "
            f"params={params:12,d} "
            f"kernel={module.kernel_size} "
            f"stride={module.stride} "
            f"groups={module.groups} "
            f"in={module.in_channels} "
            f"out={module.out_channels}"
        )

    elif isinstance(module, nn.Linear):

        print(
            f"{name:70s} "
            f"{type(module).__name__:20s} "
            f"params={params:12,d} "
            f"in={module.in_features} "
            f"out={module.out_features}"
        )

    elif isinstance(
        module,
        (
            nn.ConvTranspose2d,
            nn.PixelShuffle,
            nn.LayerNorm,
            nn.BatchNorm2d,
            nn.GroupNorm,
            nn.ReLU,
            nn.GELU,
            nn.SiLU
        )
    ):

        print(
            f"{name:70s} "
            f"{type(module).__name__:20s} "
            f"params={params:12,d}"
        )

print()

# ============================================================
# LOAD WEIGHTS
# ============================================================

print("=" * 80)
print("LOADING WEIGHTS")
print("=" * 80)

try:

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    print("Missing keys   :", len(missing))
    print("Unexpected keys:", len(unexpected))

    if missing:
        print("\nMissing:")
        for x in missing[:50]:
            print("  ", x)

    if unexpected:
        print("\nUnexpected:")
        for x in unexpected[:50]:
            print("  ", x)

except Exception as e:

    print("FAILED TO LOAD WEIGHTS")
    print(repr(e))
    raise

print()

# ============================================================
# PARAMETER DTYPE
# ============================================================

print("=" * 80)
print("MODEL PARAMETER DTYPES")
print("=" * 80)

dtype_counts = {}

for name, param in model.named_parameters():

    dtype = str(param.dtype)

    dtype_counts[dtype] = dtype_counts.get(dtype, 0) + param.numel()

for dtype, count in dtype_counts.items():

    print(
        f"{dtype:15s}: "
        f"{count:,} elements "
        f"({count / total * 100:.2f}%)"
    )

print()

# ============================================================
# SAVE A COMPACT REPORT
# ============================================================

report_path = "pulse_mse_model_report.txt"

with open(report_path, "w", encoding="utf-8") as f:

    f.write("PULSE-MSE MODEL REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"PyTorch: {torch.__version__}\n")
    f.write(f"CUDA available: {torch.cuda.is_available()}\n")

    if torch.cuda.is_available():
        f.write(
            f"GPU: {torch.cuda.get_device_name(0)}\n"
        )

    f.write("\n")

    f.write(f"Total parameters: {total:,}\n")
    f.write(f"Trainable parameters: {trainable:,}\n")
    f.write(
        f"FP32 parameter size: "
        f"{total * 4 / 1024**2:.2f} MB\n"
    )
    f.write(
        f"INT8 parameter size: "
        f"{total / 1024**2:.2f} MB\n"
    )

    f.write("\n\nMODEL STRUCTURE\n")
    f.write("=" * 80 + "\n")
    f.write(str(model))

    f.write("\n\nMODULES\n")
    f.write("=" * 80 + "\n")

    for name, module in model.named_modules():

        params = sum(p.numel() for p in module.parameters())

        f.write(
            f"{name:70s} "
            f"{type(module).__name__:25s} "
            f"params={params:,}\n"
        )

print("=" * 80)
print("DONE")
print("=" * 80)

print()
print("Report saved as:")
print(os.path.abspath(report_path))