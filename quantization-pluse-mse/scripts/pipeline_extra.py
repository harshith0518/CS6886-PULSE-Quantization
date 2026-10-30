"""Run the extra floating-point precisions through the same pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXTRA = ["fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["kodak", "clic_professional"])
    parser.add_argument("--formats", nargs="+", default=EXTRA)
    parser.add_argument("--components", nargs="+", default=["all"])
    parser.add_argument("--qps", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ledger", default="quantization-pluse-mse/results/_ledger")
    parser.add_argument("--output-root", default="quantization-pluse-mse")
    parser.add_argument("--log-dir", default="quantization-pluse-mse/logs")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pipeline = Path(__file__).with_name("pipeline.py")
    cmd = [
        sys.executable, str(pipeline), "overnight",
        "--datasets", *args.datasets,
        "--formats", *args.formats,
        "--components", *args.components,
        "--qps", *map(str, args.qps),
        "--device", args.device,
        "--threads", str(args.threads),
        "--ledger", args.ledger,
        "--output-root", args.output_root,
        "--log-dir", args.log_dir,
    ]
    if args.max_images:
        cmd += ["--max-images", str(args.max_images)]
    if args.skip_reference:
        cmd += ["--skip-reference"]
    if args.dry_run:
        cmd += ["--dry-run"]
    print(subprocess.list2cmdline(cmd), flush=True)
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
