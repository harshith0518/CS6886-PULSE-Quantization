"""Final queued stage: INT8 tier at 12000 steps, after everything else has finished.

Waits for `results/night/queue_results.json`, which the earlier queued runner writes last, then
fine-tunes the 14.401 MB INT8 tier with the augmented-Lagrangian controller and merges every tier's
result into one table for the report.

    python quantization-pluse-mse/scripts/queued_int8_12k.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "quantization-pluse-mse"
PY = sys.executable
DATASETS = ("kodak", "clic_professional", "set14")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hold_awake() -> bool:
    try:
        import ctypes
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001))
    except Exception:
        return False


def release_awake() -> None:
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass


def run(command: list[str], log_path: Path, label: str) -> int:
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n=== {now()} [{label}] {' '.join(command)}\n")
        stream.flush()
        code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, cwd=str(ROOT))
        stream.write(f"=== {now()} [{label}] exit {code}\n")
    return code


def merge(results_out: Path, log_path: Path) -> list[dict]:
    """One table across every tier and step count, for the report."""
    sources = [
        ("INT8 tier 2k", WORKSPACE / "results" / "final_2000steps" / "qat_attempt" / "summary.csv"),
        ("INT4 tier2 2k", WORKSPACE / "results" / "final_2000steps" / "qat_attempt" / "summary.csv"),
        ("INT8 tier 12k", results_out / "qat_int8_12k" / "summary.csv"),
        ("INT4 tier2 12k", WORKSPACE / "results" / "final" / "qat_attempt" / "summary.csv"),
    ]
    rows: list[dict] = []
    for label, path in sources:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                tier = row.get("tier", "")
                if label.startswith("INT8") and "int8" not in tier:
                    continue
                if label.startswith("INT4 tier2") and "tier2" not in tier:
                    continue
                rows.append({"group": label, **row})
    if rows:
        with (results_out / "all_tiers_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wait-for", default="quantization-pluse-mse/results/night/queue_results.json")
    parser.add_argument("--timeout-hours", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--output", default="quantization-pluse-mse/results/night")
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "queue_int8.log"
    awake = hold_awake()
    log_path.write_text(f"=== {now()} int8 stage start (sleep_guard={awake})\n", encoding="utf-8")

    if not args.skip_wait:
        target = ROOT / args.wait_for
        deadline = time.time() + args.timeout_hours * 3600
        while time.time() < deadline and not target.is_file():
            time.sleep(60)
        if not target.is_file():
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"=== {now()} timeout waiting for {target}\n")
            release_awake()
            return
        time.sleep(60)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"=== {now()} upstream queue finished, starting INT8 tier\n")

    result: dict = {"stage": "qat_int8_12k"}
    try:
        result["exit"] = run([
            PY, str(WORKSPACE / "scripts" / "qat_overnight.py"),
            "--tier", "tier1_more_int8",
            "--steps", str(args.steps), "--lr", "1e-6", "--accumulate", "1",
            "--objective", "augmented_lagrangian", "--rd-weight", "1.0",
            "--mu-alpha", "1.0", "--mu-max", "20.0",
            "--eval-every", "250", "--eval-images", "6",
            "--datasets", *DATASETS, "--output", str(out / "qat_int8_12k"),
        ], log_path, "qat_int8_12k")
        summary = out / "qat_int8_12k" / "summary.csv"
        if summary.is_file():
            with summary.open(newline="", encoding="utf-8") as stream:
                result["rows"] = list(csv.DictReader(stream))
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)

    result["merged"] = merge(out, log_path)
    try:
        run([PY, str(WORKSPACE / "scripts" / "run_all_stages.py"),
             "--stages", "qat_negative", "summary", "--force"], log_path, "reports")
    except Exception as error:
        result["report_error"] = str(error)

    release_awake()
    (out / "queue_int8_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"=== {now()} int8 stage done\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
