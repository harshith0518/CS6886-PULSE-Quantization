
#!/usr/bin/env python3
"""
PULSE-MSE baseline benchmark wrapper.

Collects:
- Official per-image/QP metrics from evaluate.py:
  bpp, PSNR, encode latency, decode latency, bytes, pixels
- Approximate whole-run GPU energy (J)
- Whole-run GPU memory usage (peak and average reported usage)
- Run metadata, logs, and a CSV run ledger

Important:
GPU energy and GPU memory in this script are whole-run measurements.
Per-image PyTorch allocated/reserved memory requires instrumentation
inside evaluate.py around the individual codec operations.
"""

import argparse
import csv
import datetime as dt
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PULSE-MSE and monitor whole-run GPU resources."
    )
    parser.add_argument("--model", required=True,
                        help="Official PULSE-MSE checkpoint path")
    parser.add_argument("--input", required=True,
                        help="Dataset folder or image path")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="Evaluation device")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--outdir", default="benchmark_results")
    parser.add_argument("--sample-ms", type=int, default=200,
                        help="GPU power/memory sampling interval in ms")
    args = parser.parse_args()

    if args.sample_ms < 50:
        parser.error("--sample-ms must be at least 50 ms")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_json = out / f"pulse_mse_{stamp}.json"
    log_file = out / f"pulse_mse_{stamp}.log"
    run_json = out / f"run_{stamp}.json"
    ledger = out / "run_ledger.csv"

    # Use the same Python interpreter/environment as this wrapper.
    cmd = [
        sys.executable, "evaluate.py",
        "--model", args.model,
        "--input", args.input,
        "--output", str(result_json),
        "--qps", *map(str, range(8)),
        "--device", args.device,
        "--threads", str(args.threads),
    ]

    smi = shutil.which("nvidia-smi")
    samples = []
    stop_sampling = threading.Event()
    proc = None
    interrupted = False
    exit_code = None
    start_time = time.perf_counter()

    def query_gpu():
        """Return timestamp, power W, used memory MiB, total memory MiB."""
        if not smi:
            return

        try:
            result = subprocess.run(
                [
                    smi,
                    "--query-gpu=power.draw,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )

            if result.returncode != 0:
                return

            # Assumes one visible GPU. Multiple GPU output is handled by
            # taking the first GPU, matching the usual single-GPU setup.
            line = result.stdout.strip().splitlines()[0]
            fields = [part.strip() for part in line.split(",")]

            if len(fields) != 3:
                return

            def parse_value(value):
                if value.lower() in ("n/a", "[n/a]", "not supported"):
                    return None
                return float(value)

            power = parse_value(fields[0])
            used = parse_value(fields[1])
            total = parse_value(fields[2])

            samples.append({
                "t": time.perf_counter(),
                "power_w": power,
                "memory_used_mib": used,
                "memory_total_mib": total,
            })

        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass

    def sample_gpu():
        while not stop_sampling.is_set():
            query_gpu()
            stop_sampling.wait(args.sample_ms / 1000.0)

    gpu_info = "unavailable"
    if smi:
        try:
            info = subprocess.run(
                [
                    smi,
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if info.returncode == 0:
                gpu_info = info.stdout.strip() or "GPU details unavailable"
        except (OSError, subprocess.SubprocessError):
            gpu_info = "nvidia-smi found but query failed"

    print("GPU:", gpu_info)
    print("Device:", args.device)
    print("Output directory:", out.resolve())
    print("Running:", subprocess.list2cmdline(cmd))
    print("Expected evaluations: 8 QPs × number of input images")
    print("GPU sampling interval:", args.sample_ms, "ms\n")

    worker = threading.Thread(target=sample_gpu, daemon=True)
    worker.start()

    try:
        with log_file.open("w", encoding="utf-8") as log:
            log.write("COMMAND: " + subprocess.list2cmdline(cmd) + "\n")
            log.write("GPU: " + gpu_info + "\n\n")
            log.flush()

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
            )

            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()

            exit_code = proc.wait()

    except KeyboardInterrupt:
        interrupted = True
        print("\n\nInterrupt received. Stopping evaluator...")
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        exit_code = proc.returncode if proc is not None else 130

    finally:
        stop_sampling.set()
        worker.join(timeout=4)

        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    elapsed = time.perf_counter() - start_time

    # Integrate GPU power over the observed sampling interval.
    valid_power = [
        s for s in samples if s["power_w"] is not None
    ]
    energy_j = None
    if len(valid_power) >= 2:
        valid_power.sort(key=lambda s: s["t"])
        energy_j = sum(
            (valid_power[i - 1]["power_w"] + valid_power[i]["power_w"])
            / 2.0
            * (valid_power[i]["t"] - valid_power[i - 1]["t"])
            for i in range(1, len(valid_power))
        )

    # These are total GPU memory readings, not process-specific PyTorch
    # allocated/reserved memory.
    valid_mem = [
        s["memory_used_mib"] for s in samples
        if s["memory_used_mib"] is not None
    ]
    peak_gpu_mem_mib = max(valid_mem) if valid_mem else None
    avg_gpu_mem_mib = (
        sum(valid_mem) / len(valid_mem) if valid_mem else None
    )

    total_mem_values = [
        s["memory_total_mib"] for s in samples
        if s["memory_total_mib"] is not None
    ]
    total_gpu_mem_mib = total_mem_values[-1] if total_mem_values else None

    metrics_exists = result_json.exists()
    metrics_size = result_json.stat().st_size if metrics_exists else 0

    meta = {
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "command": cmd,
        "gpu": gpu_info,
        "device": args.device,
        "threads": args.threads,
        "wall_time_s": elapsed,
        "exit_code": exit_code,
        "interrupted": interrupted,
        "power_sample_interval_ms": args.sample_ms,
        "gpu_samples": len(samples),
        "valid_power_samples": len(valid_power),
        "gpu_energy_estimate_j_whole_run": energy_j,
        "gpu_energy_scope": (
            "Approximate GPU energy over the observed evaluation process; "
            "not per-image energy and not whole-laptop energy."
        ),
        "gpu_memory_peak_used_mib_whole_run": peak_gpu_mem_mib,
        "gpu_memory_average_used_mib_whole_run": avg_gpu_mem_mib,
        "gpu_memory_total_mib": total_gpu_mem_mib,
        "gpu_memory_scope": (
            "nvidia-smi total GPU memory usage sampled during the run; "
            "may include other processes and is not PyTorch allocated/reserved memory."
        ),
        "metrics_json": str(result_json),
        "metrics_json_exists": metrics_exists,
        "metrics_json_size_bytes": metrics_size,
        "log": str(log_file),
    }

    run_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Append one metadata row per run. Keep columns stable across runs.
    columns = list(meta.keys())
    row = {
        key: json.dumps(value) if isinstance(value, (list, dict)) else value
        for key, value in meta.items()
    }

    ledger_exists = ledger.exists() and ledger.stat().st_size > 0
    with ledger.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=columns, extrasaction="ignore"
        )
        if not ledger_exists:
            writer.writeheader()
        writer.writerow(row)

    print("\n========== BENCHMARK SUMMARY ==========")
    print(f"Wall time: {elapsed:.2f} s")
    print(f"GPU samples: {len(samples)}")
    print(
        "Whole-run GPU energy estimate:",
        f"{energy_j:.2f} J" if energy_j is not None else "unavailable",
    )
    print(
        "Peak total GPU memory used:",
        f"{peak_gpu_mem_mib:.2f} MiB"
        if peak_gpu_mem_mib is not None else "unavailable",
    )
    print(
        "Average total GPU memory used:",
        f"{avg_gpu_mem_mib:.2f} MiB"
        if avg_gpu_mem_mib is not None else "unavailable",
    )
    print("Official metrics JSON:", result_json)
    print("Run metadata JSON:", run_json)
    print("Detailed log:", log_file)
    print("Run ledger CSV:", ledger)

    if interrupted:
        print(
            "\nWARNING: Run was interrupted. Treat it as incomplete. "
            "The official metrics JSON may be missing or incomplete."
        )
    elif exit_code != 0:
        print(
            "\nWARNING: Evaluator failed. Do not use this run as a valid "
            "baseline until the error is fixed and the run completes."
        )

    if len(valid_power) < 3:
        print(
            "WARNING: Too few valid power samples for a reliable "
            "whole-run energy estimate."
        )

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
