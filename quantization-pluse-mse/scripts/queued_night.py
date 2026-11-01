"""Wait for the running 12k QAT job, then run the remaining night work unattended.

Chain:
  1. wait for results/final/qat_attempt/summary.csv to appear (the 12k run writes it last)
  2. fine ILP sweep at 1.2 / 1.4 / 1.6 / 1.8 percent, measure the two that matter (1.6, 1.8)
  3. QAT on the ILP 2 percent assignment: the only attempt that could beat the current champion
  4. regenerate the report stages

The process holds the machine awake, logs every step to one file, and a failing stage does not stop
the ones after it.

    python quantization-pluse-mse/scripts/queued_night.py
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


def wait_for(path: Path, log_path: Path, timeout_hours: float) -> bool:
    deadline = time.time() + timeout_hours * 3600
    while time.time() < deadline:
        if path.is_file():
            time.sleep(60)  # let the writer close the file
            return True
        time.sleep(60)
    return False


def read_worst(directory: Path, dataset: str) -> float | None:
    path = directory / f"configs_{dataset}.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None
    return float(rows[0]["worst_qp_mean_penalty_pct"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wait-for", default="quantization-pluse-mse/results/final/qat_attempt/summary.csv")
    parser.add_argument("--timeout-hours", type=float, default=6.0)
    parser.add_argument("--qat-steps", type=int, default=8000)
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
    log_path = out / "queue.log"
    awake = hold_awake()
    log_path.write_text(f"=== {now()} queued runner start (sleep_guard={awake})\n", encoding="utf-8")

    if not args.skip_wait:
        waited = time.time()
        ready = wait_for(ROOT / args.wait_for, log_path, args.timeout_hours)
        minutes = (time.time() - waited) / 60
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"=== {now()} wait finished ready={ready} after {minutes:.0f} min\n")
        if not ready:
            print(json.dumps({"stage": "wait", "status": "timeout"}), flush=True)
            release_awake()
            return

    results = []

    # ---- stage A: fine ILP sweep, measure the two budgets near the knee
    try:
        code = run([PY, str(WORKSPACE / "scripts" / "solve_ilp.py"),
                    "--budget", "1.2", "1.4", "1.6", "1.8", "--headline", "1.6"],
                   log_path, "ilp_sweep")
        solved = []
        for budget in ("1.2", "1.4", "1.6", "1.8"):
            run([PY, str(WORKSPACE / "scripts" / "solve_ilp.py"),
                 "--budget", budget, "--headline", budget], log_path, f"ilp_{budget}")
            source = WORKSPACE / "results" / "ilp" / "assignment_ilp.json"
            target = WORKSPACE / "results" / "ilp" / f"assignment_ilp_b{budget.replace('.', 'p')}.json"
            if source.is_file():
                target.write_bytes(source.read_bytes())
                payload = json.loads(source.read_text(encoding="utf-8"))
                solved.append({"budget_pct": float(budget),
                               "size_mb": payload["predicted"]["size_mb"],
                               "predicted_pct": payload["predicted"]["worst_qp_penalty_pct"]})
        (out / "ilp_fine_sweep.json").write_text(json.dumps(solved, indent=2) + "\n", encoding="utf-8")
        # leave the canonical 1 percent champion in place, the sweep overwrote it
        run([PY, str(WORKSPACE / "scripts" / "solve_ilp.py"), "--budget", "1.0", "--headline", "1.0"],
            log_path, "ilp_restore_1p0")

        measured = {}
        for budget in ("1.6", "1.8"):
            assignment = WORKSPACE / "results" / "ilp" / f"assignment_ilp_b{budget.replace('.', 'p')}.json"
            if not assignment.is_file():
                continue
            directory = out / f"ilp_b{budget.replace('.', 'p')}_eval"
            for dataset in DATASETS:
                run([PY, str(WORKSPACE / "scripts" / "eval_assignment.py"),
                     "--model", "checkpoints/pulse-mse", "--configs", str(assignment),
                     "--output", str(directory), "--dataset", dataset], log_path, f"eval_{budget}_{dataset}")
            measured[budget] = {d: read_worst(directory, d) for d in DATASETS}
        (out / "ilp_fine_measured.json").write_text(json.dumps(measured, indent=2) + "\n", encoding="utf-8")
        results.append({"stage": "ilp_fine", "exit": code, "solved": solved, "measured": measured})
    except Exception as error:
        results.append({"stage": "ilp_fine", "status": "failed", "error": str(error)})

    # ---- stage B: QAT on the ILP 2 percent assignment
    try:
        assignment = json.loads((WORKSPACE / "results" / "ilp" / "assignment_ilp_b2p0.json")
                                .read_text(encoding="utf-8"))["assignment"]
        tier_file = WORKSPACE / "configs" / "qat_ilp2p0.json"
        tier_file.write_text(json.dumps({
            "ilp_2pct_qat": {
                "label": "ILP 2% + QAT",
                "intent": "fine-tune the measured 18.389 MB assignment to recover its 1.94% penalty",
                "predicted_worst_qp_penalty_pct": 1.9864,
                "target_size_mb": 18.389,
                "assignment": assignment,
            }
        }, indent=2) + "\n", encoding="utf-8")
        code = run([PY, str(WORKSPACE / "scripts" / "qat_overnight.py"),
                    "--tier", "ilp_2pct_qat", "--tiers", str(tier_file),
                    "--steps", str(args.qat_steps), "--lr", "1e-6", "--accumulate", "1",
                    "--objective", "augmented_lagrangian", "--rd-weight", "1.0",
                    "--mu-alpha", "1.0", "--mu-max", "20.0",
                    "--eval-every", "250", "--eval-images", "6",
                    "--datasets", *DATASETS, "--output", str(out / "qat_ilp2p0")],
                   log_path, "qat_ilp2p0")
        rows = []
        summary = out / "qat_ilp2p0" / "summary.csv"
        if summary.is_file():
            with summary.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        results.append({"stage": "qat_ilp2p0", "exit": code, "rows": rows})
    except Exception as error:
        results.append({"stage": "qat_ilp2p0", "status": "failed", "error": str(error)})

    # ---- stage C: regenerate the report stages
    try:
        code = run([PY, str(WORKSPACE / "scripts" / "run_all_stages.py"),
                    "--stages", "qat_negative", "summary", "--force"],
                   log_path, "reports")
        results.append({"stage": "reports", "exit": code})
    except Exception as error:
        results.append({"stage": "reports", "status": "failed", "error": str(error)})

    release_awake()
    (out / "queue_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"=== {now()} queue done: {json.dumps(results)[:2000]}\n")
    print(json.dumps({"stage": "queue", "status": "done", "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
