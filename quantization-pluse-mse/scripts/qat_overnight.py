"""Run the three QAT tiers sequentially, unattended, with before/after measurement.

For each tier it: measures the assignment before fine-tuning, fine-tunes, measures again, and
records the recovery. Errors in one tier do not stop the others. Everything is appended to a log
and summarised at the end.

    # full run
    python quantization-pluse-mse/scripts/qat_overnight.py

    # wiring check, about two minutes
    python quantization-pluse-mse/scripts/qat_overnight.py --smoke
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
PY = sys.executable

TIER_ORDER = ["tier1_more_int8", "tier2_moderate_int4", "tier3_more_int4"]


def hold_awake() -> bool:
    """Stop Windows sleeping mid-run. The previous overnight run died when the machine slept."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        return bool(kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
    except Exception:
        return False


def release_awake() -> None:
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def run(command: list[str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n=== {now()} {' '.join(command)}\n")
        stream.flush()
        code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, cwd=str(ROOT))
        stream.write(f"=== {now()} exit {code}\n")
    return code


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarise(phase_dir: Path, datasets: list[str]) -> dict:
    worst, per_dataset = None, {}
    for dataset in datasets:
        rows = read_csv(phase_dir / f"configs_{dataset}.csv")
        if not rows:
            continue
        row = rows[0]
        value = row.get("worst_qp_mean_penalty_pct") or ""
        per_dataset[dataset] = {
            "worst_qp_penalty_pct": float(value) if value not in ("", None) else None,
            "size_mb": float(row["quant_mb"]) if row.get("quant_mb") else None,
            "saving_pct": float(row["saving_pct"]) if row.get("saving_pct") else None,
        }
        if value not in ("", None):
            worst = max(worst or 0.0, float(value))
    return {"worst_qp_penalty_pct": worst, "per_dataset": per_dataset}


def evaluate(model_path: str, assignment_path: Path, datasets: list[str],
             output: Path, max_images: int, log_path: Path) -> int:
    code = 0
    for dataset in datasets:
        command = [PY, str(ROOT / "quantization-pluse-mse" / "scripts" / "eval_assignment.py"),
                   "--model", model_path, "--output", str(output),
                   "--configs", str(assignment_path), "--dataset", dataset]
        if max_images:
            command += ["--max-images", str(max_images)]
        code = run(command, log_path) or code
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/pulse-mse")
    parser.add_argument("--tiers", default="quantization-pluse-mse/configs/qat_tiers.json")
    parser.add_argument("--tier", nargs="+", default=TIER_ORDER)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--rd-weight", type=float, default=1.0)
    parser.add_argument("--rate-weight", type=float, default=1.0)
    parser.add_argument("--objective", choices=["rate_preserving", "rd"], default="rate_preserving")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-images", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--scope", default="backbone")
    parser.add_argument("--datasets", nargs="+", default=["kodak", "clic_professional", "set14"])
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--pool", default="datasets/qat_pool/div2k_valid")
    parser.add_argument("--output", default="quantization-pluse-mse/results/qat")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="redo phases whose output exists")
    parser.add_argument("--smoke", action="store_true", help="3 steps, kodak only, 2 images")
    args = parser.parse_args()

    if args.smoke:
        args.steps = 3
        args.datasets = ["kodak"]
        args.max_images = 2
        args.accumulate = 1

    awake = hold_awake()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "overnight.log"
    log_path.write_text(f"=== {now()} start\n", encoding="utf-8")
    log({"events": "start", "sleep_guard": str(bool(awake))})
    tiers = json.loads(Path(args.tiers).read_text(encoding="utf-8"))

    results = []
    for name in args.tier:
        if name not in tiers:
            log({"tier": name, "events": "skipped", "reason": "not in tier file"})
            continue
        tier_dir = out / name
        tier_dir.mkdir(parents=True, exist_ok=True)
        assignment_path = tier_dir / "assignment.json"
        assignment_path.write_text(
            json.dumps({"name": name, "note": tiers[name].get("intent", ""),
                        "assignment": tiers[name]["assignment"]}, indent=2) + "\n",
            encoding="utf-8")

        record = {"tier": name, "label": tiers[name].get("label"),
                  "predicted_raw_pct": tiers[name].get("predicted_worst_qp_penalty_pct"),
                  "target_size_mb": tiers[name].get("target_size_mb"),
                  "started": now()}

        before_dir = tier_dir / "before"
        if args.force or not (before_dir / f"configs_{args.datasets[0]}.csv").is_file():
            log({"tier": name, "events": "measure_before"})
            record["before_status"] = evaluate(args.model, assignment_path, args.datasets,
                                               before_dir, args.max_images, log_path)
        record["before"] = summarise(before_dir, args.datasets)

        bundle = out / "bundles" / name
        train_json = out / f"{name}_train.json"
        if args.force or not train_json.is_file():
            log({"tier": name, "events": "qat", "steps": args.steps})
            code = run([PY, "quantization-pluse-mse/scripts/qat_run.py",
                        "--model", args.model, "--tier", args.tiers,
                        "--tier-name", name, "--pool", args.pool,
                        "--output", str(out), "--steps", str(args.steps),
                        "--lr", str(args.lr), "--accumulate", str(args.accumulate),
                        "--crop", str(args.crop), "--scope", args.scope,
                        "--rd-weight", str(args.rd_weight),
                        "--rate-weight", str(args.rate_weight),
                        "--objective", args.objective,
                        "--eval-every", str(args.eval_every),
                        "--eval-images", str(args.eval_images),
                        "--device", args.device], log_path)
            record["qat_status"] = code
        if train_json.is_file():
            record["training"] = json.loads(train_json.read_text(encoding="utf-8"))

        after_dir = tier_dir / "after"
        if bundle.is_dir() and (args.force or not (after_dir / f"configs_{args.datasets[0]}.csv").is_file()):
            log({"tier": name, "events": "measure_after"})
            record["after_status"] = evaluate(str(bundle), assignment_path, args.datasets,
                                              after_dir, args.max_images, log_path)
        record["after"] = summarise(after_dir, args.datasets)

        before = record["before"].get("worst_qp_penalty_pct")
        after = record["after"].get("worst_qp_penalty_pct")
        if before is not None and after is not None:
            record["recovery_pct_points"] = round(before - after, 3)
            record["recovery_ratio"] = round((before - after) / before, 3) if before else None
        record["finished"] = now()
        results.append(record)
        log({k: record.get(k) for k in ("tier", "before", "after", "recovery_pct_points")})
        (tier_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tier", "label", "size_mb", "before_pct", "after_pct",
                         "recovery_pp", "recovery_ratio", "steps", "train_minutes"])
        for r in results:
            size = None
            for value in (r.get("after", {}), r.get("before", {})):
                for entry in value.get("per_dataset", {}).values():
                    if entry.get("size_mb"):
                        size = entry["size_mb"]
                        break
                if size:
                    break
            training = r.get("training") or {}
            writer.writerow([r["tier"], r.get("label"), size,
                             r.get("before", {}).get("worst_qp_penalty_pct"),
                             r.get("after", {}).get("worst_qp_penalty_pct"),
                             r.get("recovery_pct_points"), r.get("recovery_ratio"),
                             training.get("steps"), training.get("train_minutes")])
    release_awake()
    log({"events": "done", "output": str(out), "tiers": len(results), "kept_awake": awake})


if __name__ == "__main__":
    main()
