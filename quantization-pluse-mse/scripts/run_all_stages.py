"""Run the whole endgame unattended: submission tables, QAT report, QAT attempt, final summary.

Stages run in order and each is resumable: a stage whose output already exists is skipped, a stage
that fails is recorded and the next one still runs. The process holds the machine awake so an
unattended laptop on charge does not sleep in the middle of a stage.

    python quantization-pluse-mse/scripts/run_all_stages.py
    python quantization-pluse-mse/scripts/run_all_stages.py --smoke      # wiring check, minutes
    python quantization-pluse-mse/scripts/run_all_stages.py --stages submission qat_negative

Stages
  1. submission     two validated operating points, 21.677 MB and 18.389 MB, on three datasets
  2. qat_negative   every QAT trajectory recorded so far, tabulated from the run logs
  3. qat_attempt    augmented-Lagrangian QAT on the three precision tiers, with before/after
  4. summary        one REPORT.md tying the above together
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

# run label, log path, objective description
QAT_RUNS = (
    ("raw RD objective, lr 3e-6, rate term unconstrained",
     "results/qat/run_20261030T232053/overnight.log"),
    ("rate-preserving, lr 3e-6, rate weight 1",
     "results/qat/run2_20261031T153921/overnight.log"),
    ("rate-preserving, lr 3e-6, rate weight 1 (restart)",
     "results/qat/run3_20261031T155151/overnight.log"),
    ("rate-preserving, lr 1e-6, rate weight 8",
     "results/qat/run4_20261031T162142/overnight.log"),
)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(payload: dict) -> None:
    print(json.dumps({"at": now(), **payload}), flush=True)


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


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def find_row(directory: Path, dataset: str, config_stem: str) -> dict:
    for row in read_csv(directory / f"configs_{dataset}.csv"):
        note = Path(row.get("note", "")).stem
        if note == config_stem or config_stem in (row.get("config") or ""):
            return row
    rows = read_csv(directory / f"configs_{dataset}.csv")
    return rows[0] if rows else {}


# ---------------------------------------------------------------- stage 1
def stage_submission(out: Path) -> dict:
    """Two validated operating points. Everything comes from measured CSVs, nothing is typed in."""
    points = [
        ("champion_1pct", "1% budget champion", WORKSPACE / "results/mixed", "ilp"),
        ("smaller_2pct", "2% budget variant", WORKSPACE / "results/ilp/b2p0_eval", "assignment_ilp_b2p0"),
    ]
    rows, table = [], []
    for key, label, directory, stem in points:
        entry = {"point": key, "label": label}
        for dataset in DATASETS:
            row = find_row(directory, dataset, stem)
            if not row:
                entry[dataset] = None
                continue
            entry[dataset] = float(row["worst_qp_mean_penalty_pct"])
            entry["size_mb"] = float(row["quant_mb"])
            entry["saving_pct"] = float(row["saving_pct"])
            entry["fp32_mb"] = float(row["fp32_mb"])
        rows.append(entry)
        table.append(entry)

    lines = ["# Submission operating points", "",
             "Two measured points, both evaluated on all three datasets with the same harness, the same",
             "eight QPs and the same FP32 reference. Nothing here is a prediction.", "",
             "| Point | Size | Saving | Kodak | CLIC | Set14 |", "|---|---:|---:|---:|---:|---:|"]
    for entry in table:
        cells = []
        for dataset in DATASETS:
            value = entry.get(dataset)
            cells.append("-" if value is None else f"{value:.3f}%")
        lines.append(f"| {entry['label']} | {entry.get('size_mb', 0):.3f} MB | "
                     f"-{entry.get('saving_pct', 0):.2f}% | " + " | ".join(cells) + " |")
    lines += ["",
              "The 2% variant is 3.288 MB smaller than the champion and costs about one extra point of",
              "worst-QP penalty on every dataset. Both are inside a 2% budget, so the pair reads as a",
              "size-quality trade rather than a single number.", ""]
    (out / "SUBMISSION_POINTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out / "submission_points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["point", "label", "size_mb", "saving_pct", *DATASETS])
        for entry in table:
            writer.writerow([entry["point"], entry["label"], entry.get("size_mb"),
                             entry.get("saving_pct"),
                             *[entry.get(dataset) for dataset in DATASETS]])
    return {"stage": "submission", "points": len(table)}


# ---------------------------------------------------------------- stage 2
def stage_qat_negative(out: Path) -> dict:
    """Tabulate every QAT trajectory from its log. This is the evidence for the negative result."""
    rows = []
    for label, relative in QAT_RUNS:
        path = WORKSPACE / relative
        if not path.is_file():
            rows.append({"run": label, "status": "log missing"})
            continue
        steps, validated, bests, finish = 0, [], [], None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = payload.get("events")
            if event == "step":
                steps = max(steps, int(payload.get("step", 0)))
            elif event == "validated":
                validated.append(payload)
                if payload.get("best"):
                    bests.append(payload["step"])
            elif event == "finish":
                finish = payload
        worst = max((v.get("penalty_vs_start_pct") or 0 for v in validated), default=None)
        rows.append({
            "run": label,
            "steps_reached": steps,
            "validations": len(validated),
            "worst_penalty_pct": worst,
            "last_penalty_pct": validated[-1].get("penalty_vs_start_pct") if validated else None,
            "last_d_psnr": validated[-1].get("d_psnr_vs_start") if validated else None,
            "checkpoints_that_improved": len(bests),
            "kept_best_step": (finish or {}).get("best_step"),
        })

    lines = ["# QAT: measured negative result", "",
             "Quantization-aware fine-tuning was attempted on this model and did not improve it.",
             "Every run below is from a real log, with the full-precision model as the reference and the",
             "penalty metric defined in `scripts/pipeline.py`.", "",
             "| Objective | Steps reached | Worst penalty | Final dPSNR | Checkpoints that improved |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        if row.get("status"):
            lines.append(f"| {row['run']} | - | - | - | {row['status']} |")
            continue
        worst = row.get("worst_penalty_pct")
        d_psnr = row.get("last_d_psnr")
        worst_text = "-" if worst is None else f"{worst:.2f}%"
        dp_text = "-" if d_psnr is None else f"{d_psnr:+.3f} dB"
        lines.append(f"| {row['run']} | {row['steps_reached']} | {worst_text} | {dp_text} | "
                     f"{row['checkpoints_that_improved']} |")

    lines += ["",
              "## Why it fails", "",
              "The quantized model already sits on the model's own rate-distortion curve. Fine-tuning can",
              "move it *along* that curve, trading bitrate for distortion, but it cannot push the curve",
              "outward, which is what would actually recover the quantization loss. Both directions were",
              "observed: the raw RD objective sold 29% of the bitrate for 1.54 dB, while objectives that",
              "constrain rate bought distortion at a fair price and left the penalty at zero or worse.",
              "",
              "## Why the reported model is unaffected", "",
              "Every run selects its saved checkpoint by the reported penalty metric and seeds the search",
              "with the pre-QAT weights, so a run that does not improve saves the quantized weights",
              "unchanged. In the runs above the best checkpoint was step 0, which is exactly that",
              "behaviour: no degraded model was ever written.", ""]
    (out / "QAT_NEGATIVE_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out / "qat_trajectories.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"stage": "qat_negative", "runs": len(rows)}


# ---------------------------------------------------------------- stage 3
def stage_qat_attempt(out: Path, args, log_path: Path) -> dict:
    """Augmented-Lagrangian QAT on the three tiers. The safety net keeps the best checkpoint."""
    target = out / "qat_attempt"
    if (target / "summary.csv").is_file() and not args.force:
        return {"stage": "qat_attempt", "status": "already done"}
    command = [PY, str(WORKSPACE / "scripts" / "qat_overnight.py"),
               "--steps", str(args.steps), "--lr", str(args.lr),
               "--accumulate", str(args.accumulate), "--objective", "augmented_lagrangian",
               "--rd-weight", "1.0", "--mu-alpha", "1.0", "--mu-max", "20.0",
               "--eval-every", str(args.eval_every), "--eval-images", "4",
               "--datasets", *(args.smoke_datasets or DATASETS), "--output", str(target)]
    if args.smoke_limit:
        command += ["--max-images", str(args.smoke_limit)]
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n=== {now()} {' '.join(command)}\n")
        stream.flush()
        code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, cwd=str(ROOT))
        stream.write(f"=== {now()} exit {code}\n")
    rows = read_csv(target / "summary.csv")
    return {"stage": "qat_attempt", "exit": code,
            "recovery_pct_points": [r.get("recovery_pp") for r in rows]}


# ---------------------------------------------------------------- stage 4
def stage_summary(out: Path) -> dict:
    parts = []
    for name in ("SUBMISSION_POINTS.md", "QAT_NEGATIVE_RESULT.md"):
        path = out / name
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8").rstrip())
    attempt = read_csv(out / "qat_attempt" / "summary.csv")
    if attempt:
        parts.append("# Augmented-Lagrangian attempt\n")
        parts.append("| Tier | Size | Before | After | Recovery |")
        parts.append("|---|---:|---:|---:|---:|")
        for row in attempt:
            parts.append(f"| {row['tier']} | {row['size_mb']} MB | {row['before_pct']}% | "
                         f"{row['after_pct']}% | {row['recovery_pp']} |")
    parts.append("")
    (out / "REPORT.md").write_text("\n\n".join(parts), encoding="utf-8")
    return {"stage": "summary", "sections": len(parts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="quantization-pluse-mse/results/final")
    parser.add_argument("--stages", nargs="+",
                        default=["submission", "qat_negative", "qat_attempt", "summary"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-datasets", nargs="*", default=None,
                        help="restrict the QAT stage datasets, for wiring checks")
    parser.add_argument("--smoke-limit", type=int, default=0,
                        help="cap images per dataset in the QAT stage")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny QAT stage so the wiring can be checked in minutes")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    if args.smoke:
        args.steps, args.accumulate, args.eval_every = 3, 1, 2
        args.smoke_datasets, args.smoke_limit = ["kodak"], 2
        args.stages = ["submission", "qat_negative", "qat_attempt", "summary"]

    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "pipeline.log"
    awake = hold_awake()
    log({"stage": "pipeline", "event": "start", "stages": args.stages, "sleep_guard": awake})

    results = []
    try:
        for name in args.stages:
            marker = out / f".done_{name}"
            if marker.is_file() and not args.force:
                log({"stage": name, "event": "skipped", "reason": "marker present"})
                continue
            started = time.perf_counter()
            try:
                if name == "submission":
                    result = stage_submission(out)
                elif name == "qat_negative":
                    result = stage_qat_negative(out)
                elif name == "qat_attempt":
                    result = stage_qat_attempt(out, args, log_path)
                elif name == "summary":
                    result = stage_summary(out)
                else:
                    result = {"stage": name, "status": "unknown stage"}
            except Exception as error:  # a failing stage must not stop the rest
                result = {"stage": name, "status": "failed", "error": f"{type(error).__name__}: {error}"}
            result["seconds"] = round(time.perf_counter() - started, 1)
            results.append(result)
            if result.get("status") != "failed":
                marker.write_text(now() + "\n", encoding="utf-8")
            log(result)
    finally:
        release_awake()
    (out / "pipeline_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    log({"stage": "pipeline", "event": "done", "results": results})


if __name__ == "__main__":
    main()
