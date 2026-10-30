"""Mixed-precision selection by MILP, with a greedy baseline for comparison.

Formulation. PULSE ships one weight file serving all eight QPs, so the assignment cannot differ
per QP: the variables are `x[c, f]`, one precision per component, and the quality constraint is
enforced separately at every QP. That makes the binding constraint the worst-QP total penalty,
which is the physically meaningful budget.

    minimise    sum size[c, f] * x[c, f]
    subject to  sum_f x[c, f] = 1                       (one precision per component)
                sum_{c,f} penalty[c, f, q] * x[c, f] <= eps   for every QP q

Solved with HiGHS through `scipy.optimize.milp`; no extra dependency. Penalties are the
worst-QP mean rate penalty measured by the single-component sweep, which
`scripts/analyze_additivity.py` shows predicts the whole-model penalty within about 5%.

    python quantization-pluse-mse/scripts/solve_ilp.py --budget 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[2]

FORMATS = ["fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "int8", "int4", "fp4_e2m1"]
OPTIONS = ["fp32"] + FORMATS
PRETTY = {
    "fp32": "FP32", "fp16": "FP16", "bf16": "BF16", "fp8_e4m3": "FP8-E4M3",
    "fp8_e5m2": "FP8-E5M2", "int8": "INT8", "int4": "INT4", "fp4_e2m1": "FP4-E2M1",
}


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_problem(root: Path, datasets: list[str]) -> tuple:
    inventory = json.loads((root / "artifacts" / "component_inventory.json").read_text(encoding="utf-8"))
    components = sorted(inventory.get("main_sweep", []))

    size, fp32 = {}, {}
    penalty = defaultdict(lambda: defaultdict(dict))
    for fmt in FORMATS:
        configs = {row["component"]: row for row in read_csv(root / fmt / "results" / f"configs_{fmt}.csv")}
        for component, row in configs.items():
            if component not in components:
                continue
            size[(component, fmt)] = number(row.get("quant_bytes"))
            if number(row.get("fp32_bytes")):
                fp32[component] = number(row.get("fp32_bytes"))
        for row in read_csv(root / fmt / "results" / "summary.csv"):
            if row.get("control") != "none" or row.get("dataset") not in datasets:
                continue
            value = number(row.get("mean_penalty_pct"))
            if value is None or row["component"] not in components:
                continue
            qp = int(row["qp"])
            previous = penalty[(row["component"], fmt)][qp].get("value")
            penalty[(row["component"], fmt)][qp]["value"] = value if previous is None else max(previous, value)

    for component in components:
        size[(component, "fp32")] = fp32[component]
    return components, size, penalty


def coefficient(size, penalty, component, fmt, qp):
    value = size.get((component, fmt))
    if value is None:
        return None, None
    if fmt == "fp32":
        return value, 0.0
    entry = penalty.get((component, fmt), {}).get(qp, {}).get("value")
    if entry is None:
        return None, None
    return value, max(entry, 0.0)


def solve(components, size, penalty, budget: float, qps: list[int]):
    index = [(component, fmt) for component in components for fmt in OPTIONS]
    position = {key: offset for offset, key in enumerate(index)}
    cost = np.zeros(len(index))
    quality = np.zeros((len(qps), len(index)))
    feasible = np.zeros(len(index), dtype=bool)

    for offset, (component, fmt) in enumerate(index):
        sizes = [coefficient(size, penalty, component, fmt, qp)[0] for qp in qps]
        if sizes[0] is None:
            continue
        cost[offset] = sizes[0]
        feasible[offset] = True
        for row, qp in enumerate(qps):
            _, value = coefficient(size, penalty, component, fmt, qp)
            if value is None:
                feasible[offset] = False
                break
            quality[row, offset] = value

    rows, lower, upper = [], [], []
    for component in components:
        row = np.zeros(len(index))
        for fmt in OPTIONS:
            key = (component, fmt)
            if key in position and feasible[position[key]]:
                row[position[key]] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)
    for row in range(len(qps)):
        coefficients = np.where(feasible, quality[row], 0.0)
        rows.append(coefficients)
        lower.append(-np.inf)
        upper.append(budget)

    matrix = np.vstack(rows)
    result = milp(
        c=cost,
        constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
        integrality=np.ones(len(index)),
        bounds=Bounds(np.zeros(len(index)), np.ones(len(index))),
    )
    if result.x is None:
        return None
    chosen = {}
    for offset, (component, fmt) in enumerate(index):
        if result.x[offset] > 0.5:
            chosen[component] = fmt
    return {
        "status": int(result.status),
        "message": str(result.message),
        "objective_bytes": float(result.fun),
        "assignment": chosen,
    }


def evaluate_assignment(components, size, penalty, assignment, qps):
    per_qp = {}
    for qp in qps:
        total = 0.0
        for component in components:
            fmt = assignment.get(component, "fp32")
            _, value = coefficient(size, penalty, component, fmt, qp)
            total += value or 0.0
        per_qp[qp] = total
    total_bytes = sum(size.get((c, assignment.get(c, "fp32"))) or 0.0 for c in components)
    fp32_bytes = sum(size.get((c, "fp32")) or 0.0 for c in components)
    return {
        "size_mb": round(total_bytes / 1e6, 3),
        "fp32_mb": round(fp32_bytes / 1e6, 3),
        "saving_pct": round(100.0 * (1 - total_bytes / fp32_bytes), 2),
        "worst_qp_penalty_pct": round(max(per_qp.values()), 4),
        "per_qp_penalty_pct": {str(k): round(v, 4) for k, v in per_qp.items()},
    }


def greedy(components, size, penalty, budget: float, qps: list[int]):
    assignment = {component: "fp32" for component in components}
    current = {qp: 0.0 for qp in qps}

    for _ in range(len(components) * len(OPTIONS)):
        best = None
        for component in components:
            current_size = size.get((component, assignment[component])) or 0.0
            for fmt in FORMATS:
                candidate_size = size.get((component, fmt))
                if candidate_size is None or candidate_size >= current_size - 1e-9:
                    continue
                deltas, values = {}, {}
                ok = True
                for qp in qps:
                    _, value = coefficient(size, penalty, component, fmt, qp)
                    if value is None:
                        ok = False
                        break
                    previous = coefficient(size, penalty, component, assignment[component], qp)[1] or 0.0
                    deltas[qp] = value - previous
                    values[qp] = value
                if not ok:
                    continue
                if any(current[qp] + deltas[qp] > budget for qp in qps):
                    continue
                saved = (current_size - candidate_size) / 1e6
                added = max(deltas.values())
                efficiency = float("inf") if added <= 0 else saved / added
                if best is None or efficiency > best[0]:
                    best = (efficiency, component, fmt, deltas, values)
        if best is None:
            break
        _, component, fmt, deltas, values = best
        assignment[component] = fmt
        for qp in qps:
            current[qp] += deltas[qp]
    return assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="quantization-pluse-mse")
    parser.add_argument("--output", default="quantization-pluse-mse/results/ilp")
    parser.add_argument("--dataset", nargs="+", default=["kodak", "clic_professional"])
    parser.add_argument("--qps", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--budget", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--headline", type=float, default=1.0)
    args = parser.parse_args()

    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    components, size, penalty = load_problem(root, args.dataset)

    rows, payloads = [], {}
    for budget in args.budget:
        solved = solve(components, size, penalty, budget, args.qps)
        if solved is None:
            print(json.dumps({"budget": budget, "status": "infeasible"}))
            continue
        ilp_eval = evaluate_assignment(components, size, penalty, solved["assignment"], args.qps)
        greedy_assignment = greedy(components, size, penalty, budget, args.qps)
        greedy_eval = evaluate_assignment(components, size, penalty, greedy_assignment, args.qps)
        rows.append({
            "budget_pct": budget,
            "ilp_size_mb": ilp_eval["size_mb"],
            "ilp_saving_pct": ilp_eval["saving_pct"],
            "ilp_worst_qp_penalty_pct": ilp_eval["worst_qp_penalty_pct"],
            "greedy_size_mb": greedy_eval["size_mb"],
            "greedy_saving_pct": greedy_eval["saving_pct"],
            "greedy_worst_qp_penalty_pct": greedy_eval["worst_qp_penalty_pct"],
            "ilp_beats_greedy_mb": round(greedy_eval["size_mb"] - ilp_eval["size_mb"], 3),
            "components_quantized": sum(1 for c in components if solved["assignment"][c] != "fp32"),
            "status": solved["status"],
        })
        payloads[budget] = {"ilp": solved, "ilp_eval": ilp_eval, "greedy": greedy_assignment, "greedy_eval": greedy_eval}
        print(json.dumps(rows[-1]))

    write_csv(output / "solve.csv", rows)

    headline = payloads.get(args.headline)
    if headline:
        assignment = {c: f for c, f in headline["ilp"]["assignment"].items() if f != "fp32"}
        payload = {
            "name": "ilp",
            "note": f"MILP optimum at a {args.headline}% worst-QP penalty budget (HiGHS via scipy.optimize.milp)",
            "budget_pct": args.headline,
            "assignment": dict(sorted(assignment.items())),
            "predicted": headline["ilp_eval"],
            "greedy_same_budget": headline["greedy_eval"],
            "datasets": args.dataset,
        }
        (output / "assignment_ilp.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        detail = []
        for component in components:
            fmt = headline["ilp"]["assignment"][component]
            detail.append({
                "component": component,
                "format": fmt,
                "size_mb": round((size.get((component, fmt)) or 0.0) / 1e6, 4),
                "penalty_qp0_pct": round(coefficient(size, penalty, component, fmt, 0)[1] or 0.0, 4),
                "penalty_qp7_pct": round(coefficient(size, penalty, component, fmt, 7)[1] or 0.0, 4),
            })
        write_csv(output / "assignment_ilp_detail.csv", detail)
        print(json.dumps({"headline_budget": args.headline, **headline["ilp_eval"]}))


if __name__ == "__main__":
    main()

