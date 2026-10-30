"""Per-QP breakdown for every measured configuration, as tables and CSV pivots.

    python quantization-pluse-mse/scripts/report_per_qp.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

CONFIGS = ["fp32", "fp16", "bf16", "int8", "safe", "greedy", "ilp"]
PRETTY = {
    "fp32": "FP32 base", "fp16": "FP16 uniform", "bf16": "BF16 uniform",
    "int8": "INT8 uniform", "safe": "safe mixed", "greedy": "greedy mixed", "ilp": "ILP mixed",
}
METRICS = [
    ("mean_bpp", "bitrate (bpp)", "{:.4f}", 1),
    ("mean_psnr", "PSNR (dB)", "{:.4f}", 1),
    ("mean_d_psnr", "d_PSNR (dB)", "{:+.4f}", -1),
    ("worst_d_psnr", "worst-image d_PSNR (dB)", "{:+.4f}", -1),
    ("mean_d_bpp_pct", "d_bpp (%)", "{:+.3f}", -1),
    ("mean_penalty_pct", "penalty (%)", "{:+.3f}", -1),
    ("max_penalty_pct", "worst-image penalty (%)", "{:+.3f}", -1),
]


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed", default="quantization-pluse-mse/results/mixed")
    parser.add_argument("--dataset", nargs="+", default=["kodak", "clic_professional", "set14"])
    parser.add_argument("--output", default="quantization-pluse-mse/results/mixed")
    args = parser.parse_args()

    mixed, output = Path(args.mixed), Path(args.output)
    lines = ["# Per-QP breakdown, all configurations", "",
             "Measured through the real codec, all images per dataset. `penalty` is the"
             " rate-equivalent cost; `worst-image` is the single worst image at that QP.", ""]

    for dataset in args.dataset:
        summary = read_csv(mixed / f"summary_{dataset}.csv")
        configs_meta = {r["config"]: r for r in read_csv(mixed / f"configs_{dataset}.csv")}
        if not summary:
            continue
        table = defaultdict(dict)
        for row in summary:
            table[(row["config"], int(row["qp"]))][row["field"] if "field" in row else ""] = row
        values = {(r["config"], int(r["qp"])): r for r in summary}

        lines.append(f"## {dataset}")
        lines.append("")
        if dataset in configs_meta:
            lines.append("| config | weights MB | saving % |")
            lines.append("|---|---:|---:|")
            for config in CONFIGS:
                meta = configs_meta.get(config)
                if meta:
                    lines.append(f"| {PRETTY[config]} | {meta['quant_mb']} | {meta['saving_pct']} |")
            lines.append("")

        for key, label, fmt, better in METRICS:
            rows_out = []
            header = "| config | " + " | ".join(f"QP{q}" for q in range(8)) + " | worst |"
            sep = "|---|" + "---:|" * 9
            lines.append(f"**{label}**")
            lines.append("")
            lines.append(header)
            lines.append(sep)
            for config in CONFIGS:
                cells, numbers = [], []
                for qp in range(8):
                    row = values.get((config, qp))
                    if row is None or row.get(key) in ("", None):
                        cells.append("-")
                        continue
                    value = float(row[key])
                    numbers.append(value)
                    cells.append(fmt.format(value))
                if numbers:
                    extreme = min(numbers) if better < 0 else max(numbers)
                    cells.append(fmt.format(extreme))
                else:
                    cells.append("-")
                lines.append(f"| {PRETTY[config]} | " + " | ".join(cells) + " |")
                rows_out.append({"config": config,
                                 **{f"qp{q}": (values.get((config, q), {}).get(key, "")) for q in range(8)}})
            lines.append("")
            write_csv(output / f"per_qp_{key}_{dataset}.csv", rows_out)

    (output / "per_qp_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output / 'per_qp_tables.md'}")


if __name__ == "__main__":
    main()
