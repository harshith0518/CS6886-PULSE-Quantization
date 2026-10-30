"""One entry point for inventory, reference, sweep, and compact export."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quantization-pluse-mse" / "src"))

import torch
from PIL import Image

from codec.checkpoint import load_model
from codec.images import image_paths, pad_image, read_image
from pulsequant import plots
from pulsequant.core import (
    COMPONENTS, MAIN_SWEEP, apply_quant, assert_handle, component_size,
    conv_modules, side, weight_stats,
)
from pulsequant.eval import (
    append_jsonl, build_manifest, configure_eval, decode_fixed, delta_metrics,
    evaluate_image, output_delta, sha256_file, sha256_json, slopes_from_curve,
    write_json,
)

FROZEN = {
    "kodak_thumb": "benchmark_results/pulse_mse_20261011_043023.json",
    "clic_professional": "benchmark_results/pulse_mse_20261011_045724.json",
}
REF_FIELDS = [
    "dataset", "image", "qp", "bytes", "pixels", "bpp", "mse", "psnr",
    "encode_ms", "decode_ms", "stream_hash", "recon_hash", "decode_ok",
    "image_height", "image_width", "tf32", "run_pass",
]
SWEEP_FIELDS = [
    "sweep_id", "config_id", "dataset", "image", "h", "w", "qp",
    "bytes", "pixels", "bpp", "mse", "psnr", "stream_hash", "recon_hash",
    "ref_bpp", "ref_psnr", "ref_stream_hash", "ref_recon_hash",
    "decode_ok", "status", "error",
]
CONFIG_FIELDS = [
    "config_id", "sweep_id", "component", "format", "mode", "granularity",
    "qmax", "control", "orig_weight_sha256", "quant_weight_sha256",
    "restored_weight_sha256", "restored_ok", "output_delta", "fp32_bytes",
    "quant_bytes", "size_reduction_pct", "effective_bits_per_param",
    "base_model_hash", "quant_config_hash",
]
WEIGHT_FIELDS = [
    "config_id", "layer", "sqnr_db", "clip_frac", "max_abs_err",
    "fan_in", "effective_bits",
]


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def summarize(rows: list[dict]) -> dict:
    psnr = [float(row["psnr"]) for row in rows]
    bpp = [float(row["bpp"]) for row in rows]
    d_psnr = [float(row["d_psnr"]) for row in rows]
    d_bpp = [float(row["d_bpp_pct"]) for row in rows]
    penalty = [float(row["penalty_pct"]) for row in rows if row["penalty_pct"] not in ("", "None", None)]
    return {
        "rows": len(rows),
        "failures": sum(1 for row in rows if row["status"] != "ok"),
        "mean_psnr": statistics.mean(psnr),
        "mean_bpp": statistics.mean(bpp),
        "mean_d_psnr": statistics.mean(d_psnr),
        "worst_d_psnr": min(d_psnr),
        "mean_d_bpp_pct": statistics.mean(d_bpp),
        "mean_penalty_pct": statistics.mean(penalty) if penalty else None,
        "max_penalty_pct": max(penalty) if penalty else None,
        "p90_penalty_pct": percentile(penalty, 0.9) if penalty else None,
        "recon_match_rate": statistics.mean(1.0 if row["recon_match_ref"] else 0.0 for row in rows),
        "stream_match_rate": statistics.mean(1.0 if row["stream_match_ref"] else 0.0 for row in rows),
    }


def weight_bits(fmt: str, fan_in: float) -> float:
    if fmt in {"fp16", "bf16"}:
        return 16.0
    if fmt in {"fp8_e4m3", "fp8_e5m2"}:
        return 8.0 + 32.0 / fan_in
    if fmt == "int8":
        return 8.0 + 32.0 / fan_in
    if fmt == "int4":
        return 4.0 + 32.0 / fan_in
    if fmt == "fp4_e2m1":
        return 4.0 + 32.0 / fan_in
    return 32.0


def size_table(inventory: dict, fmt: str) -> list[dict]:
    rows = []
    for component, spec in inventory.get("components", {}).items():
        conv_numel = 0.0
        quant_bits = 0.0
        for row in inventory.get("conv_rows", []):
            if row.get("component") != component:
                continue
            numel = float(row["out_channels"] * row["fan_in"])
            conv_numel += numel
            quant_bits += numel * weight_bits(fmt, float(row["fan_in"]))
        total = float(spec.get("params", 0))
        non_conv = max(0.0, total - conv_numel)
        fp32_bytes = total * 4.0
        quant_bytes = quant_bits / 8.0 + non_conv * 4.0
        rows.append({
            "component": component,
            "name": spec.get("name"),
            "side": spec.get("side"),
            "total_params": int(total),
            "conv_weight_params": int(conv_numel),
            "fp32_bytes": int(round(fp32_bytes)),
            "quant_bytes": int(round(quant_bytes)),
            "size_reduction_pct": 100.0 * (1.0 - quant_bytes / fp32_bytes) if fp32_bytes else 0.0,
            "effective_bits_per_param": quant_bytes * 8.0 / total if total else 0.0,
        })
    return rows


def load_baseline(reference_dir: Path) -> dict:
    baseline = {}
    for dataset_dir in reference_dir.glob("*"):
        if not dataset_dir.is_dir():
            continue
        by_qp = defaultdict(list)
        for row in read_csv(dataset_dir / "pass1.csv"):
            by_qp[int(row["qp"])].append(row)
        for qp, rows in by_qp.items():
            baseline[(dataset_dir.name, qp)] = {
                "dataset": dataset_dir.name,
                "qp": qp,
                "ref_rows": len(rows),
                "ref_bpp": statistics.mean(float(row["bpp"]) for row in rows),
                "ref_psnr": statistics.mean(float(row["psnr"]) for row in rows),
                "ref_encode_ms": statistics.mean(float(row["encode_ms"]) for row in rows),
                "ref_decode_ms": statistics.mean(float(row["decode_ms"]) for row in rows),
                "ref_total_ms": statistics.mean(float(row["encode_ms"]) + float(row["decode_ms"]) for row in rows),
            }
    return baseline


def run_inventory(args):
    torch.manual_seed(0)
    model = load_model(args.model, "cpu", entropy=False).eval()
    image = torch.rand(1, 3, args.height, args.width)
    shapes = {}

    def hook(name):
        def capture(_module, _inputs, output):
            shapes[name] = tuple(output.shape)
        return capture

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(hook(name)))
    with torch.no_grad():
        y_hat, _ = model.encode(image, qp=args.qp)
        model.decode(y_hat, qp=args.qp)
    for handle in handles:
        handle.remove()

    pixels = args.height * args.width
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        shape = shapes.get(name)
        fan_in = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        component = next(
            (key for key, spec in COMPONENTS.items()
             if any(name == prefix or name.startswith(prefix + ".") for prefix in spec["prefixes"])),
            "UNMAPPED",
        )
        macs = shape[-2] * shape[-1] * module.out_channels * fan_in if shape else None
        rows.append({
            "name": name, "component": component, "in_channels": module.in_channels,
            "out_channels": module.out_channels, "kernel": list(module.kernel_size),
            "stride": list(module.stride), "padding": list(module.padding),
            "groups": module.groups, "fan_in": fan_in,
            "output_shape": list(shape) if shape else None,
            "macs": macs, "kmac_per_px": macs / pixels / 1000.0 if macs else None,
        })

    by_component = defaultdict(float)
    for row in rows:
        if row["kmac_per_px"] is not None:
            by_component[row["component"]] += row["kmac_per_px"]

    groups = {
        "analysis": ["E0", "E1", "E2", "E3", "E4", "E5"],
        "hyper_enc": ["H0"], "hyper_dec": ["H1"], "em1": ["H2"], "em2": ["H3"],
        "linear_cdf": ["H4"],
        "synthesis": ["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"],
    }
    expected = {
        "analysis": 43.22, "hyper_enc": 0.45, "hyper_dec": 0.24,
        "em1": 0.58, "em2": 0.74, "linear_cdf": 0.12, "synthesis": 3.53,
    }
    table4 = {}
    for group, components in groups.items():
        measured = sum(by_component.get(component, 0.0) for component in components)
        tolerance = max(0.005, 0.01 * expected[group])
        table4[group] = {
            "expected": expected[group], "measured": measured,
            "delta": measured - expected[group],
            "pct": 100.0 * (measured / expected[group] - 1.0),
            "tolerance": tolerance,
            "pass": abs(measured - expected[group]) <= tolerance,
        }

    kodak_sizes = {}
    for path in image_paths(args.kodak):
        with Image.open(path) as source:
            kodak_sizes[path.name] = list(source.size)

    payload = {
        "model": str(Path(args.model)),
        "input": {"height": args.height, "width": args.width, "qp": args.qp},
        "pixels": pixels,
        "components": {
            component: {
                **spec,
                "params": sum(
                    parameter.numel() for name, parameter in model.named_parameters()
                    if any(name == prefix or name.startswith(prefix + ".") for prefix in spec["prefixes"])
                ),
                "kmac_per_px": by_component.get(component, 0.0),
            }
            for component, spec in COMPONENTS.items()
        },
        "main_sweep": MAIN_SWEEP,
        "table4_check": table4,
        "kodak_sizes": kodak_sizes,
        "conv_rows": sorted(rows, key=lambda row: row["name"]),
    }
    write_json(args.output, payload)
    print(json.dumps(table4, indent=2))
    print(json.dumps(kodak_sizes, indent=2))


def run_reference(args):
    input_path = Path(args.input or f"datasets/{args.dataset}")
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = image_paths(input_path)[: args.max_images] if args.max_images else image_paths(input_path)

    if args.dataset in {"kodak", "kodak_thumb"}:
        sizes = {}
        for path in paths:
            with Image.open(path) as source:
                sizes[path.name] = list(source.size)
        print(json.dumps({"kodak_sizes": sizes}, indent=2))

    model = load_model(args.model, args.device)
    passes = [("pass1", False, 1), ("pass2", False, 2), ("tf32_on", True, 3)]
    rows_by_pass = {}
    for name, tf32, run_pass in passes:
        flags = configure_eval(args.device, args.threads, tf32=tf32)
        rows = []
        for path in paths:
            image = read_image(path, args.device)
            for qp in args.qps:
                row = evaluate_image(model, image, qp, legacy=args.legacy)
                row.update({"dataset": args.dataset, "image": path.name, "qp": qp,
                             "tf32": tf32, "run_pass": run_pass})
                rows.append(row)
        rows_by_pass[name] = rows
        write_csv(output_dir / f"{name}.csv", rows)
        print(f"{name}: {len(rows)} rows")

    keyed = lambda rows: {(row["image"], int(row["qp"])): row for row in rows}
    k1, k2, k3 = keyed(rows_by_pass["pass1"]), keyed(rows_by_pass["pass2"]), keyed(rows_by_pass["tf32_on"])
    common = sorted(set(k1) & set(k2) & set(k3))

    def max_delta(left, right, field, percent=False):
        values = []
        for key in common:
            a, b = float(left[key][field]), float(right[key][field])
            values.append(abs(100.0 * (a / b - 1.0)) if percent else abs(a - b))
        return max(values) if values else None

    noise = {
        "pairs": len(common),
        "off_max_abs_d_psnr": max_delta(k1, k2, "psnr"),
        "off_max_abs_d_bpp_pct": max_delta(k1, k2, "bpp", True),
        "off_stream_hash_identical": all(k1[k]["stream_hash"] == k2[k]["stream_hash"] for k in common),
        "off_recon_hash_identical": all(k1[k]["recon_hash"] == k2[k]["recon_hash"] for k in common),
        "tf32_max_abs_d_psnr": max_delta(k1, k3, "psnr"),
        "tf32_max_abs_d_bpp_pct": max_delta(k1, k3, "bpp", True),
        "tf32_stream_hash_identical": all(k1[k]["stream_hash"] == k3[k]["stream_hash"] for k in common),
        "tf32_recon_hash_identical": all(k1[k]["recon_hash"] == k3[k]["recon_hash"] for k in common),
    }
    write_json(output_dir / "noise_floor.json", noise)

    by_qp = defaultdict(list)
    for row in rows_by_pass["pass1"]:
        by_qp[int(row["qp"])].append(row)
    curve = [
        {"qp": qp, "bpp": statistics.mean(float(r["bpp"]) for r in rows),
         "psnr": statistics.mean(float(r["psnr"]) for r in rows), "images": len(rows)}
        for qp, rows in sorted(by_qp.items())
    ]
    slopes = slopes_from_curve(curve)
    write_json(output_dir / "curve.json", {"curve": curve, "slopes": slopes})

    frozen_path = Path(args.frozen_json) if args.frozen_json else (
        Path(FROZEN[args.dataset]) if args.dataset in FROZEN else None
    )
    if frozen_path is not None and frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["summary"]
        g0a = {}
        for point in curve:
            ref = frozen[str(point["qp"])]
            g0a[str(point["qp"])] = {
                "bpp": point["bpp"], "psnr": point["psnr"],
                "frozen_bpp": ref["bpp"], "frozen_psnr": ref["psnr"],
                "d_psnr": point["psnr"] - ref["psnr"],
                "d_bpp_pct": 100.0 * (point["bpp"] / ref["bpp"] - 1.0),
            }
        write_json(output_dir / "g0a.json", g0a)
    write_json(output_dir / "manifest.json", build_manifest(
        model_path=args.model,
        config={"dataset": args.dataset, "input": str(input_path), "qps": args.qps,
                  "max_images": args.max_images, "legacy": args.legacy,
                  "passes": ["pass1", "pass2", "tf32_on"]},
        command=subprocess.list2cmdline(sys.argv),
    ))
    print(json.dumps({"noise_floor": noise, "curve": curve, "slopes": slopes}, indent=2))


def run_sweep(args):
    if args.format == "fp32":
        raise SystemExit("use the reference subcommand for FP32")
    components = MAIN_SWEEP if "all" in args.components else args.components
    unknown = [component for component in components if component not in COMPONENTS]
    if unknown:
        raise SystemExit(f"unknown components: {unknown}")

    reference_dir = Path(args.reference_dir) / args.dataset
    reference = {(row["image"], int(row["qp"])): row for row in read_csv(reference_dir / "pass1.csv")}
    if not reference:
        raise SystemExit(f"missing reference: {reference_dir / 'pass1.csv'}")

    output_path = Path(args.output)
    configs_path = Path(args.configs_output)
    weights_path = Path(args.weight_stats_output)
    if args.overwrite:
        for path in (output_path, configs_path, weights_path):
            if path.is_file():
                path.unlink()

    existing_sweep = read_csv(output_path)
    existing_configs = {row["config_id"]: row for row in read_csv(configs_path)}
    existing_weights = {(row["config_id"], row["layer"]): row for row in read_csv(weights_path)}
    done = {(row["config_id"], row["dataset"], row["image"], int(row["qp"])) for row in existing_sweep}

    paths = image_paths(Path(f"datasets/{args.dataset}"))
    if args.max_images:
        paths = paths[: args.max_images]
    model = load_model(args.model, args.device)
    flags = configure_eval(args.device, args.threads, tf32=args.tf32 == "on")
    sweep_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    command = subprocess.list2cmdline(sys.argv)
    model_sha = sha256_file(Path(args.model) / "model.pt")
    effective_format = "int4" if args.control == "int4" else args.format
    manifest = build_manifest(model_path=args.model, config={
        "format": effective_format, "components": components, "mode": args.mode,
        "control": args.control, "dataset": args.dataset, "qps": args.qps,
        "max_images": args.max_images, "legacy": args.legacy, "tf32": args.tf32,
    }, command=command)
    manifest["eval_flags"] = flags
    manifest_path = Path(args.manifests) / f"{sweep_id}.json"
    write_json(manifest_path, manifest)

    sweep_rows = []
    config_rows = []
    weight_rows = []
    index = 0
    consecutive_failures = 0

    def flush():
        write_csv(output_path, existing_sweep + sweep_rows)
        merged = dict(existing_configs)
        merged.update({row["config_id"]: row for row in config_rows})
        write_csv(configs_path, [merged[key] for key in sorted(merged)])
        merged = dict(existing_weights)
        merged.update({(row["config_id"], row["layer"]): row for row in weight_rows})
        write_csv(weights_path, [merged[key] for key in sorted(merged)])

    for component in components:
        comp_side = side(component)
        fmt = "int4" if args.control == "int4" else args.format
        qmax = 7 if fmt == "int4" else 127
        config = {"component": component, "format": fmt, "mode": args.mode,
                  "granularity": "per_output_channel_symmetric",
                  "control": args.control, "qmax": qmax}
        config_hash = sha256_json(config)
        config_id = config_hash[:12]
        size = component_size(model, component, fmt)
        config_row = {
            "config_id": config_id, "sweep_id": sweep_id, "component": component,
            "format": fmt, "mode": args.mode, "granularity": config["granularity"],
            "qmax": qmax, "control": args.control,
            "orig_weight_sha256": "", "quant_weight_sha256": "",
            "restored_weight_sha256": "", "restored_ok": "",
            "output_delta": "", **size,
            "base_model_hash": model_sha, "quant_config_hash": config_hash,
        }
        config_done = config_id in existing_configs
        weights_done = any(key[0] == config_id for key in existing_weights)

        abort_config = False
        for path in paths:
            image = read_image(path, args.device)
            padded = pad_image(image)
            for qp in args.qps:
                key = (config_id, args.dataset, path.name, qp)
                run_id = f"{sweep_id}:{index:06d}"
                index += 1
                if key in done:
                    print(f"skip {component} {fmt} {path.name} qp={qp}")
                    continue
                ref = reference.get((path.name, qp))
                if ref is None:
                    raise KeyError(f"missing reference for {path.name} qp={qp}")
                row = {
                    "sweep_id": sweep_id, "config_id": config_id,
                    "dataset": args.dataset, "image": path.name,
                    "h": "", "w": "", "qp": qp,
                    "bytes": "", "pixels": "", "bpp": "", "mse": "", "psnr": "",
                    "stream_hash": "", "recon_hash": "",
                    "ref_bpp": float(ref["bpp"]), "ref_psnr": float(ref["psnr"]),
                    "ref_stream_hash": ref["stream_hash"],
                    "ref_recon_hash": ref.get("recon_hash", ""),
                    "decode_ok": False, "status": "ok", "error": "",
                }
                handle = None
                try:
                    if comp_side == "decode":
                        with torch.no_grad():
                            y_hat, _ = model.encode(padded, qp=qp)
                        before = decode_fixed(model, y_hat, qp)
                        handle = apply_quant(model, component, fmt=fmt, qmax=qmax, control=args.control)
                        after = decode_fixed(model, y_hat, qp)
                        delta = output_delta(before, after)
                    else:
                        with torch.no_grad():
                            before, _ = model.encode(padded, qp=qp)
                        handle = apply_quant(model, component, fmt=fmt, qmax=qmax, control=args.control)
                        with torch.no_grad():
                            after, _ = model.encode(padded, qp=qp)
                        delta = output_delta(before, after)
                    metrics = evaluate_image(model, image, qp, legacy=args.legacy)
                    row.update({
                        "h": metrics["image_height"], "w": metrics["image_width"],
                        "bytes": metrics["bytes"], "pixels": metrics["pixels"],
                        "bpp": metrics["bpp"], "mse": metrics["mse"],
                        "psnr": metrics["psnr"], "stream_hash": metrics["stream_hash"],
                        "recon_hash": metrics["recon_hash"], "decode_ok": True,
                    })
                    if comp_side == "decode" and metrics["stream_hash"] != ref["stream_hash"]:
                        row.update({
                            "status": "invariant_failed",
                            "error": "decoder-only stream hash changed",
                        })
                    if not config_done:
                        config_row.update({
                            "orig_weight_sha256": handle.original_hash,
                            "quant_weight_sha256": handle.quantized_hash,
                            "output_delta": delta,
                        })
                        config_rows.append(config_row)
                        config_done = True
                    if not weights_done:
                        for stat in weight_stats(handle, fmt):
                            weight_rows.append({"config_id": config_id, **stat})
                        weights_done = True
                except Exception as error:
                    row.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
                finally:
                    if handle is not None:
                        handle.restore()
                        assert_handle(handle)
                        if config_rows and config_rows[-1]["config_id"] == config_id:
                            config_rows[-1].update({
                                "restored_weight_sha256": handle.restored_hash,
                                "restored_ok": handle.weights_restored,
                            })
                sweep_rows.append(row)
                append_jsonl(Path(args.progress), row)
                print(f"{component} {fmt} {args.control} {path.name} qp={qp} "
                      f"status={row['status']}")
                if row["status"] == "ok":
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                if row["status"] == "invariant_failed":
                    abort_config = True
                    break
            if abort_config:
                break
        flush()
        if consecutive_failures >= 3:
            print("stopping after 3 consecutive failures")
            break

    flush()
    print(json.dumps({"rows": len(sweep_rows), "output": str(output_path)}, indent=2))


def run_export(args):
    fmt = args.format
    results = Path(args.output_root) / fmt / "results"
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    configs_all = read_csv(Path(args.configs))
    configs = [
        row for row in configs_all
        if row["format"] == fmt or (fmt == "int4" and row["control"] == "int4")
    ]
    config_ids = {row["config_id"] for row in configs}
    sweep = [row for row in read_csv(Path(args.sweep)) if row["config_id"] in config_ids]
    weights = [row for row in read_csv(Path(args.weight_stats)) if row["config_id"] in config_ids]
    write_csv(results / f"sweep_{fmt}.csv", sweep)
    write_csv(results / f"configs_{fmt}.csv", configs)
    write_csv(results / f"weight_stats_{fmt}.csv", weights)

    baseline = load_baseline(Path(args.reference))
    slopes = {
        d.name: json.loads((d / "curve.json").read_text(encoding="utf-8")).get("slopes", {})
        for d in Path(args.reference).glob("*") if (d / "curve.json").is_file()
    }
    write_json(results / "reference.json", {
        "baseline": [baseline[key] for key in sorted(baseline)],
        "noise_floor": {d.name: json.loads((d / "noise_floor.json").read_text(encoding="utf-8"))
                        for d in Path(args.reference).glob("*") if (d / "noise_floor.json").is_file()},
        "g0a": {d.name: json.loads((d / "g0a.json").read_text(encoding="utf-8"))
                for d in Path(args.reference).glob("*") if (d / "g0a.json").is_file()},
        "slopes": slopes,
    })

    manifests = []
    for manifest in Path(args.manifests).glob("*.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        config = payload.get("quant_config", {})
        if config.get("format") == fmt or (fmt == "int4" and config.get("control") == "int4"):
            manifests.append(payload)
    write_json(results / "manifests.json", manifests)

    config_by_id = {row["config_id"]: row for row in configs}
    metric_rows = []
    for row in sweep:
        config = config_by_id.get(row["config_id"], {})
        ref_bpp, ref_psnr = float(row["ref_bpp"]), float(row["ref_psnr"])
        bpp, psnr = float(row["bpp"]), float(row["psnr"])
        d_psnr = psnr - ref_psnr
        d_bpp = 100.0 * (bpp / ref_bpp - 1.0)
        slope = slopes.get(row["dataset"], {}).get(row["qp"])
        penalty = None if slope is None else 100.0 * ((1.0 + d_bpp / 100.0) * 2.0 ** (-d_psnr / slope) - 1.0)
        metric_rows.append({
            "dataset": row["dataset"], "component": config.get("component", ""),
            "control": config.get("control", ""), "qp": int(row["qp"]),
            "bpp": bpp, "psnr": psnr, "d_psnr": d_psnr,
            "d_bpp_pct": d_bpp, "penalty_pct": penalty,
            "recon_match_ref": row["recon_hash"] == row["ref_recon_hash"],
            "stream_match_ref": row["stream_hash"] == row["ref_stream_hash"],
            "status": row["status"],
        })

    groups = defaultdict(list)
    for row in metric_rows:
        groups[(row["dataset"], row["component"], row["control"], row["qp"])].append(row)
    summary_rows = []
    for (dataset, component, control, qp), group in sorted(groups.items()):
        summary_rows.append({
            "dataset": dataset, "component": component, "control": control, "qp": qp,
            **summarize(group),
        })
    write_csv(results / "summary.csv", summary_rows)

    weight_groups = defaultdict(list)
    for row in weights:
        config = config_by_id.get(row["config_id"], {})
        weight_groups[(config.get("component", ""), config.get("control", ""))].append(row)
    weight_summary = []
    for (component, control), group in sorted(weight_groups.items()):
        weight_summary.append({
            "component": component, "control": control,
            "layers": len(group),
            "mean_sqnr_db": statistics.mean(float(row["sqnr_db"]) for row in group),
            "max_clip_frac": max(float(row["clip_frac"]) for row in group),
            "max_abs_err": max(float(row["max_abs_err"]) for row in group),
            "mean_effective_bits": statistics.mean(float(row["effective_bits"]) for row in group),
        })

    size_by_component = {row["component"]: row for row in size_table(
        json.loads(Path(args.inventory).read_text(encoding="utf-8")), fmt)}
    table = [f"# {fmt} results", "", f"Rows: {len(sweep)}.", "",
             "Fake-quant timing is not meaningful; latency and energy belong to the real-format benchmark.", "",
             "## Baseline", "",
             "| Dataset | QP | bpp | PSNR | encode ms | decode ms | total ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(baseline):
        row = baseline[key]
        table.append(f"| {row['dataset']} | {row['qp']} | {row['ref_bpp']:.4f} | "
                     f"{row['ref_psnr']:.4f} | {row['ref_encode_ms']:.2f} | "
                     f"{row['ref_decode_ms']:.2f} | {row['ref_total_ms']:.2f} |")

    control_groups = defaultdict(list)
    for row in metric_rows:
        control_groups[(row["component"], row["control"])].append(row)
    table.extend(["", "## Controls", "",
                  "| Component | Control | rows | recon match | stream match | mean d_bpp | mean d_psnr |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for (component, control), group in sorted(control_groups.items()):
        summary = summarize(group)
        table.append(f"| {component} | {control} | {summary['rows']} | "
                     f"{summary['recon_match_rate']:.4f} | {summary['stream_match_rate']:.4f} | "
                     f"{summary['mean_d_bpp_pct']:+.4f}% | {summary['mean_d_psnr']:+.4f} |")
    table.append("")

    metrics = [
        ("mean_d_psnr", "d_PSNR (dB)", "dB", "d_psnr"),
        ("mean_d_bpp_pct", "d_bpp (%)", "percent", "d_bpp"),
        ("mean_penalty_pct", "rate-equivalent penalty (%)", "percent", "penalty"),
        ("mean_bpp", "bpp", "bpp", "bpp"),
        ("mean_psnr", "PSNR (dB)", "dB", "psnr"),
    ]
    datasets = sorted({row["dataset"] for row in metric_rows})
    primary = "clic_professional" if "clic_professional" in datasets else (datasets[0] if datasets else None)
    for dataset in datasets:
        dataset_rows = [row for row in summary_rows if row["dataset"] == dataset]
        analysis = [row for row in dataset_rows if row["control"] != "identity"] or dataset_rows
        components = sorted({row["component"] for row in analysis})
        table.extend(["", f"## {dataset} component x QP", ""])
        for metric, label, unit, short in metrics:
            table.extend([f"### {label}", "",
                          "| Component | Baseline | QP0 | QP1 | QP2 | QP3 | QP4 | QP5 | QP6 | QP7 |",
                          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
            for component in components:
                values = [next((f"{row[metric]:.4f}" if row[metric] is not None else "-"
                                for row in analysis if row["component"] == component and row["qp"] == qp), "-")
                          for qp in range(8)]
                table.append(f"| {component} | 0.0000 | " + " | ".join(values) + " |")
            table.append("")
            if dataset == primary:
                series = []
                for index, component in enumerate(components):
                    values = [next((row[metric] for row in analysis
                                    if row["component"] == component and row["qp"] == qp), None)
                              for qp in range(8)]
                    series.append((component, values, plots.PALETTE[index % len(plots.PALETTE)]))
                plots.line_chart(figures / f"component_qp_{short}_lines.png",
                                 f"{fmt}: {label} by QP ({dataset})",
                                 [str(qp) for qp in range(8)], series, unit)

        table.extend(["### correctness", "",
                      "| Component | Control | recon match | stream match | mean d_bpp |",
                      "|---|---|---:|---:|---:|"])
        for row in analysis:
            table.append(f"| {row['component']} | {row['control']} | "
                         f"{row['recon_match_rate']:.4f} | {row['stream_match_rate']:.4f} | "
                         f"{row['mean_d_bpp_pct']:+.4f}% |")
        table.append("")

    table.extend(["### size reduction", "",
                  "| Component | FP32 bytes | quant bytes | reduction % | effective bits/param |",
                  "|---|---:|---:|---:|---:|"])
    for component in sorted(size_by_component):
        size = size_by_component[component]
        table.append(f"| {component} | {size['fp32_bytes']} | {size['quant_bytes']} | "
                     f"{size['size_reduction_pct']:.2f}% | {size['effective_bits_per_param']:.3f} |")
    table.extend(["", "### weight statistics", "",
                  "| Component | Control | layers | mean SQNR dB | max clip frac | max abs err | mean effective bits |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for row in weight_summary:
        table.append(f"| {row['component']} | {row['control']} | {row['layers']} | "
                     f"{row['mean_sqnr_db']:.2f} | {row['max_clip_frac']:.4f} | "
                     f"{row['max_abs_err']:.6f} | {row['mean_effective_bits']:.3f} |")

    main = [row for row in size_by_component.values() if row["component"] not in {"H4", "H5", "D9"}]
    fp_main = sum(row["fp32_bytes"] for row in main)
    q_main = sum(row["quant_bytes"] for row in main)
    table.extend(["", "## Theoretical size", "",
                  f"- Main 19 components: {fp_main / 1e6:.2f} MB -> {q_main / 1e6:.2f} MB "
                  f"({100.0 * (1.0 - q_main / fp_main):.2f}% reduction).", "",
                  "## Files", "", f"- `sweep_{fmt}.csv`", f"- `configs_{fmt}.csv`",
                  f"- `weight_stats_{fmt}.csv`", "- `summary.csv`", "- `reference.json`",
                  "- `manifests.json`", "- `figures/`"])
    (results / "README.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    if main:
        plots.h_bar_chart(figures / "size_reduction_by_component.png",
                          f"{fmt}: theoretical size reduction",
                          [row["component"] for row in main],
                          [row["size_reduction_pct"] for row in main],
                          "size reduction (%)")
    print(json.dumps({"format": fmt, "rows": len(sweep), "output": str(results)}, indent=2))


def run_overnight(args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"overnight_{stamp}.log"
    ledger = Path(args.ledger)
    output_root = Path(args.output_root)
    pipeline = Path(__file__).resolve()
    log_stream = log_path.open("w", encoding="utf-8")

    def log(message):
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        log_stream.write(line + "\n")
        log_stream.flush()

    def call(arguments):
        cmd = [sys.executable, str(pipeline), *map(str, arguments)]
        log("RUN " + subprocess.list2cmdline(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            log_stream.write(result.stdout)
        if result.stderr:
            log_stream.write(result.stderr)
        log_stream.flush()
        log(f"EXIT {result.returncode}")
        return result.returncode

    if args.dry_run:
        args.datasets = ["kodak_thumb"]
        args.formats = ["int8"]
        args.components = ["D1"]
        args.qps = [0]
        args.max_images = 1
        args.ledger = "C:/Users/ASUS/AppData/Local/Temp/opencode/overnight_dry/ledger"
        args.output_root = "C:/Users/ASUS/AppData/Local/Temp/opencode/overnight_dry/out"
        args.log_dir = "C:/Users/ASUS/AppData/Local/Temp/opencode/overnight_dry/logs"
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ledger = Path(args.ledger)
        output_root = Path(args.output_root)
        log_path = log_dir / f"overnight_{stamp}.log"
        log_stream.close()
        log_stream = log_path.open("w", encoding="utf-8")

    ledger.mkdir(parents=True, exist_ok=True)
    reference_dir = ledger / "reference"
    log(f"datasets={args.datasets} formats={args.formats} components={args.components} qps={args.qps}")
    log(f"ledger={ledger} output_root={output_root}")

    ready_datasets = []
    for dataset in args.datasets:
        path = Path("datasets") / dataset
        if not path.is_dir():
            log(f"skip dataset {dataset}: missing {path}")
            continue
        count = len(image_paths(path))
        if dataset == "kodak" and count < 24:
            log(f"skip dataset kodak: {count}/24 full-resolution PNGs present")
            continue
        ready_datasets.append(dataset)
        log(f"dataset {dataset}: {count} images")
    if not ready_datasets:
        log("no datasets ready; stopping")
        log_stream.close()
        return

    failures = 0
    for dataset in ready_datasets:
        pass1 = reference_dir / dataset / "pass1.csv"
        if args.skip_reference and pass1.is_file():
            log(f"reference exists for {dataset}; skipping")
        else:
            code = call(["reference", "--dataset", dataset, "--qps", *args.qps,
                         "--output-dir", str(reference_dir), "--device", args.device,
                         "--threads", args.threads])
            if code:
                failures += 1
                if failures >= 3:
                    log("stopping after 3 consecutive failures")
                    log_stream.close()
                    return
            else:
                failures = 0

    for fmt in args.formats:
        for dataset in ready_datasets:
            code = call(["sweep", "--format", fmt, "--components", *args.components,
                         "--control", "none", "--dataset", dataset, "--qps", *args.qps,
                         *(["--max-images", str(args.max_images)] if args.max_images else []),
                         "--reference-dir", str(reference_dir),
                         "--output", str(ledger / "sweep.csv"),
                         "--configs-output", str(ledger / "configs.csv"),
                         "--weight-stats-output", str(ledger / "weight_stats.csv"),
                         "--progress", str(ledger / "progress.jsonl"),
                         "--manifests", str(ledger / "manifests"),
                         "--device", args.device, "--threads", args.threads])
            if code:
                failures += 1
                if failures >= 3:
                    log("stopping after 3 consecutive failures")
                    log_stream.close()
                    return
            else:
                failures = 0

            if fmt == "int8" and dataset in {"kodak", "kodak_thumb", "clic_professional"}:
                control_qps = [q for q in args.qps if q in {0, 3, 7}] or args.qps
                control_images = args.max_images or 8
                code = call(["sweep", "--format", "int8", "--components", "D1",
                             "--control", "identity", "--dataset", dataset,
                             "--qps", *control_qps, "--max-images", str(control_images),
                             "--reference-dir", str(reference_dir),
                             "--output", str(ledger / "sweep.csv"),
                             "--configs-output", str(ledger / "configs.csv"),
                             "--weight-stats-output", str(ledger / "weight_stats.csv"),
                             "--progress", str(ledger / "progress.jsonl"),
                             "--manifests", str(ledger / "manifests"),
                             "--device", args.device, "--threads", args.threads])
                if code:
                    failures += 1
                    if failures >= 3:
                        log("stopping after 3 consecutive failures")
                        log_stream.close()
                        return
                else:
                    failures = 0

        code = call(["export", "--format", fmt, "--sweep", str(ledger / "sweep.csv"),
                     "--configs", str(ledger / "configs.csv"),
                     "--weight-stats", str(ledger / "weight_stats.csv"),
                     "--reference", str(reference_dir), "--manifests", str(ledger / "manifests"),
                     "--inventory", "quantization-pluse-mse/artifacts/component_inventory.json",
                     "--output-root", str(output_root)])
        if code:
            failures += 1
            if failures >= 3:
                log("stopping after 3 consecutive failures")
                log_stream.close()
                return
        else:
            failures = 0

    sweep_rows = read_csv(ledger / "sweep.csv")
    configs = read_csv(ledger / "configs.csv")
    weights = read_csv(ledger / "weight_stats.csv")
    summary = {
        "timestamp": stamp,
        "datasets": ready_datasets,
        "formats": args.formats,
        "sweep_rows": len(sweep_rows),
        "configs": len(configs),
        "weight_rows": len(weights),
        "failures": sum(1 for row in sweep_rows if row.get("status") != "ok"),
        "status_counts": {},
    }
    for row in sweep_rows:
        summary["status_counts"][row.get("status", "")] = summary["status_counts"].get(row.get("status", ""), 0) + 1
    write_json(output_root / "overnight_summary.json", summary)
    report = [
        "# Overnight INT8/INT4 run",
        "",
        f"Datasets: {', '.join(ready_datasets)}",
        f"Formats: {', '.join(args.formats)}",
        f"Sweep rows: {summary['sweep_rows']}",
        f"Configs: {summary['configs']}",
        f"Weight-stat rows: {summary['weight_rows']}",
        f"Failures: {summary['failures']}",
        "",
        "## Status counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in sorted(summary["status_counts"].items()):
        report.append(f"| {status} | {count} |")
    report.extend([
        "",
        "## Notes",
        "",
        "- No quantized checkpoints are saved; weights are restored after every run.",
        "- Fake-quant latency, memory and energy are excluded from this sweep.",
        "- CDF/symbol mismatch counts are not implemented; correctness is decode/stream/recon/weight-hash based.",
        "- Full-resolution Kodak is skipped if `datasets/kodak` has fewer than 24 files.",
    ])
    (output_root / "overnight_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    log(json.dumps(summary, indent=2))
    log_stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("inventory")
    inv.add_argument("--model", default="checkpoints/pulse-mse")
    inv.add_argument("--height", type=int, default=1088)
    inv.add_argument("--width", type=int, default=1920)
    inv.add_argument("--qp", type=int, default=3)
    inv.add_argument("--kodak", default="datasets/kodak_thumb")
    inv.add_argument("--output", default="quantization-pluse-mse/artifacts/component_inventory.json")
    inv.set_defaults(func=run_inventory)

    ref = sub.add_parser("reference")
    ref.add_argument("--model", default="checkpoints/pulse-mse")
    ref.add_argument("--dataset", default="kodak")
    ref.add_argument("--input", default=None)
    ref.add_argument("--qps", type=int, nargs="+", default=list(range(8)))
    ref.add_argument("--max-images", type=int, default=None)
    ref.add_argument("--device", default="cuda")
    ref.add_argument("--threads", type=int, default=1)
    ref.add_argument("--legacy", action="store_true")
    ref.add_argument("--output-dir", default="quantization-pluse-mse/results/_ledger/reference")
    ref.add_argument("--frozen-json", default=None)
    ref.set_defaults(func=run_reference)

    swp = sub.add_parser("sweep")
    swp.add_argument("--model", default="checkpoints/pulse-mse")
    swp.add_argument("--format", default="int8", choices=[
        "fp32", "int8", "int4", "fp16", "bf16",
        "fp8_e4m3", "fp8_e5m2", "fp4_e2m1",
    ])
    swp.add_argument("--components", nargs="+", default=["D1"])
    swp.add_argument("--mode", default="w_only", choices=["w_only"])
    swp.add_argument("--control", default="none", choices=["none", "identity", "int4"])
    swp.add_argument("--dataset", default="kodak")
    swp.add_argument("--qps", type=int, nargs="+", default=list(range(8)))
    swp.add_argument("--max-images", type=int, default=None)
    swp.add_argument("--device", default="cuda")
    swp.add_argument("--threads", type=int, default=1)
    swp.add_argument("--tf32", default="off", choices=["off", "on"])
    swp.add_argument("--legacy", action="store_true")
    swp.add_argument("--reference-dir", default="quantization-pluse-mse/results/_ledger/reference")
    swp.add_argument("--output", default="quantization-pluse-mse/results/_ledger/sweep.csv")
    swp.add_argument("--progress", default="quantization-pluse-mse/results/_ledger/progress.jsonl")
    swp.add_argument("--manifests", default="quantization-pluse-mse/results/_ledger/manifests")
    swp.add_argument("--configs-output", default="quantization-pluse-mse/results/_ledger/configs.csv")
    swp.add_argument("--weight-stats-output", default="quantization-pluse-mse/results/_ledger/weight_stats.csv")
    swp.add_argument("--overwrite", action="store_true")
    swp.set_defaults(func=run_sweep)

    exp = sub.add_parser("export")
    exp.add_argument("--format", required=True)
    exp.add_argument("--sweep", default="quantization-pluse-mse/results/_ledger/sweep.csv")
    exp.add_argument("--configs", default="quantization-pluse-mse/results/_ledger/configs.csv")
    exp.add_argument("--weight-stats", default="quantization-pluse-mse/results/_ledger/weight_stats.csv")
    exp.add_argument("--reference", default="quantization-pluse-mse/results/_ledger/reference")
    exp.add_argument("--manifests", default="quantization-pluse-mse/results/_ledger/manifests")
    exp.add_argument("--inventory", default="quantization-pluse-mse/artifacts/component_inventory.json")
    exp.add_argument("--output-root", default="quantization-pluse-mse")
    exp.set_defaults(func=run_export)

    night = sub.add_parser("overnight")
    night.add_argument("--datasets", nargs="+", default=["kodak_thumb", "clic_professional"])
    night.add_argument("--formats", nargs="+", default=[
        "int8", "int4", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1",
    ])
    night.add_argument("--components", nargs="+", default=["all"])
    night.add_argument("--qps", type=int, nargs="+", default=list(range(8)))
    night.add_argument("--max-images", type=int, default=None)
    night.add_argument("--device", default="cuda")
    night.add_argument("--threads", type=int, default=1)
    night.add_argument("--ledger", default="quantization-pluse-mse/results/_ledger")
    night.add_argument("--output-root", default="quantization-pluse-mse")
    night.add_argument("--log-dir", default="quantization-pluse-mse/logs")
    night.add_argument("--skip-reference", action="store_true")
    night.add_argument("--dry-run", action="store_true")
    night.set_defaults(func=run_overnight)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
