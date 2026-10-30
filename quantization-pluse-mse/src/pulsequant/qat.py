"""Quantization-aware fine-tuning: straight-through quantized convolutions and bundle handling."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from pulsequant.core import fake_quant, parse_format, conv_modules, COMPONENTS


class QuantSTEConv2d(nn.Module):
    """A Conv2d whose weight is fake-quantized on every forward, with a straight-through gradient."""

    def __init__(self, conv: nn.Conv2d, spec: str):
        super().__init__()
        self.conv = conv
        self.spec = spec
        base, _ = parse_format(spec)
        self.qmax = 7 if base == "int4" else 127

    def forward(self, x):
        weight = self.conv.weight
        with torch.no_grad():
            quantized = fake_quant(weight, self.spec, self.qmax)
        weight = weight + (quantized - weight).detach()
        return F.conv2d(x, weight, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)

    def extra_repr(self) -> str:
        return f"spec={self.spec}"


@dataclass
class WrapRecord:
    parent: nn.Module
    name: str
    wrapper: QuantSTEConv2d
    original: nn.Conv2d
    component: str


def _parent_of(model: nn.Module, target: str) -> tuple[nn.Module, str]:
    head, _, tail = target.rpartition(".")
    parent = model.get_submodule(head) if head else model
    return parent, tail


def wrap_assignment(model: nn.Module, assignment: dict[str, str]) -> list[WrapRecord]:
    """Replace each assigned component's Conv2d modules with STE quantized wrappers."""
    records: list[WrapRecord] = []
    for component, spec in sorted(assignment.items()):
        if component not in COMPONENTS:
            raise ValueError(f"unknown component {component!r}")
        parse_format(spec)
        for name, conv in list(conv_modules(model, component)):
            parent, attribute = _parent_of(model, name)
            wrapper = QuantSTEConv2d(conv, spec)
            setattr(parent, attribute, wrapper)
            records.append(WrapRecord(parent, attribute, wrapper, conv, component))
    if not records:
        raise ValueError("assignment wrapped no convolutions")
    return records


def unwrap(records: list[WrapRecord]) -> None:
    """Restore the original Conv2d modules, keeping whatever weights training produced."""
    for record in records:
        setattr(record.parent, record.name, record.wrapper.conv)
    records.clear()


def set_trainable(model: nn.Module, scope: str, components: set[str]) -> tuple[int, int]:
    """Freeze by scope. Returns (trainable params, frozen params)."""
    if scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        prefixes: list[str] = []
        if scope == "backbone":
            prefixes = ["encoder", "decoder"]
        elif scope == "assigned":
            for component in components:
                prefixes.extend(str(p).rstrip(".") for p in COMPONENTS[component]["prefixes"])
        elif scope == "encoder":
            prefixes = ["encoder"]
        else:
            raise ValueError(f"unknown train scope {scope!r}")
        for name, parameter in model.named_parameters():
            if any(name == p or name.startswith(p + ".") for p in prefixes):
                parameter.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def save_bundle(model: nn.Module, source: Path, destination: Path) -> Path:
    """Write a loadable bundle that keeps the native checkpoint schema.

    The released `model.pt` stores the float scale decoder, while a model loaded with the integer
    entropy control holds different keys under `scale_dec`. Saving the loaded model directly would
    therefore produce a bundle that cannot be reloaded. Instead the baseline state dict is used as
    the schema and every matching tensor is overwritten with the fine-tuned value.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copy2(source / "config.json", destination / "config.json")
    baseline = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
    schema = baseline["state_dict"] if "state_dict" in baseline else baseline
    current = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    differing = [
        key for key, value in schema.items()
        if key.startswith("entropy_model.")
        and current.get(key) is not None
        and (current[key].shape != value.shape
             or not torch.equal(current[key].float(), value.float()))
    ]
    if differing:
        raise RuntimeError(
            "entropy_model tensors changed during fine-tuning, so the integer CDF artifact is no "
            f"longer valid ({len(differing)} tensors, e.g. {differing[:3]}). Freeze the entropy "
            "components, or re-run postprocess.py on this bundle to regenerate the artifact.")

    merged, replaced = {}, 0
    for key, value in schema.items():
        candidate = current.get(key)
        if candidate is not None and candidate.shape == value.shape:
            merged[key] = candidate
            replaced += 1
        else:
            merged[key] = value
    model_path = destination / "model.pt"
    torch.save({"state_dict": merged}, model_path)

    artifact = torch.load(source / "entropy_control_int.pt", map_location="cpu", weights_only=True)
    calibration = dict(artifact.get("y_cdf_calibration") or {})
    previous = calibration.get("checkpoint_sha256")
    calibration["checkpoint_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    calibration["rebound_after_qat"] = True
    calibration["rebound_from"] = previous
    artifact["y_cdf_calibration"] = calibration
    torch.save(artifact, destination / "entropy_control_int.pt")

    (destination / "qat_schema.json").write_text(json.dumps({
        "baseline_keys": len(schema),
        "replaced_with_finetuned": replaced,
        "entropy_tensors_changed": 0,
        "entropy_hash_rebound": True,
        "previous_checkpoint_sha256": previous,
    }, indent=2) + "\n", encoding="utf-8")
    return destination
    (destination / "qat_schema.json").write_text(
        json.dumps({"baseline_keys": len(schema), "replaced_with_finetuned": replaced},
                   indent=2) + "\n", encoding="utf-8")
    return destination
