"""Component map, INT8/INT4 fake quantization, and in-place restore."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

COMPONENTS = {
    "E0": {"name": "patch_embedding", "prefixes": ["encoder.patch_embed"], "side": "encode"},
    "E1": {"name": "encoder_dcb_0", "prefixes": ["encoder.blocks.0"], "side": "encode"},
    "E2": {"name": "encoder_dcb_1", "prefixes": ["encoder.blocks.1"], "side": "encode"},
    "E3": {"name": "encoder_dcb_2", "prefixes": ["encoder.blocks.2"], "side": "encode"},
    "E4": {"name": "encoder_dcb_3", "prefixes": ["encoder.blocks.3"], "side": "encode"},
    "E5": {"name": "latent_projection", "prefixes": ["encoder.norm_out", "encoder.proj_out"], "side": "encode"},
    "H0": {"name": "hyper_encoder", "prefixes": ["entropy_model.hyper_enc"], "side": "encode"},
    "H1": {"name": "hyper_decoder", "prefixes": ["entropy_model.hyper_dec"], "side": "both"},
    "H2": {"name": "em_step_1", "prefixes": ["entropy_model.em_step1"], "side": "both"},
    "H3": {"name": "em_step_2", "prefixes": ["entropy_model.em_step2"], "side": "both"},
    "H4": {"name": "scale_decoder_control", "prefixes": ["entropy_model.scale_dec"], "side": "both"},
    "H5": {"name": "entropy_lookup_control", "prefixes": ["entropy_model.entropy_bottleneck", "entropy_model.meta_prior"], "side": "both"},
    "D0": {"name": "initial_projection", "prefixes": ["decoder.up0"], "side": "decode"},
    "D1": {"name": "decoder_dcb_0", "prefixes": ["decoder.blocks.0"], "side": "decode"},
    "D2": {"name": "decoder_dcb_1", "prefixes": ["decoder.blocks.1"], "side": "decode"},
    "D3": {"name": "latent_upsampling", "prefixes": ["decoder.renderer.latent_upsample"], "side": "decode"},
    "D4": {"name": "condition_projection", "prefixes": ["decoder.renderer.condition_proj"], "side": "decode"},
    "D5": {"name": "renderer_mlp", "prefixes": ["decoder.renderer.mlp"], "side": "decode"},
    "D6": {"name": "adaln", "prefixes": ["decoder.renderer.adaln"], "side": "decode"},
    "D7": {"name": "output_projection", "prefixes": ["decoder.renderer.output"], "side": "decode"},
    "D8": {"name": "renderer_input", "prefixes": ["decoder.renderer.input_proj"], "side": "decode"},
    "D9": {
        "name": "qp_gates_control",
        "prefixes": [
            "decoder.q_scale_codec_dec", "decoder.q_scale_dico", "decoder.renderer.q_scale",
            "q_scale_enc", "entropy_model.q_scale_hyper_z", "entropy_model.q_scale_hyper_m",
            "entropy_model.q_scale_prior_2m", "entropy_model.q_basic",
        ],
        "side": "both",
    },
}

MAIN_SWEEP = [
    "E0", "E1", "E2", "E3", "E4", "E5", "H0", "H1", "H2", "H3",
    "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",
]
CONTROLS = ["H4", "H5", "D9"]


def prefixes(component: str) -> list[str]:
    return COMPONENTS[component]["prefixes"]


def side(component: str) -> str:
    return COMPONENTS[component]["side"]


def conv_modules(model: nn.Module, component: str):
    names = prefixes(component)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and any(
            name == prefix or name.startswith(prefix + ".") for prefix in names
        ):
            yield name, module


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def weights_sha256(weights: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(weight.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def quant_scale(source: torch.Tensor, fmt: str, qmax: int = 127):
    max_abs = source.abs().amax(dim=tuple(range(1, source.ndim)), keepdim=True).clamp_min(1e-12)
    if fmt in {"fp16", "bf16"}:
        return None
    if fmt == "fp8_e4m3":
        return max_abs / 448.0
    if fmt == "fp8_e5m2":
        return max_abs / 57344.0
    if fmt == "fp4_e2m1":
        return max_abs / 6.0
    if fmt == "int4":
        return max_abs / 7.0
    return max_abs / qmax


def fake_quant(weight: torch.Tensor, fmt: str = "int8", qmax: int = 127) -> torch.Tensor:
    source = weight.detach().float()
    if fmt in {"fp16", "bf16"}:
        dtype = torch.float16 if fmt == "fp16" else torch.bfloat16
        return source.to(dtype).to(weight.dtype)
    if qmax < 1 or qmax > 127:
        raise ValueError(f"qmax must be in [1, 127], got {qmax}")
    scale = quant_scale(source, fmt, qmax)
    if fmt in {"fp8_e4m3", "fp8_e5m2"}:
        dtype = torch.float8_e4m3fn if fmt == "fp8_e4m3" else torch.float8_e5m2
        return ((source / scale).to(dtype).to(torch.float32) * scale).to(weight.dtype)
    if fmt == "fp4_e2m1":
        grid = torch.tensor(
            [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
             0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            device=source.device,
        )
        boundaries = (grid[:-1] + grid[1:]) / 2.0
        return (grid[torch.bucketize(source / scale, boundaries)] * scale).to(weight.dtype)
    if fmt in {"int8", "int4"}:
        return (torch.round(source / scale).clamp(-qmax, qmax) * scale).to(weight.dtype)
    raise ValueError(f"unknown format {fmt!r}")


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


def component_size(model: nn.Module, component: str, fmt: str) -> dict:
    total = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes(component))
    )
    conv_numel = 0.0
    quant_bits = 0.0
    for name, module in conv_modules(model, component):
        numel = float(module.weight.numel())
        fan_in = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        conv_numel += numel
        quant_bits += numel * weight_bits(fmt, fan_in)
    non_conv = max(0.0, total - conv_numel)
    fp32_bytes = total * 4.0
    quant_bytes = quant_bits / 8.0 + non_conv * 4.0
    return {
        "fp32_bytes": int(round(fp32_bytes)),
        "quant_bytes": int(round(quant_bytes)),
        "size_reduction_pct": 100.0 * (1.0 - quant_bytes / fp32_bytes) if fp32_bytes else 0.0,
        "effective_bits_per_param": quant_bytes * 8.0 / total if total else 0.0,
    }


def weight_stats(handle: "QuantHandle", fmt: str) -> list[dict]:
    rows = []
    for name in handle.module_names:
        original = handle.originals[name].detach().float()
        quantized = handle.modules[name].weight.detach().float()
        error = original - quantized
        signal = original.pow(2).sum().item()
        noise = error.pow(2).sum().item()
        fan_in = (
            handle.modules[name].in_channels // handle.modules[name].groups
        ) * handle.modules[name].kernel_size[0] * handle.modules[name].kernel_size[1]
        scale = quant_scale(original, fmt, handle.qmax)
        if scale is None:
            clip_frac = 0.0
        elif fmt in {"fp8_e4m3", "fp8_e5m2"}:
            max_finite = 448.0 if fmt == "fp8_e4m3" else 57344.0
            clip_frac = float(((original / scale).abs() >= max_finite * 0.999).float().mean().item())
        elif fmt == "fp4_e2m1":
            clip_frac = float(((original / scale).abs() >= 6.0 * 0.999).float().mean().item())
        else:
            clip_frac = float(
                (torch.round(original / scale).clamp(-handle.qmax, handle.qmax).abs() >= handle.qmax)
                .float().mean().item()
            )
        rows.append({
            "layer": name,
            "sqnr_db": 10.0 * math.log10(signal / noise) if noise > 0 else float("inf"),
            "clip_frac": clip_frac,
            "max_abs_err": float(error.abs().max().item()),
            "fan_in": fan_in,
            "effective_bits": weight_bits(fmt, float(fan_in)),
        })
    return rows


@dataclass
class QuantHandle:
    component: str
    qmax: int
    control: str
    module_names: list[str]
    originals: dict[str, torch.Tensor] = field(repr=False)
    fmt: str = "int8"
    original_hash: str = ""
    quantized_hash: str = ""
    restored_hash: str = ""
    modules: dict[str, nn.Conv2d] = field(default_factory=dict, repr=False)

    def restore(self) -> None:
        for name, module in self.modules.items():
            with torch.no_grad():
                module.weight.copy_(self.originals[name])
        self.restored_hash = weights_sha256(
            [self.modules[name].weight.detach() for name in self.module_names]
        )

    @property
    def weights_changed(self) -> bool:
        return self.quantized_hash != self.original_hash

    @property
    def weights_restored(self) -> bool:
        return self.restored_hash == self.original_hash


def apply_quant(
    model: nn.Module, component: str, *, fmt: str = "int8", qmax: int = 127, control: str = "none"
) -> QuantHandle:
    modules = dict(conv_modules(model, component))
    if not modules:
        raise ValueError(f"component {component!r} resolved no Conv2d modules")
    names = sorted(modules)
    originals = {name: modules[name].weight.detach().clone() for name in names}
    original_hash = weights_sha256([originals[name] for name in names])
    if control == "identity":
        quantized_hash = original_hash
    elif control in {"none", "int4"}:
        if control == "int4":
            fmt, qmax = "int4", 7
        with torch.no_grad():
            for name in names:
                modules[name].weight.copy_(fake_quant(modules[name].weight, fmt, qmax))
        quantized_hash = weights_sha256([modules[name].weight.detach() for name in names])
    else:
        raise ValueError(f"unknown control {control!r}")
    return QuantHandle(
        component=component, qmax=qmax, control=control, module_names=names,
        originals=originals, fmt=fmt, original_hash=original_hash,
        quantized_hash=quantized_hash, modules=modules,
    )


def assert_handle(handle: QuantHandle) -> None:
    if handle.control == "identity":
        if handle.weights_changed:
            raise AssertionError("identity control changed the weights")
    elif not handle.weights_changed:
        raise AssertionError("quantization did not change the weights")
    if not handle.weights_restored:
        raise AssertionError("weights were not restored exactly")
