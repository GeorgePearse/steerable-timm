"""Reversible taps on named modules: ways to get signal and gradient moving without changing the architecture.

Each tap is a forward hook, so nothing is added to the model's parameters or state_dict and removing the tap
restores the original function exactly. For a tapped module f with input x:

    y = drop(scale * f(x)) + skip * x

Every value fades linearly to its neutral value (skip 0, scale 1, drops 0) over flow.fade_steps.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F

from .state import State

NEUTRAL = {"skip": 0.0, "scale": 1.0, "drop_path": 0.0, "dropout": 0.0}
KINDS = tuple(NEUTRAL)


class Shape(NamedTuple):
    tensor: bool  # one tensor in, one tensor out
    same: bool  # input and output shapes are equal
    mutates: bool  # the module changes its input in place


def module_matches(name: str, pattern: str) -> bool:
    """Exact module names: a glob or a re: pattern. Unlike parameter patterns, children are not implied."""
    if pattern.startswith("re:"):
        return re.fullmatch(pattern[3:], name) is not None
    return fnmatch.fnmatchcase(name, pattern)


def targets(module: torch.nn.Module, weights: dict[str, float]) -> dict[str, float]:
    """Module name -> value; the first matching pattern wins."""
    out: dict[str, float] = {}
    for name, _ in module.named_modules():
        value = next((v for p, v in weights.items() if name and module_matches(name, p)), None)
        if value is not None:
            out[name] = value
    return out


def shapes(live: Any, state: State) -> dict[str, Shape]:
    """One forward of a tiny random batch in eval mode, cached on live. Modules forward never calls are absent."""
    if live.shapes is not None:
        return live.shapes
    module = live.module
    x = live.example_input
    if x is None:
        first = next((p for p in module.parameters() if p.ndim == 4), None)
        if first is None:
            raise ValueError("flow: set Live.example_input for a model without a conv stem")
        res = state.data.resolution
        noise = torch.randn(2, first.shape[1], res, res, generator=torch.Generator().manual_seed(0))
        x = noise.to(device=first.device, dtype=first.dtype)
    seen: dict[str, Shape] = {}
    before: dict[str, torch.Tensor] = {}
    handles = []
    for name, m in module.named_modules():
        if not name:
            continue

        def pre(m: torch.nn.Module, args: tuple, name: str = name) -> None:
            if len(args) == 1 and torch.is_tensor(args[0]):
                before[name] = args[0].clone()

        def post(m: torch.nn.Module, args: tuple, out: Any, name: str = name) -> None:
            ok = name in before and torch.is_tensor(out)
            x0 = before.pop(name, None)
            mutates = ok and (out is args[0] or not torch.equal(x0, args[0]))
            seen[name] = Shape(ok, ok and x0.shape == out.shape, mutates)

        handles += [m.register_forward_pre_hook(pre), m.register_forward_hook(post)]
    modes = {m: m.training for m in module.modules()}
    module.eval()
    try:
        with torch.no_grad():
            module(x)
    finally:
        for h in handles:
            h.remove()
        for m, training in modes.items():
            m.training = training
    live.shapes = seen
    return seen


def modules(live: Any, state: State) -> list[dict[str, Any]]:
    """Every named module with its type, parameter count and which taps it can take."""
    info = shapes(live, state)
    return [
        {
            "name": name,
            "type": type(m).__name__,
            "params": sum(p.numel() for p in m.parameters()),
            "tap": name in info and info[name].tensor,
            "skip": name in info and info[name].same,
        }
        for name, m in live.module.named_modules()
        if name
    ]


def check_flow(live: Any, state: State) -> None:
    flow = state.flow
    if not any(getattr(flow, kind) for kind in KINDS):
        return
    info = shapes(live, state)
    names = [name for name, _ in live.module.named_modules() if name]
    for kind in KINDS:
        weights = getattr(flow, kind)
        for pattern in weights:
            if not any(module_matches(name, pattern) for name in names):
                raise ValueError(f"flow.{kind}: no module matches {pattern!r}")
        for name in targets(live.module, weights):
            if name not in info or not info[name].tensor:
                raise ValueError(f"flow.{kind}: {name} does not take and return one tensor in forward")
            if kind == "skip" and not info[name].same:
                raise ValueError(f"flow.skip: {name} changes the shape of its input, so x cannot be added")


def gain(live: Any, fade_steps: int) -> float:
    if not fade_steps:
        return 1.0
    return min(1.0, max(0.0, 1.0 - (live.step - live.flow_origin) / fade_steps))


def apply_flow(live: Any, state: State, step: int) -> None:
    """Remove every tap, then register the ones the State asks for."""
    for handle in live.taps:
        handle.remove()
    live.taps = []
    flow = state.flow
    live.step = max(live.step, step)
    live.flow_origin = step if flow.origin is None else flow.origin
    live.flow_faded = False
    per: dict[str, dict[str, float]] = {}
    for kind in KINDS:
        for name, value in targets(live.module, getattr(flow, kind)).items():
            if value != NEUTRAL[kind]:
                per.setdefault(name, {})[kind] = value
    if not per:
        return
    info = shapes(live, state) if any("skip" in v for v in per.values()) else {}
    mods = dict(live.module.named_modules())
    for name, values in per.items():
        clone = "skip" in values and info[name].mutates
        live.taps += _tap(live, mods[name], values, flow.fade_steps, clone)


def _tap(live: Any, module: torch.nn.Module, values: dict[str, float], fade_steps: int, clone: bool) -> list:
    stash: dict[str, torch.Tensor] = {}

    def pre(m: torch.nn.Module, args: tuple) -> None:
        if gain(live, fade_steps):
            stash["x"] = args[0].clone()

    def post(m: torch.nn.Module, args: tuple, out: torch.Tensor) -> torch.Tensor | None:
        g = gain(live, fade_steps)
        if not g:
            return None
        scale = 1.0 + (values.get("scale", 1.0) - 1.0) * g
        if scale != 1.0:
            out = out * scale
        if live.module.training:
            if p := values.get("drop_path", 0.0) * g:
                keep = out.new_empty(out.shape[0], *[1] * (out.ndim - 1)).bernoulli_(1.0 - p)
                out = out * keep / (1.0 - p)
            if p := values.get("dropout", 0.0) * g:
                out = F.dropout(out, p, training=True)
        if alpha := values.get("skip", 0.0) * g:
            out = out + alpha * (stash.pop("x") if clone else args[0])
        return out

    handles = [module.register_forward_pre_hook(pre)] if clone else []
    return [*handles, module.register_forward_hook(post)]


def tick(live: Any, state: State, step: int) -> None:
    """Advance the fade clock. set_lr calls this every step."""
    live.step = step
    if live.taps and not live.flow_faded and not gain(live, state.flow.fade_steps):
        live.flow_faded = True
        live.notes.append(f"flow faded out at step {step}: taps are no-ops until flow changes")


def grad_flow(module: torch.nn.Module, depth: int = 1) -> dict[str, float]:
    """Gradient L2 norm per parameter-name prefix of `depth` parts. Call after backward, before zero_grad."""
    squares: dict[str, torch.Tensor] = {}
    for name, p in module.named_parameters():
        if p.grad is not None:
            key = ".".join(name.split(".")[:depth])
            squares[key] = squares.get(key, 0.0) + p.grad.detach().float().pow(2).sum()
    if not squares:
        return {}
    norms = torch.stack(list(squares.values())).sqrt().tolist()
    return dict(zip(squares, norms, strict=True))
