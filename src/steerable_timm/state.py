from __future__ import annotations

import fnmatch
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Weights = dict[str, float]


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Optimizer(Section):
    name: str = Field("adamw", description="timm optimizer name; a change rebuilds the optimizer")
    lr: float = Field(1e-3, gt=0, le=10, description="base lr, before lr_mult and the schedule factor")
    weight_decay: float = Field(0.05, ge=0, le=1, description="not applied to biases and 1-d params")
    betas: tuple[float, float] | None = Field(None, description="only for optimizers that have betas")
    momentum: float = Field(0.9, ge=0, lt=1, description="only for optimizers that have momentum")
    grad_clip: float | None = Field(None, gt=0, description="max grad norm; null disables clipping")
    lr_mult: Weights = Field({}, description="param-name pattern -> lr multiplier; first match wins")


class Schedule(Section):
    kind: Literal["pinned", "cosine", "tanh", "step", "poly"] = Field(
        "pinned", description="pinned = lr stays at optimizer.lr; else a timm scheduler shapes a factor"
    )
    total_steps: int = Field(1000, ge=1, description="length of the schedule in steps")
    warmup_steps: int = Field(0, ge=0)
    min_factor: float = Field(0.0, ge=0, le=1, description="floor of the factor multiplying optimizer.lr")
    decay_steps: int = Field(100, ge=1, description="step schedule: steps between decays")
    decay_rate: float = Field(0.1, gt=0, le=1, description="step schedule: factor per decay")
    origin: int | None = Field(None, ge=0, description="step where t=0; null = the step it was applied")


class Freeze(Section):
    frozen: list[str] = Field([], description="param-name patterns that do not train")
    norm: Literal["follow", "train", "eval"] = Field(
        "follow", description="batch-norm mode: follow = eval when all its params are frozen"
    )


class Stage(Section):
    mix: Weights
    steps: int = Field(ge=1)


class Data(Section):
    mix: Weights = Field({"train": 1.0}, description="source name -> weight; each batch is one source")
    queue: list[Stage] = Field([], description="stages of {mix, steps} run in order before mix resumes")
    batch_size: int = Field(64, ge=1, le=4096)
    accumulation: int = Field(1, ge=1, le=64, description="batches per optimizer step")
    resolution: int = Field(224, ge=16, le=1024)
    mixup: float = Field(0.0, ge=0, le=2, description="mixup alpha")
    cutmix: float = Field(0.0, ge=0, le=2, description="cutmix alpha")
    randaug: int | None = Field(None, ge=0, le=30, description="RandAugment magnitude; null = off")
    label_smoothing: float = Field(0.0, ge=0, le=0.5)
    drop_path: float = Field(0.0, ge=0, le=0.9, description="max stochastic-depth rate")
    workers: int = Field(4, ge=0, le=64)

    @model_validator(mode="after")
    def _weights(self) -> Data:
        for mix in [self.mix, *(s.mix for s in self.queue)]:
            if any(w < 0 for w in mix.values()) or sum(mix.values()) <= 0:
                raise ValueError("mix weights must be >= 0 and sum to > 0")
        if (self.mixup or self.cutmix) and self.batch_size % 2:
            raise ValueError("mixup/cutmix need an even batch_size")
        return self


class Run(Section):
    pause: bool = Field(False, description="hold at the next step boundary until false")
    stop: bool = Field(False, description="end the run at the next step boundary")
    eval_every: int = Field(0, ge=0, description="steps between evals; 0 = never")
    checkpoint_every: int = Field(0, ge=0, description="steps between checkpoints; 0 = never")
    eval_now: int = Field(0, ge=0, description="change the value to request one eval")
    checkpoint_now: int = Field(0, ge=0, description="change the value to request one checkpoint")


class State(Section):
    version: int = Field(0, description="set by the trainer; +1 per applied change")
    optimizer: Optimizer = Optimizer()
    schedule: Schedule = Schedule()
    freeze: Freeze = Freeze()
    data: Data = Data()
    run: Run = Run()
    extra: dict[str, Any] = Field({}, description="free-form values for your own hooks")


def matches(name: str, pattern: str) -> bool:
    if pattern.startswith("re:"):
        return re.fullmatch(pattern[3:], name) is not None
    return fnmatch.fnmatchcase(name, pattern) or fnmatch.fnmatchcase(name, pattern + ".*")


def apply_patch(state: State, patch: dict[str, Any]) -> State:
    """Return a new validated State. Keys are dotted paths; a trailing + or - adds to / removes from a list."""
    if not isinstance(patch, dict) or not patch:
        raise ValueError("patch must be a non-empty object of dotted path -> value")
    doc = state.model_dump(mode="json")
    for key, value in patch.items():
        op = key[-1] if key[-1] in "+-" else ""
        *parents, leaf = key.rstrip("+-").split(".")
        if not parents and leaf == "version":
            raise ValueError("version is set by the trainer")
        node = doc
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
            if not isinstance(node, dict):
                raise ValueError(f"{key}: no such path")
        if op:
            items = value if isinstance(value, list) else [value]
            if not isinstance(node.get(leaf), list):
                raise ValueError(f"{key}: not a list")
            keep = [v for v in node[leaf] if v not in items]
            node[leaf] = keep + items if op == "+" else keep
        else:
            node[leaf] = value
    try:
        return State.model_validate(doc)
    except ValidationError as e:
        raise ValueError(
            "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
        ) from None


def parse_assignments(pairs: list[str]) -> dict[str, Any]:
    """["optimizer.lr=3e-4", "freeze.frozen+=layer1"] -> patch. Values are JSON, else plain strings."""
    patch: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise ValueError(f"{pair!r}: expected path=value")
        try:
            patch[key] = json.loads(raw)
        except json.JSONDecodeError:
            patch[key] = raw
    return patch


def flatten(doc: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(doc, dict):
        return {prefix: doc}
    out: dict[str, Any] = {}
    for k, v in doc.items():
        out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    return out


def diff(old: State, new: State) -> dict[str, list[Any]]:
    a, b = flatten(old.model_dump(mode="json")), flatten(new.model_dump(mode="json"))
    return {
        k: [a.get(k), b.get(k)]
        for k in sorted(a.keys() | b.keys())
        if a.get(k) != b.get(k) and k != "version"
    }


def touches(changed: dict[str, Any], prefix: str) -> bool:
    return any(k == prefix or k.startswith(prefix + ".") or prefix.startswith(k + ".") for k in changed)


def schema() -> list[dict[str, Any]]:
    rows = []
    for section, info in State.model_fields.items():
        model = info.annotation
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            continue
        for name, f in model.model_fields.items():
            bounds = {a: getattr(m, a) for m in f.metadata for a in ("gt", "ge", "lt", "le") if hasattr(m, a)}
            kind = str(f.annotation).replace("typing.", "").replace("steerable_timm.state.", "")
            kind = kind.removeprefix("<class '").removesuffix("'>")
            rows.append(
                {
                    "path": f"{section}.{name}",
                    "type": kind,
                    **bounds,
                    "default": f.default,
                    "doc": f.description,
                }
            )
    return rows
