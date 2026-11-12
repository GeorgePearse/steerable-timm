from __future__ import annotations

import itertools
import random
import threading
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from timm.data import Mixup
from timm.layers import DropPath
from timm.optim import create_optimizer_v2, get_optimizer_info, list_optimizers
from timm.scheduler import create_scheduler_v2
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader, Dataset

from .state import Data, Optimizer, State, matches
from .steer import Hooks

LOADER_FIELDS = ("batch_size", "resolution", "randaug", "workers")


@dataclass
class Live:
    """The live objects the hooks act on. The loop reads model/optimizer/feeder/mixup from here every step."""

    fabric: Any
    module: torch.nn.Module
    sources: dict[str, Callable[[Data], Dataset]]
    num_classes: int
    seed: int = 0
    model: Any = None
    optimizer: torch.optim.Optimizer | None = None
    stepper: Any = None
    schedule: Any = None
    schedule_origin: int = 0
    feeder: Feeder | None = None
    mixup: Mixup | None = None
    requests: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


builtin = Hooks()


@builtin.check
def _check(live: Live, state: State) -> None:
    if state.optimizer.name not in list_optimizers():
        raise ValueError(f"optimizer.name: unknown timm optimizer {state.optimizer.name!r}")
    names = {n for mix in [state.data.mix, *(s.mix for s in state.data.queue)] for n in mix}
    if unknown := names - live.sources.keys():
        raise ValueError(f"data: unknown source {sorted(unknown)}; have {sorted(live.sources)}")
    if all(is_frozen(n, state) for n, _ in live.module.named_parameters()):
        raise ValueError("freeze.frozen: at least one parameter must stay trainable")
    if state.data.drop_path and not any(isinstance(m, DropPath) for m in live.module.modules()):
        raise ValueError("data.drop_path: model has no DropPath modules (create it with drop_path_rate > 0)")


def is_frozen(name: str, state: State) -> bool:
    return any(matches(name, p) for p in state.freeze.frozen)


def train_mode(live: Live, state: State) -> None:
    """model.train(), then put batch-norm layers in the mode freeze.norm asks for. Call after every eval."""
    live.module.train()
    if state.freeze.norm == "train":
        return
    for m in live.module.modules():
        if isinstance(m, _BatchNorm) and (
            state.freeze.norm == "eval" or not any(p.requires_grad for p in m.parameters())
        ):
            m.eval()


@builtin.on_change("freeze")
def freeze(live: Live, old: State | None, new: State, step: int) -> None:
    for name, p in live.module.named_parameters():
        p.requires_grad_(not is_frozen(name, new))
    if live.model is None or live.fabric.world_size > 1:
        live.model = live.fabric.setup_module(live.module)
    train_mode(live, new)


def param_groups(module: torch.nn.Module, opt: Optimizer) -> list[dict[str, Any]]:
    groups: dict[tuple[float, bool], dict[str, Any]] = {}
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        scale = next((v for k, v in opt.lr_mult.items() if matches(name, k)), 1.0)
        decay = p.ndim > 1 and not name.endswith(".bias")
        group = groups.setdefault(
            (scale, decay),
            {
                "params": [],
                "lr_scale": scale,
                "decay": decay,
                "weight_decay": opt.weight_decay if decay else 0.0,
            },
        )
        group["params"].append(p)
    return list(groups.values())


@builtin.on_change("freeze.frozen", "optimizer.name", "optimizer.lr_mult")
def rebuild_optimizer(live: Live, old: State | None, new: State, step: int) -> None:
    """New optimizer over the trainable params. Per-param state survives when the optimizer type is unchanged;
    a type change resets all moment buffers, and newly unfrozen params always start with fresh state."""
    o = new.optimizer
    kwargs = {"betas": o.betas} if o.betas and get_optimizer_info(o.name).has_betas else {}
    optimizer = create_optimizer_v2(
        param_groups(live.module, o),
        opt=o.name,
        lr=o.lr,
        weight_decay=o.weight_decay,
        momentum=o.momentum,
        **kwargs,
    )
    if live.optimizer is not None and type(live.optimizer) is type(optimizer):
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p in live.optimizer.state:
                    optimizer.state[p] = live.optimizer.state[p]
    live.module.zero_grad(set_to_none=True)
    live.optimizer = optimizer
    live.stepper = live.fabric.setup_optimizers(optimizer)


@builtin.on_change("optimizer.weight_decay", "optimizer.betas", "optimizer.momentum")
def optimizer_in_place(live: Live, old: State | None, new: State, step: int) -> None:
    o = new.optimizer
    for group in live.optimizer.param_groups:
        if group["decay"]:
            group["weight_decay"] = o.weight_decay
        if o.betas and "betas" in group:
            group["betas"] = tuple(o.betas)
        if "momentum" in group:
            group["momentum"] = o.momentum


@builtin.on_change("schedule")
def schedule(live: Live, old: State | None, new: State, step: int) -> None:
    """A timm scheduler over a dummy lr of 1.0 yields a factor; set_lr multiplies it into every group."""
    s = new.schedule
    live.schedule_origin = step if s.origin is None else s.origin
    live.schedule = None
    if s.kind != "pinned":
        dummy = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        live.schedule, _ = create_scheduler_v2(
            dummy,
            sched=s.kind,
            num_epochs=s.total_steps,
            warmup_epochs=s.warmup_steps,
            warmup_lr=0.0,
            min_lr=s.min_factor,
            decay_epochs=s.decay_steps,
            decay_rate=s.decay_rate,
        )


def set_lr(live: Live, state: State, step: int) -> list[float]:
    """Call every step: lr = optimizer.lr * group multiplier * schedule factor. Returns the lr per group."""
    factor = 1.0
    if live.schedule is not None:
        live.schedule.step(max(0, step - live.schedule_origin))
        factor = live.schedule.optimizer.param_groups[0]["lr"]
    for group in live.optimizer.param_groups:
        group["lr"] = state.optimizer.lr * group["lr_scale"] * factor
    return [g["lr"] for g in live.optimizer.param_groups]


@builtin.on_change("data.mixup", "data.cutmix")
def mixup(live: Live, old: State | None, new: State, step: int) -> None:
    d = new.data
    live.mixup = None
    if d.mixup or d.cutmix:
        live.mixup = Mixup(
            mixup_alpha=d.mixup, cutmix_alpha=d.cutmix, label_smoothing=0.0, num_classes=live.num_classes
        )


@builtin.on_change("data.drop_path")
def drop_path(live: Live, old: State | None, new: State, step: int) -> None:
    mods = [m for m in live.module.modules() if isinstance(m, DropPath)]
    for i, m in enumerate(mods):
        m.drop_prob = new.data.drop_path * (i + 1) / len(mods)


@builtin.on_change("data.mix", "data.queue", *(f"data.{f}" for f in LOADER_FIELDS))
def data(live: Live, old: State | None, new: State, step: int) -> None:
    live.feeder = live.feeder or Feeder(live)
    live.feeder.request(new.data, step)


@builtin.on_change("run.eval_now", "run.checkpoint_now")
def run_requests(live: Live, old: State | None, new: State, step: int) -> None:
    if old is not None and new.run.eval_now != old.run.eval_now:
        live.requests.add("eval")
    if old is not None and new.run.checkpoint_now != old.run.checkpoint_now:
        live.requests.add("checkpoint")


def due(live: Live, state: State, step: int) -> set[str]:
    """What the loop should do after finishing `step` steps: requested once, or periodic."""
    out, live.requests = live.requests, set()
    for what, every in (("eval", state.run.eval_every), ("checkpoint", state.run.checkpoint_every)):
        if every and step % every == 0:
            out.add(what)
    return out


class Feeder:
    """One infinite iterator per data source. Each step draws one source by weight (same draw on every rank).
    Loader changes are built in a background thread and switched in at a step boundary once every rank is ready."""

    def __init__(self, live: Live):
        self.live = live
        self.cfg: Data | None = None
        self.iters: dict[str, Iterator[Any]] = {}
        self.origin = 0
        self.pending: tuple[Data, dict[str, Any], threading.Thread | None] | None = None

    def request(self, cfg: Data, step: int) -> None:
        need = {n for mix in [cfg.mix, *(s.mix for s in cfg.queue)] for n, w in mix.items() if w > 0}
        same = self.cfg is not None and all(getattr(self.cfg, f) == getattr(cfg, f) for f in LOADER_FIELDS)
        box: dict[str, Any] = {n: it for n, it in self.iters.items() if same and n in need}
        missing = need - box.keys()

        def build() -> None:
            try:
                box.update({n: self._iterator(n, cfg) for n in missing})
            except Exception as e:
                box["error"] = e

        thread = None
        if missing and self.cfg is not None:
            thread = threading.Thread(target=build, daemon=True)
            thread.start()
        else:
            build()
        self.pending = (cfg, box, thread)
        if thread is None:
            self._switch(step)

    def _iterator(self, name: str, cfg: Data) -> Iterator[Any]:
        loader = DataLoader(
            self.live.sources[name](cfg),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.workers,
            persistent_workers=cfg.workers > 0,
            generator=torch.Generator().manual_seed(self.live.seed + zlib.crc32(name.encode())),
        )
        if len(loader) == 0:
            raise ValueError(f"source {name!r} has fewer samples than batch_size {cfg.batch_size}")
        loader = self.live.fabric.setup_dataloaders(loader)
        it = (batch for _ in itertools.count() for batch in loader)
        return itertools.chain([next(it)], it)

    def _ready(self) -> bool:
        thread = self.pending[2]
        done = thread is None or not thread.is_alive()
        if self.live.fabric.world_size > 1:
            done = bool(self.live.fabric.all_reduce(torch.tensor(float(done)), reduce_op="mean") == 1)
        return done

    def _switch(self, step: int) -> None:
        cfg, box, _ = self.pending
        self.pending = None
        if "error" in box:
            if self.cfg is None:
                raise box["error"]
            self.live.notes.append(f"data change dropped, loaders failed to build: {box['error']!r}")
            return
        if self.cfg is None or cfg.queue != self.cfg.queue:
            self.origin = step
        if self.cfg is not None and any(box[n] is not self.iters.get(n) for n in box):
            fields = " ".join(f"{f}={getattr(cfg, f)}" for f in LOADER_FIELDS)
            self.live.notes.append(f"loaders switched at step {step}: {fields} sources={sorted(box)}")
        self.cfg, self.iters = cfg, box

    def mix(self, step: int) -> dict[str, float]:
        t = step - self.origin
        for stage in self.cfg.queue:
            if t < stage.steps:
                return stage.mix
            t -= stage.steps
        return self.cfg.mix

    def next(self, step: int) -> tuple[str, Any]:
        if self.pending is not None and self._ready():
            self._switch(step)
        mix = self.mix(step)
        names = sorted(n for n, w in mix.items() if w > 0)
        name = random.Random(self.live.seed * 1_000_003 + step).choices(names, [mix[n] for n in names])[0]
        return name, next(self.iters[name])
