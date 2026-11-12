"""A Lightning Fabric loop on a timm model that can be steered while it trains.

    python examples/train.py --run runs/demo --steer --steps 3000 \
        --set optimizer.lr=1e-3 --set 'freeze.frozen=["conv1","bn1","layer1","layer2","layer3","layer4"]'
    steer --run runs/demo unfreeze layer4 --why "head has plateaued"
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from timm.data import create_transform
from torch.utils.data import Dataset, Subset

from steerable_timm import (
    CommandLog,
    Live,
    State,
    Steer,
    apply_patch,
    builtin,
    due,
    parse_assignments,
    set_lr,
    train_mode,
)

HARD = (
    "maple_tree",
    "oak_tree",
    "palm_tree",
    "pine_tree",
    "willow_tree",
    "baby",
    "boy",
    "girl",
    "man",
    "woman",
)


def fit(
    fabric: Fabric,
    module: torch.nn.Module,
    sources: dict[str, Callable[..., Dataset]],
    state: State,
    *,
    steps: int,
    run_dir: str | Path,
    num_classes: int,
    steer: bool = True,
    evaluate: Callable[[torch.nn.Module], dict[str, float]] | None = None,
    log_every: int = 10,
    seed: int = 0,
    resume: str | None = None,
    hooks=builtin,
) -> list[float]:
    live = Live(fabric, module, sources, num_classes, seed)
    start = 0
    if resume:
        ckpt = fabric.load(resume)
        state, start = State.model_validate(ckpt["state"]), ckpt["step"]
        module.load_state_dict(ckpt["model"])
    hooks.validate(live, state)
    hooks.apply(live, None, state, start)
    if resume:
        live.optimizer.load_state_dict(ckpt["optimizer"])
    ctl = Steer(CommandLog(run_dir), live, state, hooks, step=start) if steer else None
    losses: list[float] = []
    window: list[float] = []
    grad_norm, clock, samples = 0.0, time.perf_counter(), 0

    def save(step: int, name: str) -> None:
        doc = {
            "model": module,
            "optimizer": live.optimizer,
            "step": step,
            "state": state.model_dump(mode="json"),
        }
        fabric.save(Path(run_dir) / name, doc)

    for step in range(start, steps):
        if ctl:
            state = ctl.sync(step)
        if state.run.stop:
            break
        lrs = set_lr(live, state, step)
        source, (x, y) = live.feeder.next(step)
        if live.mixup is not None:
            x, y = live.mixup(x, y)
        loss = F.cross_entropy(live.model(x), y, label_smoothing=state.data.label_smoothing)
        fabric.backward(loss / state.data.accumulation)
        if (step + 1) % state.data.accumulation == 0:
            clip = state.optimizer.grad_clip or float("inf")
            grad_norm = float(
                fabric.clip_gradients(live.model, live.stepper, max_norm=clip, error_if_nonfinite=False)
            )
            live.stepper.step()
            live.optimizer.zero_grad()
        losses.append(loss.item())
        window.append(losses[-1])
        samples += len(x) * fabric.world_size

        if (step + 1) % log_every == 0 and fabric.global_rank == 0:
            params = list(module.parameters())
            row = {
                "loss": sum(window) / len(window),
                "lr": lrs,
                "grad_norm": grad_norm,
                "frozen": sum(p.numel() for p in params if not p.requires_grad)
                / sum(p.numel() for p in params),
                "samples_per_s": samples / (time.perf_counter() - clock),
                "source": source,
            }
            print(
                f"step {step + 1} "
                + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
            )
            if ctl:
                ctl.metrics(step + 1, **row)
            window, clock, samples = [], time.perf_counter(), 0
        while ctl and live.notes:
            ctl.metrics(step + 1, kind="note", text=live.notes.pop(0))
        for what in sorted(due(live, state, step + 1)):
            if what == "checkpoint":
                save(step + 1, f"step{step + 1}.ckpt")
            elif evaluate is not None and fabric.global_rank == 0:
                result = evaluate(module)
                train_mode(live, state)
                print(f"step {step + 1} eval {result}")
                if ctl:
                    ctl.metrics(step + 1, kind="eval", **result)
    save(len(losses) + start, "final.ckpt")
    if ctl:
        ctl.flush()
    return losses


def cifar100(
    root: str, resolution: int, fabric: Fabric, eval_batch: int = 500
) -> tuple[dict[str, Any], Callable]:
    from torchvision.datasets import CIFAR100

    train = CIFAR100(root, train=True, download=True)
    test = CIFAR100(root, train=False, download=True, transform=create_transform(resolution, crop_pct=1.0))
    hard = {train.class_to_idx[c] for c in HARD}
    hard_rows = [i for i, t in enumerate(train.targets) if t in hard]

    def source(rows: list[int] | None) -> Callable[..., Dataset]:
        def build(cfg) -> Dataset:
            aug = f"rand-m{cfg.randaug}-mstd0.5-inc1" if cfg.randaug is not None else None
            tf = create_transform(cfg.resolution, is_training=True, auto_augment=aug, scale=(0.35, 1.0))
            ds = CIFAR100(root, train=True, transform=tf)
            return ds if rows is None else Subset(ds, rows)

        return build

    loader = fabric.setup_dataloaders(
        torch.utils.data.DataLoader(test, batch_size=eval_batch, num_workers=4), use_distributed_sampler=False
    )
    is_hard = torch.zeros(100, dtype=torch.bool)
    is_hard[list(hard)] = True

    @torch.no_grad()
    def evaluate(module: torch.nn.Module) -> dict[str, float]:
        module.eval()
        hits, targets = [], []
        for x, y in loader:
            hits.append(module(x).argmax(1) == y)
            targets.append(y)
        hit, mask = torch.cat(hits).float().cpu(), is_hard[torch.cat(targets).cpu()]
        return {"top1": hit.mean().item(), "top1_hard": hit[mask].mean().item()}

    return {"train": source(None), "hard": source(hard_rows)}, evaluate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="runs/demo")
    p.add_argument("--model", default="resnet18.a1_in1k")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--steer", action="store_true", help="read steer.db every step")
    p.add_argument(
        "--set", action="append", default=[], metavar="PATH=VALUE", help="initial State, steer-set syntax"
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--eval-resolution", type=int, default=96)
    p.add_argument("--devices", default="auto")
    p.add_argument("--precision", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume")
    args = p.parse_args()

    fabric = Fabric(devices=args.devices, precision=args.precision)
    fabric.launch()
    fabric.seed_everything(args.seed)
    base = {
        "data.resolution": 96,
        "data.batch_size": 128,
        "run.eval_every": 250,
        "run.checkpoint_every": 1000,
    }
    state = apply_patch(State(), {**base, **parse_assignments(args.set)})
    module = timm.create_model(
        args.model, pretrained=not args.no_pretrained, num_classes=100, drop_path_rate=0.1
    )
    sources, evaluate = cifar100(args.data_dir, args.eval_resolution, fabric)
    fit(
        fabric,
        module,
        sources,
        state,
        steps=args.steps,
        run_dir=args.run,
        num_classes=100,
        steer=args.steer,
        evaluate=evaluate,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
