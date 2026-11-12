"""Two-process CPU DDP run steered mid-flight. Each rank writes a trace; test_ddp.py compares them."""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from conftest import BASE, make_module, make_sources
from lightning.fabric import Fabric

from steerable_timm import CommandLog, Live, State, Steer, apply_patch, builtin, set_lr

COMMANDS = {
    2: {"freeze.frozen": ["layer1", "layer2", "layer3", "layer4"], "optimizer.lr": 2e-3},
    5: {"freeze.frozen": ["layer1"], "optimizer.lr_mult": {"fc": 5.0}},
    8: {"data.batch_size": 4, "data.mix": {"train": 1, "hard": 1}},
    11: {"optimizer.name": "sgd", "optimizer.lr": 0.01},
    14: {"freeze.frozen": [], "data.batch_size": 9999},
}


def main(run_dir: Path) -> None:
    fabric = Fabric(accelerator="cpu", devices=2, strategy="ddp")
    fabric.launch()
    torch.set_num_threads(1)
    state = apply_patch(State(), BASE)
    live = Live(fabric, make_module(), make_sources(), num_classes=10)
    builtin.apply(live, None, state, 0)
    steer = Steer(CommandLog(run_dir), live, state, builtin)
    writer = CommandLog(run_dir) if fabric.global_rank == 0 else None
    trace = []
    for step in range(18):
        state = steer.sync(step)
        lrs = set_lr(live, state, step)
        source, (x, y) = live.feeder.next(step)
        if live.feeder.pending is not None:
            live.feeder.pending[2].join()
        fabric.backward(F.cross_entropy(live.model(x), y))
        live.stepper.step()
        live.optimizer.zero_grad()
        trainable = sum(p.requires_grad for p in live.module.parameters())
        trace.append([step, state.version, lrs, source, len(x), trainable, type(live.optimizer).__name__])
        if writer and step in COMMANDS:
            writer.set(COMMANDS[step], author="test", check=False)
    checksum = [float(p.double().sum()) for p in live.module.parameters()]
    out = {"trace": trace, "checksum": checksum, "sample": x.flatten()[:4].tolist()}
    (run_dir / f"rank{fabric.global_rank}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
