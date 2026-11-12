import sys
from pathlib import Path

import pytest
import timm
import torch
from lightning.fabric import Fabric
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
torch.set_num_threads(2)


def make_sources(n: int = 64, classes: int = 10):
    def source(offset: int):
        def build(cfg):
            g = torch.Generator().manual_seed(offset)
            x = torch.randn(n, 3, cfg.resolution, cfg.resolution, generator=g)
            return TensorDataset(x, torch.randint(0, classes, (n,), generator=g))

        return build

    return {"train": source(1), "hard": source(2)}


def make_module(seed: int = 0):
    torch.manual_seed(seed)
    return timm.create_model("test_resnet", num_classes=10, drop_path_rate=0.1)


BASE = {"data.batch_size": 8, "data.resolution": 32, "data.workers": 0}


@pytest.fixture
def fabric():
    f = Fabric(accelerator="cpu", devices=1)
    f.launch()
    return f


@pytest.fixture
def rig(fabric, tmp_path):
    from steerable_timm import CommandLog, Live, State, Steer, apply_patch, builtin

    state = apply_patch(State(), BASE)
    live = Live(fabric, make_module(), make_sources(), num_classes=10)
    builtin.apply(live, None, state, 0)
    log = CommandLog(tmp_path)
    return log, live, Steer(log, live, state, builtin.copy())
