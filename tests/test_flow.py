import pytest
import torch
from conftest import BASE

from steerable_timm import Live, State, apply_patch, grad_flow, modules, set_lr
from steerable_timm.cli import main as cli
from steerable_timm.flow import apply_flow, check_flow


def capture(live, name):
    """Input (cloned before the module runs) and final output of one module, across one forward."""
    module, box = dict(live.module.named_modules())[name], {}
    handles = [
        module.register_forward_pre_hook(lambda m, args: box.__setitem__("x", args[0].clone())),
        module.register_forward_hook(lambda m, args, out: box.__setitem__("y", out.clone())),
    ]
    return box, handles


def forward(live, x, name):
    box, handles = capture(live, name)
    with torch.no_grad():
        out = live.module(x)
    for h in handles:
        h.remove()
    return box["x"], box["y"], out


def test_taps_change_the_function_as_documented_and_removal_restores_it(rig):
    log, live, steer = rig
    live.module.eval()
    x = torch.randn(4, 3, 32, 32)
    conv_in, conv_out, base = forward(live, x, "layer1.0.conv2")
    relu_in, relu_out, _ = forward(live, x, "layer1.0.act1")
    assert torch.equal(relu_out, relu_in.clamp(min=0)) and not live.taps

    log.set({"flow.skip": {"layer1.0.conv2": 0.5}, "flow.scale": {"layer1.0.conv2": 0.25}})
    steer.sync(1)
    a, b, tapped = forward(live, x, "layer1.0.conv2")
    assert torch.allclose(b, 0.25 * conv_out + 0.5 * a) and torch.equal(a, conv_in)

    log.set({"flow.skip": {"layer1.0.act1": 0.1}, "flow.scale": {}})
    steer.sync(2)
    a, b, _ = forward(live, x, "layer1.0.act1")
    assert torch.allclose(b, a.clamp(min=0) + 0.1 * a), (
        "in-place ReLU: the skip must add the input as it arrived"
    )
    assert (b < 0).any(), "a skip on a ReLU lets negative inputs through, so their gradient is no longer zero"

    log.set({"flow.skip": {}})
    steer.sync(3)
    assert not live.taps and torch.equal(forward(live, x, "layer1.0.conv2")[2], base)
    assert set(live.module.state_dict()) == set(
        dict(live.module.named_parameters()) | dict(live.module.named_buffers())
    )


def test_fade_returns_to_the_original_function(rig):
    log, live, steer = rig
    live.module.eval()
    x = torch.randn(4, 3, 32, 32)
    _, _, base = forward(live, x, "fc")
    log.set(
        {"flow.skip": {"re:layer1\\.0|conv1\\.2": 1.0}, "flow.scale": {"conv1.4": 0.5}, "flow.fade_steps": 10}
    )
    state = steer.sync(20)
    set_lr(live, state, 20)
    full = forward(live, x, "fc")[2]
    set_lr(live, state, 25)
    half = forward(live, x, "fc")[2]
    assert (
        not torch.allclose(full, base) and not torch.allclose(half, full) and not torch.allclose(half, base)
    )
    set_lr(live, state, 29)
    assert not live.notes
    set_lr(live, state, 30)
    assert torch.equal(forward(live, x, "fc")[2], base) and "faded out at step 30" in live.notes.pop()
    set_lr(live, state, 31)
    assert not live.notes and live.taps

    log.set({"flow.origin": 100})
    state = steer.sync(40)
    set_lr(live, state, 40)
    assert torch.allclose(forward(live, x, "fc")[2], full), (
        "an explicit origin in the future holds the taps at full"
    )


def test_drops_act_only_in_training(rig):
    log, live, steer = rig
    x = torch.randn(8, 3, 32, 32)
    live.module.eval()
    _, clean, _ = forward(live, x, "conv1.4")
    log.set({"flow.drop_path": {"conv1.4": 0.5}, "flow.dropout": {"layer1.0.conv1": 0.5}})
    steer.sync(1)
    assert torch.equal(forward(live, x, "conv1.4")[1], clean)
    live.module.train()
    torch.manual_seed(0)
    _, dropped, _ = forward(live, x, "conv1.4")
    per_sample = dropped.flatten(1).abs().sum(1)
    assert 0 < int((per_sample == 0).sum()) < 8, "drop_path zeroes whole samples of the tapped output"
    _, conv, _ = forward(live, x, "layer1.0.conv1")
    zeros = (conv == 0).float().mean().item()
    assert 0.4 < zeros < 0.6


def test_rejections(rig):
    log, live, steer = rig
    for patch, message in (
        ({"flow.skip": {"layer9": 0.5}}, "no module matches 'layer9'"),
        ({"flow.skip": {"layer2.0": 0.5}}, "layer2.0 changes the shape of its input"),
        ({"flow.skip": {"layer1": 0.5, "fc": 0.5}}, "fc changes the shape"),
    ):
        command = log.set(patch)
        steer.sync(1)
        row = log.command(command)
        assert row["status"] == "rejected" and message in row["error"]
    for bad in (
        {"flow.skip": {"layer1": 3}},
        {"flow.scale": {"layer1": -1}},
        {"flow.dropout": {"fc": 0.95}},
        {"flow.fade_steps": -1},
    ):
        with pytest.raises(ValueError):
            apply_patch(State(), bad)
    assert not live.taps and steer.state.version == 0
    command = log.set({"flow.skip": {"layer1": 0.5}})
    steer.sync(2)
    assert log.command(command)["status"] == "applied" and len(live.taps) == 1, (
        "a pattern names modules exactly, not their children"
    )


def test_modules_table_and_cli(rig, capsys):
    log, live, steer = rig
    table = {r["name"]: r for r in modules(live, steer.state)}
    assert table["layer1.0"]["skip"] and not table["layer2.0"]["skip"] and table["fc"]["tap"]
    assert table["layer3.0.bn3"]["params"] == 384
    assert all(m.training for m in live.module.modules()), (
        "the shape probe must leave train mode as it found it"
    )
    steer.metrics(0, kind="modules", rows=list(table.values()))
    steer.metrics(10, loss=1.0, grad_flow={"layer1": 0.5, "fc": 2.0})
    steer.flush()
    run = ["--run", str(log.path.parent), "--wait", "0"]
    assert (
        cli([*run[:2], "flow", "skip", "layer1.0=0.5", "layer3.0.act3=0.1", "--fade", "200", "--wait", "0"])
        == 0
    )
    state = steer.sync(1)
    assert state.flow.skip == {"layer1.0": 0.5, "layer3.0.act3": 0.1} and state.flow.fade_steps == 200
    assert cli([*run[:2], "flow", "clear", "--wait", "0"]) == 0
    assert steer.sync(2).flow.skip == {} and not live.taps
    capsys.readouterr()
    assert cli([*run[:2], "modules", "layer1.*"]) == 0 and cli([*run[:2], "grads"]) == 0
    out = capsys.readouterr().out
    assert (
        "layer1.0 BasicBlock 18560 yes yes" in out
        and "layer2.0" not in out
        and "step layer1 fc\n10 0.5 2" in out
    )


class Plain(torch.nn.Module):
    """A deep stack with no residual connections and a saturating activation: gradient dies on the way down."""

    def __init__(self, depth: int = 30, width: int = 32):
        super().__init__()
        self.stem = torch.nn.Linear(8, width)
        self.blocks = torch.nn.Sequential(
            *(torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.Sigmoid()) for _ in range(depth))
        )
        self.head = torch.nn.Linear(width, 2)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def test_a_skip_restores_gradient_to_the_bottom_of_a_plain_deep_stack(fabric):
    torch.manual_seed(0)
    module = Plain()
    x, y = torch.randn(16, 8), torch.randint(0, 2, (16,))
    live = Live(fabric, module, sources={}, num_classes=2, example_input=x[:2])

    def stem_grad() -> float:
        module.zero_grad()
        torch.nn.functional.cross_entropy(module(x), y).backward()
        return grad_flow(module)["stem"]

    before = stem_grad()
    state = apply_patch(State(), {**BASE, "flow.skip": {"re:blocks\\.\\d+": 1.0}})
    check_flow(live, state)
    apply_flow(live, state, 0)
    assert len(live.taps) == 30
    with_skips = stem_grad()
    assert before < 1e-12 and with_skips > 1e-3, (before, with_skips)
    apply_flow(live, State(), 1)
    assert stem_grad() == before and grad_flow(module, depth=2)["blocks.0"] < 1e-12
