import threading
import time

import pytest
import torch
from conftest import BASE, make_module, make_sources
from timm.layers import DropPath
from torch.nn.modules.batchnorm import _BatchNorm

from steerable_timm import CommandLog, State, apply_patch, diff, due, schema, set_lr
from steerable_timm.cli import main as cli


def test_patch_validation_and_list_ops():
    s = apply_patch(State(), {"optimizer.lr": 3e-4, "freeze.frozen+": ["layer1", "layer2"]})
    assert s.optimizer.lr == 3e-4 and s.freeze.frozen == ["layer1", "layer2"]
    s2 = apply_patch(s, {"freeze.frozen-": "layer1", "data.mix.hard": 0.5})
    assert s2.freeze.frozen == ["layer2"] and s2.data.mix == {"train": 1.0, "hard": 0.5}
    assert diff(s, s2) == {"data.mix.hard": [None, 0.5], "freeze.frozen": [["layer1", "layer2"], ["layer2"]]}
    for bad in (
        {"optimizer.lr": -1},
        {"optimizer.lr": 99},
        {"optimizer.nope": 1},
        {"nope.x": 1},
        {"version": 9},
        {"data.mix": {"train": 0}},
        {"data.batch_size": "big"},
        {"schedule.kind": "magic"},
        {},
    ):
        with pytest.raises(ValueError):
            apply_patch(State(), bad)
    rows = {r["path"]: r for r in schema()}
    assert rows["optimizer.lr"]["gt"] == 0 and rows["optimizer.lr"]["le"] == 10


def test_coalesce_reject_and_fast_path(rig):
    log, live, steer = rig
    calls = []
    steer.hooks.on_change("optimizer")(lambda live, old, new, step: calls.append(step))
    assert steer.sync(0).version == 0
    a = log.set({"optimizer.lr": 5e-4}, why="first", author="llm")
    b = log.set({"optimizer.lr": 1e9}, why="too big", check=False)
    c = log.set({"optimizer.weight_decay": 0.1, "optimizer.lr": 2e-4}, why="third")
    d = log.set({"data.mix": {"nowhere": 1}}, check=False)
    state = steer.sync(7)
    assert (state.version, state.optimizer.lr, state.optimizer.weight_decay) == (1, 2e-4, 0.1)
    assert calls == [7]
    assert [log.command(i)["status"] for i in (a, b, c, d)] == ["applied", "rejected", "applied", "rejected"]
    assert "unknown source" in log.command(d)["error"]
    changes = log.log()
    assert [r["step"] for r in changes] == [7] * 4 and changes[0]["reason"] == "first"
    assert changes[2]["diff"] == {"optimizer.lr": [5e-4, 2e-4], "optimizer.weight_decay": [0.05, 0.1]}
    assert log.get()["version"] == 1

    reads = []
    log.pending = lambda: reads.append(1) or []
    for step in range(8, 200):
        steer.sync(step)
    assert reads == [] and steer.state.version == 1


def test_rejected_by_client_before_queueing(rig):
    log, _, steer = rig
    with pytest.raises(ValueError):
        log.set({"optimizer.lr": -3})
    assert log.head() == 0


def test_hook_failure_rolls_back(rig):
    log, live, steer = rig

    @steer.hooks.on_change("extra.boom")
    def boom(live, old, new, step):
        if new.extra.get("boom"):
            raise RuntimeError("no")

    c = log.set({"extra.boom": True, "optimizer.weight_decay": 0.3})
    state = steer.sync(3)
    assert state.version == 0 and state.optimizer.weight_decay == 0.05
    assert all(g["weight_decay"] in (0.0, 0.05) for g in live.optimizer.param_groups)
    assert log.command(c)["status"] == "rejected" and "rolled back" in log.command(c)["error"]


def test_optimizer_in_place_and_lr_groups(rig):
    log, live, steer = rig
    before = live.optimizer
    log.set({"optimizer.lr": 2e-3, "optimizer.weight_decay": 0.2, "optimizer.betas": [0.8, 0.9]})
    state = steer.sync(1)
    assert live.optimizer is before
    assert set(set_lr(live, state, 1)) == {2e-3}
    assert {g["weight_decay"] for g in live.optimizer.param_groups} == {0.0, 0.2}
    assert all(g["betas"] == (0.8, 0.9) for g in live.optimizer.param_groups)
    log.set({"optimizer.lr_mult": {"fc": 10.0}})
    state = steer.sync(2)
    lrs = {
        g["lr_scale"]: lr for g, lr in zip(live.optimizer.param_groups, set_lr(live, state, 2), strict=True)
    }
    assert lrs == {1.0: 2e-3, 10.0: 2e-2}
    fc = [g for g in live.optimizer.param_groups if g["lr_scale"] == 10.0]
    assert sum(len(g["params"]) for g in fc) == 2


def _step(live):
    x = torch.randn(4, 3, 32, 32)
    live.model(x).sum().backward()
    live.stepper.step()
    live.optimizer.zero_grad()


def test_optimizer_swap_and_state_carry_over(rig):
    log, live, steer = rig
    _step(live)
    fc_w = live.module.fc.weight
    moment = live.optimizer.state[fc_w]["exp_avg"].clone()
    log.set({"freeze.frozen": ["layer1"]})
    steer.sync(1)
    assert type(live.optimizer).__name__ == "AdamW"
    assert torch.equal(live.optimizer.state[fc_w]["exp_avg"], moment)
    log.set({"optimizer.name": "sgd", "optimizer.momentum": 0.8})
    steer.sync(2)
    assert type(live.optimizer).__name__ == "SGD" and len(live.optimizer.state) == 0
    assert all(g["momentum"] == 0.8 for g in live.optimizer.param_groups)
    _step(live)
    c = log.set({"optimizer.name": "not_an_optimizer"}, check=False)
    steer.sync(3)
    assert log.command(c)["status"] == "rejected" and type(live.optimizer).__name__ == "SGD"


def test_freeze_param_groups_and_norm_mode(rig):
    log, live, steer = rig
    log.set({"freeze.frozen": ["conv1", "bn1", "layer[1-3]", "re:layer4\\..*bn\\d\\.(weight|bias)"]})
    steer.sync(1)
    named = dict(live.module.named_parameters())
    frozen = {n for n, p in named.items() if not p.requires_grad}
    assert frozen and all(
        n.startswith(("conv1", "bn1", "layer1", "layer2", "layer3")) or "bn" in n for n in frozen
    )
    assert all(not n.startswith("fc") for n in frozen)
    in_opt = [id(p) for g in live.optimizer.param_groups for p in g["params"]]
    assert len(in_opt) == len(set(in_opt))
    assert set(in_opt) == {id(p) for p in named.values() if p.requires_grad}
    for g in live.optimizer.param_groups:
        assert all((p.ndim > 1) == g["decay"] for p in g["params"])
    bns = {n: m for n, m in live.module.named_modules() if isinstance(m, _BatchNorm)}
    assert not bns["layer1.0.bn1"].training and not bns["layer4.0.bn1"].training
    assert live.module.fc.training

    log.set({"freeze.norm": "train"})
    steer.sync(2)
    assert all(m.training for m in bns.values())
    log.set({"freeze.frozen-": ["layer[1-3]"], "freeze.norm": "follow"})
    steer.sync(3)
    assert named["layer2.0.conv1.weight"].requires_grad and not named["conv1.0.weight"].requires_grad
    assert bns["layer1.0.bn1"].training and not bns["bn1"].training
    c = log.set({"freeze.frozen": ["*"]}, check=False)
    steer.sync(4)
    assert "must stay trainable" in log.command(c)["error"]


def test_schedule_replace(rig):
    log, live, steer = rig
    assert set_lr(live, steer.state, 50) == [1e-3] * len(live.optimizer.param_groups)
    log.set(
        {
            "schedule.kind": "cosine",
            "schedule.total_steps": 100,
            "schedule.warmup_steps": 10,
            "optimizer.lr": 1.0,
        }
    )
    state = steer.sync(200)
    lr = lambda step: set_lr(live, state, step)[0]  # noqa: E731
    assert lr(200) == 0.0 and lr(205) == pytest.approx(0.5) and lr(210) == pytest.approx(1.0, abs=0.03)
    assert lr(250) == pytest.approx(0.5, abs=0.02) and lr(400) == pytest.approx(0.0, abs=1e-6)
    log.set(
        {
            "schedule.kind": "step",
            "schedule.decay_steps": 10,
            "schedule.decay_rate": 0.5,
            "schedule.origin": 0,
        }
    )
    state = steer.sync(300)
    assert lr(25) == pytest.approx(0.25)
    log.set({"schedule.kind": "pinned"})
    state = steer.sync(301)
    assert lr(999) == 1.0


def test_loader_swap_mix_and_queue(rig):
    log, live, steer = rig
    feeder = live.feeder
    name, (x, y) = feeder.next(0)
    assert name == "train" and x.shape == (8, 3, 32, 32)
    log.set({"data.batch_size": 4, "data.resolution": 48, "data.mix": {"train": 1, "hard": 1}})
    steer.sync(1)
    assert feeder.pending is not None
    feeder.pending[2].join()
    name, (x, y) = feeder.next(1)
    assert x.shape == (4, 3, 48, 48) and feeder.pending is None
    assert any("loaders switched" in n for n in live.notes)
    drawn = {feeder.next(s)[0] for s in range(2, 40)}
    assert drawn == {"train", "hard"}

    iters = dict(feeder.iters)
    log.set({"data.mix": {"hard": 1}})
    steer.sync(40)
    assert feeder.pending is None and feeder.iters["hard"] is iters["hard"]
    assert {feeder.next(s)[0] for s in range(40, 50)} == {"hard"}
    log.set({"data.queue": [{"mix": {"train": 1}, "steps": 5}], "data.label_smoothing": 0.1})
    steer.sync(50)
    feeder.pending[2].join()
    assert [feeder.next(s)[0] for s in range(50, 58)] == ["train"] * 5 + ["hard"] * 3
    live.notes.clear()
    log.set({"data.randaug": 5})
    steer.sync(60)
    feeder.pending[2].join()
    feeder.next(60)
    assert live.notes == [
        "loaders switched at step 60: batch_size=4 resolution=48 randaug=5 workers=0 sources=['hard', 'train']"
    ]


def test_mixup_and_drop_path(rig):
    log, live, steer = rig
    assert live.mixup is None
    log.set({"data.mixup": 0.4, "data.cutmix": 1.0, "data.drop_path": 0.3})
    steer.sync(1)
    x, y = live.mixup(*live.feeder.next(1)[1])
    assert y.shape == (8, 10)
    probs = [m.drop_prob for m in live.module.modules() if isinstance(m, DropPath)]
    assert probs == sorted(probs) and probs[-1] == pytest.approx(0.3) and probs[0] > 0
    c = log.set({"data.batch_size": 7}, check=False)
    steer.sync(2)
    assert "even batch_size" in log.command(c)["error"]


def test_run_control(rig):
    log, live, steer = rig
    assert due(live, steer.state, 10) == set()
    log.set({"run.eval_every": 5, "run.checkpoint_now": 1})
    state = steer.sync(1)
    assert (
        due(live, state, 3) == {"checkpoint"}
        and due(live, state, 10) == {"eval"}
        and due(live, state, 11) == set()
    )

    log.set({"run.pause": True})
    threading.Timer(0.5, lambda: CommandLog(log.path.parent).set({"run.pause": False}, why="resume")).start()
    t = time.time()
    state = steer.sync(2)
    assert time.time() - t >= 0.45 and not state.run.pause and state.version == 3
    log.set({"run.stop": True})
    assert steer.sync(3).run.stop


def test_custom_hook_is_ten_lines(rig):
    log, live, steer = rig
    live.loss_weight = 1.0

    @steer.hooks.on_change("extra.loss_weight")
    def loss_weight(live, old, new, step):
        live.loss_weight = float(new.extra.get("loss_weight", 1.0))

    log.set({"extra.loss_weight": 2.5}, why="weight the auxiliary loss more")
    steer.sync(4)
    assert live.loss_weight == 2.5


def test_cli(rig, capsys):
    log, live, steer = rig
    run = ["--run", str(log.path.parent)]
    done = threading.Event()

    def trainer():
        step = 0
        while not done.is_set():
            steer.sync(step := step + 1)
            time.sleep(0.005)

    worker = threading.Thread(target=trainer)
    worker.start()
    assert cli([*run, "set", "optimizer.lr=3e-4", "--why", "loss is noisy", "--author", "llm"]) == 0
    assert cli([*run, "freeze", "layer[1-2]"]) == 0
    assert cli([*run, "unfreeze", "layer2"]) == 0
    assert cli([*run, "data", "mix", "train=3", "hard=1"]) == 0
    assert cli([*run, "data", "queue", "hard:20"]) == 0
    assert cli([*run, "run", "eval"]) == 0
    assert cli([*run, "set", "optimizer.lr=50"]) == 1
    assert cli([*run, "set", "optimizer.name=nope"]) == 1
    done.set()
    worker.join()
    steer.metrics(300, loss=1.5, lr=[3e-4], grad_norm=2.0, frozen=0.1, samples_per_s=100.0, source="train")
    steer.flush()
    state = steer.state
    assert state.optimizer.lr == 3e-4 and state.freeze.frozen == ["layer[1-2]"] and state.run.eval_now == 1
    assert state.data.mix == {"train": 3, "hard": 1} and state.data.queue[0].steps == 20
    capsys.readouterr()
    assert (
        cli([*run, "log", "--last", "3"]) == 0
        and cli([*run, "metrics"]) == 0
        and cli([*run, "get", "optimizer.lr"]) == 0
    )
    out = capsys.readouterr().out
    assert "rejected" in out and "unknown timm optimizer" in out and "300 1.5 [0.0003] 2 0.1 100 train" in out
    assert cli(["schema"]) == 0 and "optimizer.lr" in capsys.readouterr().out


def test_no_commands_is_identical_and_checkpoint_has_version(fabric, tmp_path):
    from train import fit

    def run(steer, sub):
        fabric.seed_everything(0)
        state = apply_patch(
            State(), {**BASE, "data.mixup": 0.2, "schedule.kind": "cosine", "schedule.total_steps": 12}
        )
        return fit(
            fabric,
            make_module(),
            make_sources(),
            state,
            steps=12,
            run_dir=tmp_path / sub,
            num_classes=10,
            steer=steer,
            log_every=4,
        )

    plain, steered = run(False, "a"), run(True, "b")
    assert plain == steered and len(plain) == 12
    assert not (tmp_path / "a" / "steer.db").exists()
    log = CommandLog(tmp_path / "b")
    assert [m["step"] for m in log.metrics()] == [4, 8, 12] and log.log() == []

    log.set({"optimizer.lr": 5e-4, "freeze.frozen": ["layer1"]}, why="queued before start")
    log.set({"run.checkpoint_now": 1})
    fabric.seed_everything(0)
    changed = fit(
        fabric,
        make_module(),
        make_sources(),
        apply_patch(State(), BASE),
        steps=6,
        run_dir=tmp_path / "b",
        num_classes=10,
        log_every=2,
    )
    assert len(changed) == 6
    ckpt = torch.load(tmp_path / "b" / "final.ckpt", weights_only=False)
    assert ckpt["state"]["version"] == 1 and ckpt["state"]["optimizer"]["lr"] == 5e-4 and ckpt["step"] == 6
    assert (tmp_path / "b" / "step1.ckpt").exists()
    resumed = fit(
        fabric,
        make_module(1),
        make_sources(),
        State(),
        steps=9,
        run_dir=tmp_path / "b",
        num_classes=10,
        resume=str(tmp_path / "b" / "final.ckpt"),
    )
    assert len(resumed) == 3 and log.get()["freeze"]["frozen"] == ["layer1"]
