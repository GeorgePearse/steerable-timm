"""Cost of Steer.sync() per training step: idle, and with another process writing 10 / 100 / 1000 patches per second.

python bench/overhead.py [--steps 400]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from lightning.fabric import Fabric

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from conftest import BASE, make_module, make_sources  # noqa: E402

from steerable_timm import CommandLog, Live, State, Steer, apply_patch, builtin, set_lr  # noqa: E402


def writer(run_dir: str, rate: float, stop) -> None:
    log, n, start = CommandLog(run_dir), 0, time.perf_counter()
    while not stop.is_set():
        log.set({"optimizer.lr": 1e-3 * (1 + (n % 7) / 10)}, why="bench", author="bench", check=False)
        n += 1
        time.sleep(max(0.0, start + n / rate - time.perf_counter()))


def measure(fabric: Fabric, rate: float, steps: int) -> dict[str, float]:
    tmp = tempfile.TemporaryDirectory(prefix="steer-bench-")
    run_dir = tmp.name
    state = apply_patch(State(), BASE)
    live = Live(fabric, make_module(), make_sources(), num_classes=10)
    builtin.apply(live, None, state, 0)
    log = CommandLog(run_dir)
    steer = Steer(log, live, state, builtin)
    stop = mp.Event()
    proc = mp.Process(target=writer, args=(run_dir, rate, stop), daemon=True) if rate else None
    if proc:
        proc.start()
        time.sleep(0.5)
    sync, step_times = [], []
    for step in range(steps):
        t0 = time.perf_counter()
        state = steer.sync(step)
        t1 = time.perf_counter()
        set_lr(live, state, step)
        _, (x, y) = live.feeder.next(step)
        fabric.backward(F.cross_entropy(live.model(x), y))
        live.stepper.step()
        live.optimizer.zero_grad()
        steer.metrics(step, loss=0.0)
        sync.append(t1 - t0)
        step_times.append(time.perf_counter() - t0)
    stop.set()
    if proc:
        proc.join()
    applied = log.db.execute("SELECT count(*) FROM commands WHERE status='applied'").fetchone()[0]
    tmp.cleanup()
    return {
        "writes_per_s": rate,
        "step_ms": 1e3 * statistics.mean(step_times),
        "sync_mean_us": 1e6 * statistics.mean(sync),
        "sync_median_us": 1e6 * statistics.median(sync),
        "sync_p99_us": 1e6 * sorted(sync)[int(0.99 * len(sync))],
        "patches_applied": applied,
        "versions": state.version,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    args = p.parse_args()
    torch.set_num_threads(2)
    fabric = Fabric(accelerator="cpu", devices=1)
    fabric.launch()
    measure(fabric, 0, 50)
    print("| writes/s | step ms | sync mean µs | sync median µs | sync p99 µs | patches applied | versions |")
    print("|---|---|---|---|---|---|---|")
    for rate in (0, 10, 100, 1000):
        r = measure(fabric, rate, args.steps)
        print(
            f"| {r['writes_per_s']:g} | {r['step_ms']:.1f} | {r['sync_mean_us']:.0f} | {r['sync_median_us']:.0f} "
            f"| {r['sync_p99_us']:.0f} | {r['patches_applied']} | {r['versions']} |"
        )
