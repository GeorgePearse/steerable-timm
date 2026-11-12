"""The demo run on a Modal GPU. The training process only reads steer.db; steering happens from other processes:

modal run --detach demo/modal_app.py
modal container list
modal container exec <container-id> -- steer metrics --last 20
modal container exec <container-id> -- steer unfreeze layer4 --why "..."
modal volume get steerable-timm-demo steer.db demo/steer.db
"""

import modal

MODEL = "resnet18.a1_in1k"
RUN = "/run/demo"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "timm", "lightning", "pydantic")
    .run_commands(
        "python -c \"from torchvision.datasets import CIFAR100; CIFAR100('/data', download=True)\"",
        f"python -c \"import timm; timm.create_model('{MODEL}', pretrained=True)\"",
    )
    .env({"STEER_RUN": RUN, "STEER_AUTHOR": "claude"})
    .add_local_dir(
        ".", "/repo", copy=True, ignore=[".venv", ".git", "runs", "data", "**/__pycache__", "demo/*.db"]
    )
    .run_commands("pip install --no-deps /repo")
)
app = modal.App("steerable-timm-demo", image=image)
volume = modal.Volume.from_name("steerable-timm-demo", create_if_missing=True)

INITIAL = [
    "optimizer.lr=1e-3",
    'freeze.frozen=["conv1","bn1","layer1","layer2","layer3","layer4"]',
    "data.batch_size=128",
    "data.resolution=96",
    "data.workers=8",
    "run.eval_every=1000",
    "run.checkpoint_every=0",
]


@app.function(gpu="T4", cpu=8.0, memory=16384, timeout=3600, volumes={"/out": volume})
def train(steps: int = 10000) -> None:
    import shutil
    import sqlite3
    import subprocess
    from pathlib import Path

    cmd = ["python", "/repo/examples/train.py", "--run", RUN, "--data-dir", "/data", "--model", MODEL]
    cmd += ["--steer", "--steps", str(steps), "--precision", "16-mixed"]
    for pair in INITIAL:
        cmd += ["--set", pair]
    try:
        subprocess.run(cmd, check=True)
    finally:
        sqlite3.connect(f"{RUN}/steer.db").execute("PRAGMA wal_checkpoint(TRUNCATE)")
        for path in Path(RUN).glob("*"):
            if path.suffix in (".db", ".ckpt"):
                shutil.copy(path, "/out")
        volume.commit()


@app.local_entrypoint()
def main(steps: int = 10000) -> None:
    train.spawn(steps)
