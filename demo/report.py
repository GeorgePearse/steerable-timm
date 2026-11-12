"""Turn a run's steer.db into demo/commands.jsonl and demo/plot.png.

uv run --with matplotlib python demo/report.py demo/steer.db
"""

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def rows(db: sqlite3.Connection, kind: str) -> list[dict]:
    cur = db.execute("SELECT step, version, data FROM events WHERE kind=? ORDER BY id", (kind,))
    return [{"step": s, "version": v, **json.loads(d)} for s, v, d in cur]


def main(path: str) -> None:
    out = Path(__file__).parent
    db = sqlite3.connect(path)
    metrics, evals, changes, notes = (rows(db, k) for k in ("metrics", "eval", "change", "note"))
    with open(out / "commands.jsonl", "w") as f:
        for r in changes + notes:
            f.write(json.dumps(r) + "\n")

    applied: dict[int, list[dict]] = {}
    for r in changes:
        if r["status"] == "applied":
            applied.setdefault(r["step"], []).append(r)
    rejected = sorted({r["step"] for r in changes if r["status"] != "applied"})
    marks = sorted(applied)

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID})
    fig, axes = plt.subplots(
        4, 1, figsize=(11, 10.5), sharex=True, height_ratios=[3, 3, 2, 1.5], facecolor=SURFACE
    )
    steps = [m["step"] for m in metrics]
    panels = [
        ("train loss (mean per logged window)", [(steps, [m["loss"] for m in metrics], BLUE, None)]),
        (
            "CIFAR-100 test top-1",
            [
                ([e["step"] for e in evals], [e["top1"] for e in evals], BLUE, "all 100 classes"),
                (
                    [e["step"] for e in evals],
                    [e["top1_hard"] for e in evals],
                    ORANGE,
                    "10 hard classes (trees, people)",
                ),
            ],
        ),
        (
            "learning rate, largest param group (log scale)",
            [(steps, [max(m["lr"]) for m in metrics], BLUE, None)],
        ),
        ("fraction of parameters frozen", [(steps, [m["frozen"] for m in metrics], BLUE, None)]),
    ]
    for ax, (title, series) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        for x, y, color, label in series:
            ax.plot(x, y, color=color, lw=1.6, label=label, marker="o" if len(x) < 60 else None, ms=3.5)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        for step in marks:
            ax.axvline(step, color=MUTED, lw=0.7, alpha=0.55)
        for step in rejected:
            ax.axvline(step, color=MUTED, lw=0.7, alpha=0.55, ls=":")
        if len(series) > 1:
            ax.legend(frameon=False, loc="lower right")
    axes[2].set_yscale("log")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_xlabel("training step")
    top, row, last = axes[0], 0, -(10**9)
    top.set_title(panels[0][0], loc="left", fontsize=9.5, color=INK, pad=54)
    for i, step in enumerate(marks, 1):
        row = (row + 1) % 3 if step - last < 700 else 0
        last = step
        top.annotate(
            str(i), (step, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 6 + 15 * row), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.5, color=INK,
        )  # fmt: skip
    lines = []
    for i, step in enumerate(marks, 1):
        what = "; ".join(
            ", ".join(f"{k} {json.dumps(a)} → {json.dumps(b)}" for k, (a, b) in r["diff"].items())
            for r in applied[step]
        )
        lines.append(f"{i}. step {step}: {what if len(what) < 128 else what[:125] + '…'}")
    if rejected:
        lines.append(f"dotted: rejected commands at steps {', '.join(map(str, rejected))}")
    fig.suptitle(
        "A resnet18 fine-tune on CIFAR-100, steered through the steer CLI while it trained",
        x=0.06, ha="left", fontsize=12, color=INK,
    )  # fmt: skip
    fig.text(
        0.06, 0.005, "\n".join(lines), fontsize=7.6, color=MUTED, va="bottom", ha="left", family="monospace"
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.885, bottom=0.07 + 0.0135 * len(lines), hspace=0.32)
    fig.savefig(out / "plot.png", dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main(sys.argv[1])
