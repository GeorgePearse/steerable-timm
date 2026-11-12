from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .log import CommandLog
from .state import parse_assignments, schema


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list):
        return "[" + ",".join(_fmt(x) for x in v) + "]"
    return str(v)


def _send(log: CommandLog, args: argparse.Namespace, patch: dict[str, Any]) -> int:
    try:
        command = log.set(patch, why=args.why, author=args.author)
    except ValueError as e:
        print(f"rejected before queueing: {e}")
        return 1
    row = log.wait(command, args.wait) if args.wait else log.command(command)
    tail = {
        "applied": f"at step {row['step']}",
        "rejected": row["error"],
        "pending": "(trainer has not synced yet)",
    }
    print(f"#{command} {row['status']} {tail[row['status']]}")
    return row["status"] == "rejected"


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--why", default="", help="free-text reason, stored with the change")
    common.add_argument("--author", default=os.environ.get("STEER_AUTHOR", "cli"))
    common.add_argument("--wait", type=float, default=5.0, help="seconds to wait for the trainer's verdict")
    p = argparse.ArgumentParser(prog="steer", description="Steer a running training job through its steer.db")
    p.add_argument("--run", default=os.environ.get("STEER_RUN", "."), help="run directory (or $STEER_RUN)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("get", help="current State as JSON").add_argument("path", nargs="?")
    sub.add_parser("schema", help="every field with type, bounds, default")
    sub.add_parser("set", parents=[common], help="set optimizer.lr=3e-4 ...").add_argument("pairs", nargs="+")
    for name in ("freeze", "unfreeze"):
        sub.add_parser(name, parents=[common], help=f"{name} params by name pattern").add_argument(
            "patterns", nargs="+"
        )
    d = sub.add_parser("data", parents=[common], help="data mix a=1 b=3 | data queue hard:200 train:500")
    d.add_argument("what", choices=["mix", "queue"])
    d.add_argument("items", nargs="*")
    sub.add_parser("run", parents=[common], help="run control").add_argument(
        "action", choices=["pause", "resume", "stop", "eval", "checkpoint"]
    )
    sub.add_parser("log", help="applied and rejected changes").add_argument("--last", type=int, default=20)
    m = sub.add_parser("metrics", help="compact metrics and eval rows")
    m.add_argument("--last", type=int, default=200)
    m.add_argument("--every", type=int, default=1, help="print every k-th metrics row")
    args = p.parse_args(argv)

    if args.cmd == "schema":
        for r in schema():
            bounds = " ".join(f"{k}={r[k]}" for k in ("gt", "ge", "lt", "le") if k in r)
            print(f"{r['path']:<24} {r['type']:<28} {bounds:<16} default={r['default']!r}  {r['doc'] or ''}")
        return 0
    try:
        log = CommandLog(args.run, create=False)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    state = log.get() or {}
    if args.cmd == "get":
        for part in (args.path or "").split("."):
            state = state[part] if part else state
        print(json.dumps(state, indent=1))
    elif args.cmd == "set":
        return _send(log, args, parse_assignments(args.pairs))
    elif args.cmd in ("freeze", "unfreeze"):
        return _send(log, args, {"freeze.frozen" + "+-"[args.cmd == "unfreeze"]: args.patterns})
    elif args.cmd == "data" and args.what == "mix":
        return _send(log, args, {"data.mix": parse_assignments(args.items)})
    elif args.cmd == "data":
        stages = [{"mix": {n: 1.0}, "steps": int(s)} for n, s in (i.rsplit(":", 1) for i in args.items)]
        return _send(log, args, {"data.queue": stages})
    elif args.cmd == "run":
        run = state.get("run", {})
        patch = {
            "pause": {"run.pause": True},
            "resume": {"run.pause": False},
            "stop": {"run.stop": True},
            "eval": {"run.eval_now": run.get("eval_now", 0) + 1},
            "checkpoint": {"run.checkpoint_now": run.get("checkpoint_now", 0) + 1},
        }[args.action]
        return _send(log, args, patch)
    elif args.cmd == "log":
        for r in log.log(args.last):
            what = " ".join(f"{k}:{_fmt(a)}->{_fmt(b)}" for k, (a, b) in r.get("diff", {}).items())
            what = what or json.dumps(r["patch"])
            err = f" ERROR {r['error']}" if r.get("error") else ""
            print(
                f"step {r['step']:>6} v{r['version']:<3} #{r['command']:<3} {r['status']:<8} {r['author']}: {what}{err} | {r['reason']}"
            )
    elif args.cmd == "metrics":
        rows = log.events(("metrics", "eval", "note"), args.last)
        print("step loss lr grad_norm frozen samples/s source version")
        for i, r in enumerate(rows):
            if r["kind"] != "metrics":
                extra = " ".join(
                    f"{k}={_fmt(v)}" for k, v in r.items() if k not in ("step", "ts", "version", "kind")
                )
                print(f"{r['step']} {r['kind']} {extra}")
            elif i % args.every == 0 or i == len(rows) - 1:
                cols = ("step", "loss", "lr", "grad_norm", "frozen", "samples_per_s", "source", "version")
                print(" ".join(_fmt(r.get(c)) for c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
