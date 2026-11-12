from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .log import CommandLog
from .state import State, apply_patch, diff, touches

Hook = Callable[[Any, "State | None", State, int], None]


class Hooks:
    """Registry of on_change handlers: (live objects, old State, new State, step) -> act on the live objects."""

    def __init__(self) -> None:
        self.handlers: list[tuple[tuple[str, ...], Hook]] = []
        self.checks: list[Callable[[Any, State], None]] = []

    def on_change(self, *prefixes: str) -> Callable[[Hook], Hook]:
        def register(fn: Hook) -> Hook:
            self.handlers.append((prefixes, fn))
            return fn

        return register

    def check(self, fn: Callable[[Any, State], None]) -> Callable[[Any, State], None]:
        self.checks.append(fn)
        return fn

    def copy(self) -> Hooks:
        new = Hooks()
        new.handlers, new.checks = list(self.handlers), list(self.checks)
        return new

    def validate(self, live: Any, state: State) -> None:
        for fn in self.checks:
            fn(live, state)

    def apply(self, live: Any, old: State | None, new: State, step: int = 0) -> None:
        """Run every handler whose prefix covers a changed path. old=None runs them all (initial build)."""
        changed = diff(old, new) if old is not None else None
        for prefixes, fn in self.handlers:
            if changed is None or any(touches(changed, p) for p in prefixes):
                fn(live, old, new, step)


class Steer:
    def __init__(
        self, log: CommandLog, live: Any, state: State, hooks: Hooks, step: int = 0, flush_every: int = 50
    ):
        fabric = getattr(live, "fabric", None)
        self.log, self.live, self.state, self.hooks = log, live, state, hooks
        self.fabric = fabric if fabric is not None and fabric.world_size > 1 else None
        self.rank0 = fabric is None or fabric.global_rank == 0
        self.flush_every = flush_every
        self.buffer: list[tuple[int, str, int, dict[str, Any]]] = []
        self.results: list[dict[str, Any]] = []
        self.seen = -1
        if self.rank0:
            log.record([(step, "state", state.version, state.model_dump(mode="json"))])

    def sync(self, step: int) -> State:
        """Call once per step. Costs one integer read unless commands are pending."""
        while True:
            doc = self._poll(step) if self.rank0 else None
            if self.fabric is not None:
                doc = self.fabric.broadcast(doc)
            if doc is not None:
                self._apply(doc, step)
            if not self.state.run.pause or self.state.run.stop:
                return self.state
            time.sleep(0.2)

    def _poll(self, step: int) -> dict[str, Any] | None:
        head = self.log.head()
        if head == self.seen:
            return None
        self.seen = head
        candidate, self.results = self.state, []
        for command, author, reason, patch in self.log.pending():
            row = {"command": command, "author": author, "reason": reason, "patch": patch}
            try:
                nxt = apply_patch(candidate, patch)
                self.hooks.validate(self.live, nxt)
                row.update(status="applied", diff=diff(candidate, nxt))
                candidate = nxt
            except (ValueError, TypeError, KeyError) as e:
                row.update(status="rejected", error=str(e))
            self.results.append(row)
        if candidate is self.state:
            self._resolve(step)
            return None
        return candidate.model_dump(mode="json")

    def _apply(self, doc: dict[str, Any], step: int) -> None:
        old, new = self.state, State.model_validate({**doc, "version": self.state.version + 1})
        try:
            self.hooks.apply(self.live, old, new, step)
            self.state = new
        except Exception as e:
            self.hooks.apply(self.live, new, old, step)
            for row in self.results:
                if row["status"] == "applied":
                    row.update(status="rejected", error=f"hook failed, rolled back: {e!r}")
        if self.rank0:
            self._resolve(step)

    def _resolve(self, step: int) -> None:
        self.flush()
        if self.results:
            self.log.resolve(step, self.state.version, self.results)
        if any(r["status"] == "applied" for r in self.results):
            self.log.record([(step, "state", self.state.version, self.state.model_dump(mode="json"))])
        self.results = []

    def metrics(self, step: int, kind: str = "metrics", **values: Any) -> None:
        """Buffer a row for the events table (rank 0 only); written every flush_every rows."""
        if self.rank0:
            self.buffer.append((step, kind, self.state.version, values))
            if len(self.buffer) >= self.flush_every or kind != "metrics":
                self.flush()

    def flush(self) -> None:
        if self.buffer:
            self.log.record(self.buffer)
            self.buffer = []
