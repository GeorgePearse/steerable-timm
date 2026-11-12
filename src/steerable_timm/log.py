from __future__ import annotations

import json
import sqlite3
import time
from functools import cached_property
from pathlib import Path
from typing import Any

from .state import State, apply_patch

DDL = """
CREATE TABLE IF NOT EXISTS commands(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, author TEXT, reason TEXT, patch TEXT,
  status TEXT DEFAULT 'pending', step INTEGER, error TEXT);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, step INTEGER, kind TEXT, version INTEGER, data TEXT);
CREATE INDEX IF NOT EXISTS events_kind ON events(kind, id);
"""


class CommandLog:
    """The run's steer.db. Writers call set(); the trainer reads it in Steer.sync(). Also the Python client."""

    def __init__(self, run_dir: str | Path, create: bool = True):
        self.path = Path(run_dir) / "steer.db"
        if not create and not self.path.exists():
            raise FileNotFoundError(f"no steer.db in {run_dir}")

    @cached_property
    def db(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, isolation_level=None, timeout=30, check_same_thread=False)
        for _ in range(50):
            try:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA synchronous=NORMAL")
                db.executescript(DDL)
                return db
            except sqlite3.OperationalError:
                time.sleep(0.1)
        raise sqlite3.OperationalError(f"could not open {self.path}")

    def set(self, patch: dict[str, Any], why: str = "", author: str = "anon", check: bool = True) -> int:
        """Queue a patch. With check=True it is first validated against the latest known State."""
        if check and (doc := self.get()) is not None:
            apply_patch(State.model_validate(doc), patch)
        cur = self.db.execute(
            "INSERT INTO commands(ts, author, reason, patch) VALUES (?,?,?,?)",
            (time.time(), author, why, json.dumps(patch)),
        )
        return cur.lastrowid

    def get(self) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT data FROM events WHERE kind='state' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return json.loads(row[0]) if row else None

    def command(self, command_id: int) -> dict[str, Any]:
        cur = self.db.execute("SELECT id, status, step, error FROM commands WHERE id=?", (command_id,))
        return dict(zip(("id", "status", "step", "error"), cur.fetchone(), strict=True))

    def wait(self, command_id: int, timeout: float = 60.0) -> dict[str, Any]:
        end = time.time() + timeout
        while (row := self.command(command_id))["status"] == "pending" and time.time() < end:
            time.sleep(0.05)
        return row

    def events(self, kinds: str | tuple[str, ...], last: int = 20) -> list[dict[str, Any]]:
        kinds = (kinds,) if isinstance(kinds, str) else kinds
        cur = self.db.execute(
            f"SELECT step, ts, version, kind, data FROM events WHERE kind IN ({','.join('?' * len(kinds))}) "
            "ORDER BY id DESC LIMIT ?",
            (*kinds, last),
        )
        rows = [
            {"step": s, "ts": t, "version": v, "kind": k, **json.loads(d)} for s, t, v, k, d in cur.fetchall()
        ]
        return rows[::-1]

    def log(self, last: int = 20) -> list[dict[str, Any]]:
        return self.events("change", last)

    def metrics(self, last: int = 200) -> list[dict[str, Any]]:
        return self.events("metrics", last)

    def head(self) -> int:
        return self.db.execute("SELECT max(id) FROM commands").fetchone()[0] or 0

    def pending(self) -> list[tuple[int, str, str, dict[str, Any]]]:
        cur = self.db.execute(
            "SELECT id, author, reason, patch FROM commands WHERE status='pending' ORDER BY id"
        )
        return [(i, a, r, json.loads(p)) for i, a, r, p in cur.fetchall()]

    def resolve(self, step: int, version: int, results: list[dict[str, Any]]) -> None:
        self.db.execute("BEGIN IMMEDIATE")
        for r in results:
            self.db.execute(
                "UPDATE commands SET status=?, step=?, error=? WHERE id=?",
                (r["status"], step, r.get("error"), r["command"]),
            )
        self.record([(step, "change", version, r) for r in results], tx=False)
        self.db.execute("COMMIT")

    def record(self, rows: list[tuple[int, str, int, dict[str, Any]]], tx: bool = True) -> None:
        if tx:
            self.db.execute("BEGIN IMMEDIATE")
        now = time.time()
        self.db.executemany(
            "INSERT INTO events(ts, step, kind, version, data) VALUES (?,?,?,?,?)",
            [(now, step, kind, version, json.dumps(data)) for step, kind, version, data in rows],
        )
        if tx:
            self.db.execute("COMMIT")
