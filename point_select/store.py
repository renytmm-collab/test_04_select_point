"""Stage receipts plus durable per-rollout ledger. No automatic restart."""
import contextlib
import os
import sqlite3
import time
from pathlib import Path
from .identity import canonical, digest, read, seal, verify, inside


def immutable(path, value):
    path = inside(path)
    if path.exists():
        if read(path) != value:
            raise RuntimeError("Immutable artifact changed: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create is deliberate: a partial file remains blocking after a crash.
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(canonical(value) + "\n")
        f.flush()
        os.fsync(f.fileno())


@contextlib.contextmanager
def lock(directory):
    directory = inside(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "writer.lock").open("a+b") as f:
        if f.seek(0, 2) == 0:
            f.write(b"0"); f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def stage(run, name, lineage):
    directory = inside(Path(run) / "stages" / name)
    if directory.exists():
        raise RuntimeError("Stage already started/completed; audit it, never overwrite or auto-retry")
    immutable(directory / "started.json", seal({"stage": name, "lineage": lineage, "started_utc_epoch": time.time()}))
    try:
        yield directory
    except BaseException as exc:
        immutable(directory / "failed.json", seal({"error_type": type(exc).__name__, "error": str(exc),
                                                   "finished_utc_epoch": time.time()}))
        raise
    else:
        # Bind every completed stage artifact, including the started receipt and exports.
        from .identity import file_sha
        files = {p.name: file_sha(p) for p in directory.glob("*.json")}
        immutable(directory / "complete.json", seal({"stage": name, "files": files,
                                                     "finished_utc_epoch": time.time()}))


class Ledger:
    def __init__(self, directory):
        self.directory = inside(directory)
        self.db = sqlite3.connect(self.directory / "rollouts.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE tasks (key TEXT PRIMARY KEY, status TEXT NOT NULL, data TEXT NOT NULL)")
        self.db.commit()

    def begin(self, key, meta):
        if self.db.execute("SELECT 1 FROM tasks WHERE key=?", (key,)).fetchone():
            raise RuntimeError("Existing task blocks duplicate or retry")
        self.db.execute("INSERT INTO tasks VALUES (?, 'started', ?)", (key, canonical(meta)))
        self.db.commit()

    def finish(self, key, status, data):
        if status not in ("ok", "failed"):
            raise ValueError("Invalid terminal status")
        cur = self.db.execute("UPDATE tasks SET status=?,data=? WHERE key=? AND status='started'",
                              (status, canonical(data), key))
        if cur.rowcount != 1:
            raise RuntimeError("Only one transition from started is allowed")
        self.db.commit()

    def count(self):
        return self.db.execute("SELECT count(*) FROM tasks").fetchone()[0]

    def export_close(self, extra):
        try:
            rows = [{"key": k, "status": s, **read_data(d)} for k, s, d in
                    self.db.execute("SELECT key,status,data FROM tasks ORDER BY key")]
            costs = {"attempt_records": len(rows), "completed": sum(r["status"] == "ok" for r in rows),
                     "failed": sum(r["status"] == "failed" for r in rows),
                     "uncertain": sum(r["status"] == "started" for r in rows),
                     "repair_invocations_known": sum(r.get("repair_invoked") is True for r in rows)}
            immutable(self.directory / "export.json", seal({"rows": rows, "costs": costs, **extra}))
        finally:
            self.db.close()


def read_data(data):
    import json
    return json.loads(data)
