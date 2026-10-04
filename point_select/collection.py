import time
import numpy as np
from .identity import canonical, words, rng, digest


def label_seed(root, group, job):
    # Same normalized action shares labels across all slot/method aliases.
    return words(root, group, job["instance"], job["step"], job["candidate"], job["repeat"])


def jobs(tables, repeats):
    return [{"instance": t["instance"], "step": t["step"], "candidate": cid, "repeat": rep}
            for t in sorted(tables, key=lambda t: (t["instance"], t["step"]))
            for cid in sorted(t["unique"]) for rep in range(repeats)]


def job_key(job):
    return canonical([job[k] for k in ("instance", "step", "candidate", "repeat")])


def execute(ledger, tasks, batch_size, cap, prepare, backend, root, group):
    for bid, start in enumerate(range(0, len(tasks), batch_size)):
        batch = tasks[start:start + batch_size]
        if ledger.count() + len(batch) > cap:
            raise RuntimeError("Attempt budget exhausted")
        metas = [{**j, "batch": bid, "group": group, "batch_jobs_digest": digest([job_key(v) for v in batch]),
                  "batch_offset": start,
                  "seed_words": label_seed(root, group, j), "repair_invoked": False,
                  "fixture": bool(getattr(backend, "environment", {}).get("fixture", False))} for j in batch]
        for j, m in zip(batch, metas):
            ledger.begin(job_key(j), m)
        invoked = False; finished = set(); tic = time.perf_counter()
        try:
            prep_start = time.perf_counter()
            prepared = [prepare(j) for j in batch]
            prepare_seconds = time.perf_counter() - prep_start
            if any(not np.isfinite(v[1]) or v[1] <= 0 for v in prepared):
                raise ValueError("Nonpositive/nonfinite original cost")
            points = [v[0].get_max_nb_input_points() for v in prepared]
            backend.sync(); rt = time.perf_counter(); invoked = True
            backend.repair([v[0] for v in prepared], [np.random.RandomState(m["seed_words"]) for m in metas])
            backend.sync(); repair_seconds = time.perf_counter() - rt
            for j, m, (v, before), pts in zip(batch, metas, prepared, points):
                after = float(v.get_costs(False))
                if not np.isfinite(after) or after <= 0:
                    raise ValueError("Nonpositive/nonfinite repaired cost")
                y = (before - after) / before
                ledger.finish(job_key(j), "ok", {**m, "repair_invoked": True, "cost_before": before,
                    "cost_after": after, "y": float(y), "z": float(max(y, 0)),
                    "prepare_batch_seconds": prepare_seconds, "repair_batch_seconds": repair_seconds,
                    "batch_size": len(batch), "input_points": pts, "padded_input_points": max(points)})
                finished.add(job_key(j))
        except BaseException as exc:
            for j, m in zip(batch, metas):
                if job_key(j) not in finished:
                    ledger.finish(job_key(j), "failed", {**m, "repair_invoked": invoked,
                        "error_type": type(exc).__name__, "error": str(exc), "attempt_seconds": time.perf_counter() - tic})
            raise


def labels(export, expected, root, group):
    rows = export["rows"]
    expected_map = {job_key(j): j for j in expected}
    if len(rows) != len(expected_map) or {r["key"] for r in rows} != set(expected_map):
        raise ValueError("Incomplete/duplicate label coverage")
    out = {}
    for r in rows:
        j = expected_map[r["key"]]
        if r["status"] != "ok" or r["group"] != group or r["seed_words"] != label_seed(root, group, j):
            raise ValueError("Failed/uncertain/mismatched label")
        if any(r[k] != j[k] for k in j):
            raise ValueError("Label identity mismatch")
        y = (r["cost_before"] - r["cost_after"]) / r["cost_before"]
        if not np.isfinite(y) or r["y"] != y or r["z"] != max(y, 0):
            raise ValueError("Invalid per-rollout y/z")
        out.setdefault((j["instance"], j["step"], j["candidate"]), []).append(r)
    return out
