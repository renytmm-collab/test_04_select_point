import math
import sys
import time
from pathlib import Path
import numpy as np
from .identity import ROOT, read, seal, verify, digest, code_identity, content_hash, words, rng, inside
from .store import immutable, lock, stage, Ledger
from .guards import manifest, allowed, artifact, dependencies
from .config import require_cap, budget
from .data import split_instances, pool, point, cost, pick
from .features import SCHEMA, cache, fast, audit, rules
from .collection import jobs, execute, labels
from .models import train, predict, specs


def lineage(run, c, prior):
    return {"manifest_digest": manifest(run, c)["digest"], "upstream": dependencies(run, prior)}


def initialize(run, c):
    immutable(Path(run) / "manifest.json", seal({"schema": 1, "experiment": "point-select-v1",
              "config": c, "identity": code_identity(), "evidence": "real backend only; no fixture mixing",
              "runtime": {"numpy": np.__version__, "python": sys.version.split()[0]},
              "streams": {s: "SHA256(root, point-select-v1, stream, semantic identity) -> 128-bit MT19937"
                          for s in ("instance", "split", "state_destroy", "state_repair", "candidate_point",
                                    "development_folds", "A", "B", "model", "bootstrap", "benchmark", "tie")}}))


def states(run, c, cap):
    cap = require_cap(c, "states", cap, budget(c)["states"])
    run = inside(run)
    if (run / "manifest.json").exists():
        raise RuntimeError("States requires a fresh output directory")
    from .backend import generate, payload, restore, destroy_checked, Backend
    with lock(run):
        initialize(run, c)
        with stage(run, "states", {"manifest": manifest(run, c)["digest"], "cap": cap}) as directory:
            split = split_instances(c)
            history = read(ROOT / "protocol/historical_instances.json")
            old = {r["content_sha256"] for r in history["instances"]}
            catalog, seen = [], set()
            for i in range(c["instances"]):
                initial = payload(generate(c, i)); h = content_hash(initial)
                if h in old or h in seen:
                    raise RuntimeError("Historical/new-corpus overlap; no seed substitution")
                seen.add(h)
                catalog.append({"instance": i, "initial": initial, "content_sha256": h,
                                "instance_seed_words": words(c["root_seed"], "instance", i)})
            immutable(directory / "catalog.json", seal({"instances": catalog, "split": split,
                      "history_scope": history["scope"], "history_entries": len(old)}))
            ledger = Ledger(directory); start = time.perf_counter(); environment = None
            snapshots = []
            try:
                it = time.perf_counter(); backend = Backend(c); backend.sync()
                init_seconds = time.perf_counter() - it; environment = backend.environment
                for entry in catalog:
                    i = entry["instance"]; current = restore(entry["initial"])
                    for step_no in range(1, max(c["state_steps"]) + 1):
                        key = f"state/{i}/{step_no}"
                        if ledger.count() >= cap:
                            raise RuntimeError("State attempt budget exhausted")
                        meta = {"instance": i, "step": step_no, "repair_invoked": False,
                                "destroy_seed_words": words(c["root_seed"], "state_destroy", i, step_no),
                                "repair_seed_words": words(c["root_seed"], "state_repair", i, step_no)}
                        ledger.begin(key, meta); invoked = False
                        try:
                            before_state = payload(current); source_hash = digest(before_state)
                            before = float(current.get_costs(False))
                            if not math.isfinite(before) or before <= 0:
                                raise ValueError("Invalid state cost")
                            proposal = restore(before_state)
                            ids, center = point(before_state, rng(c["root_seed"], "state_destroy", i, step_no))
                            partial_digest = destroy_checked(proposal, ids)
                            backend.sync(); tic = time.perf_counter(); invoked = True
                            backend.repair([proposal], [rng(c["root_seed"], "state_repair", i, step_no)])
                            backend.sync(); seconds = time.perf_counter() - tic
                            after = float(proposal.get_costs(False))
                            if not math.isfinite(after) or after <= 0 or digest(payload(current)) != source_hash:
                                raise RuntimeError("Invalid repaired cost or mutated source")
                            accepted = after <= before
                            if accepted:
                                current = proposal
                            ledger.finish(key, "ok", {**meta, "repair_invoked": True, "accepted": accepted,
                                "cost_before": before, "proposal_cost": after, "repair_seconds": seconds,
                                "customers": ids, "center": center, "partial_digest": partial_digest,
                                "accepted_state_digest": digest(payload(current))})
                        except BaseException as exc:
                            ledger.finish(key, "failed", {**meta, "repair_invoked": invoked,
                                          "error_type": type(exc).__name__, "error": str(exc)})
                            raise
                        if step_no in c["state_steps"]:
                            snapshot = {"instance": i, "step": step_no, "state": payload(current),
                                        "cost": float(current.get_costs(False)), "content_sha256": entry["content_sha256"]}
                            snapshots.append(snapshot)
                            immutable(directory / f"snapshot-{i}-{step_no}.json", seal(snapshot))
                immutable(directory / "data.json", seal({"states": snapshots, "split": split,
                          "catalog_digest": verify(read(directory / "catalog.json"))["digest"],
                          "initialization_seconds": init_seconds, "peak_memory": backend.peak()}))
            finally:
                ledger.export_close({"environment": environment, "wall_seconds": time.perf_counter() - start})
    return {"states": len(snapshots), "cap": cap}


def candidates(run, c):
    with lock(run):
        allowed(run, "candidates", c)
        data = artifact(run, "states")
        with stage(run, "candidates", lineage(run, c, ["states"])) as directory:
            start = time.perf_counter()
            tables = [pool(s["state"], c, s["instance"], s["step"]) for s in data["states"]]
            immutable(directory / "data.json", seal({"tables": tables, "states_digest": data["digest"],
                      "generation_seconds": time.perf_counter() - start, "slot_count": c["slots"]}))
    return {"unique_candidates": sum(len(t["unique"]) for t in tables)}


def features(run, c):
    with lock(run):
        allowed(run, "features", c)
        states_data, candidates_data = artifact(run, "states"), artifact(run, "candidates")
        snapshots = {(s["instance"], s["step"]): s["state"] for s in states_data["states"]}
        with stage(run, "features", lineage(run, c, ["states", "candidates"])) as directory:
            all_tables = []; start = time.perf_counter()
            for t in candidates_data["tables"]:
                state = snapshots[t["instance"], t["step"]]
                if digest(state) != t["state_digest"]:
                    raise ValueError("State/candidate identity mismatch")
                ct = time.perf_counter(); cached = cache(state); cache_seconds = time.perf_counter() - ct
                slot_rows, audits = [], {}; fast_seconds, audit_seconds = 0., 0.
                for slot in t["slots"]:
                    ids = t["unique"][slot["candidate"]]
                    ct = time.perf_counter(); vector = fast(cached, ids, slot["center"]); rs = rules(cached, ids)
                    fast_seconds += time.perf_counter() - ct
                    ct = time.perf_counter(); record = audit(state, ids, slot["center"])
                    audit_seconds += time.perf_counter() - ct
                    if not np.allclose(vector, record["values"], rtol=1e-12, atol=1e-12):
                        raise RuntimeError("Fast/audit feature mismatch")
                    audits.setdefault(slot["candidate"], {k: v for k, v in record.items() if k != "values"})
                    slot_rows.append({**slot, "values": vector, "rules": rs})
                all_tables.append({"instance": t["instance"], "step": t["step"], "slots": slot_rows,
                                   "audit": audits, "cache_seconds": cache_seconds,
                                   "fast_features_rules_seconds": fast_seconds, "audit_seconds": audit_seconds})
            immutable(directory / "data.json", seal({"schema": SCHEMA, "tables": all_tables,
                      "candidates_digest": candidates_data["digest"], "wall_seconds": time.perf_counter() - start}))
    return {"feature_schema": SCHEMA, "consistency": "every native slot checked"}


def collect(run, c, name, cap):
    prior = {"collect-dev-a": ["features"], "collect-test-a": ["freeze-test-predictions"],
             "collect-test-b": ["freeze-test-predictions", "freeze-test-reference"]}[name]
    group = "B" if name == "collect-test-b" else "A"
    # Read-only cap preflight before acquiring a writer lock or opening the backend.
    allowed(run, name, c)
    pre_states, pre_candidates = artifact(run, "states"), artifact(run, "candidates")
    pre_ids = pre_states["split"]["development" if name == "collect-dev-a" else "test"]
    pre_tables = [t for t in pre_candidates["tables"] if t["instance"] in pre_ids]
    require_cap(c, name, cap, len(jobs(pre_tables, c[group + "_repeats"])))
    with lock(run):
        allowed(run, name, c)
        states_data, candidate_data, feature_data = artifact(run, "states"), artifact(run, "candidates"), artifact(run, "features")
        ids = states_data["split"]["development" if name == "collect-dev-a" else "test"]
        tables = [t for t in candidate_data["tables"] if t["instance"] in ids]
        tasks = jobs(tables, c[group + "_repeats"])
        cap = require_cap(c, name, cap, len(tasks))
        snapshots = {(s["instance"], s["step"]): s for s in states_data["states"]}
        tt = {(t["instance"], t["step"]): t for t in tables}
        audits = {(t["instance"], t["step"]): t["audit"] for t in feature_data["tables"]}
        from .backend import Backend, restore, destroy_checked

        def prepare(job):
            key = (job["instance"], job["step"]); state = snapshots[key]["state"]
            v = restore(state)
            pd = destroy_checked(v, tt[key]["unique"][job["candidate"]])
            if pd != audits[key][job["candidate"]]["partial_digest"]:
                raise RuntimeError("Partial serialization changed since feature freeze")
            return v, snapshots[key]["cost"]

        with stage(run, name, {**lineage(run, c, prior), "cap": cap, "expected_jobs_digest": digest(tasks),
                              "batch_size": c["batch_size"]}) as directory:
            ledger = Ledger(directory); start = time.perf_counter(); env = None
            try:
                it = time.perf_counter(); backend = Backend(c); backend.sync()
                initialization_seconds = time.perf_counter() - it; env = backend.environment
                execute(ledger, tasks, c["batch_size"], cap, prepare, backend, c["root_seed"], group)
                immutable(directory / "data.json", seal({"group": group, "instances": ids,
                          "expected_jobs_digest": digest(tasks), "initialization_seconds": initialization_seconds,
                          "peak_memory": backend.peak()}))
            finally:
                ledger.export_close({"environment": env, "wall_seconds": time.perf_counter() - start})
    return {"group": group, "rollouts": len(tasks), "cap": cap}


def development_rows(run, c):
    # Explicit development-only access: this function never opens a test label path.
    states_data, candidate_data, feature_data = artifact(run, "states"), artifact(run, "candidates"), artifact(run, "features")
    dev = states_data["split"]["development"]
    tables = [t for t in candidate_data["tables"] if t["instance"] in dev]
    lab = labels(artifact(run, "collect-dev-a", "export.json"), jobs(tables, c["A_repeats"]), c["root_seed"], "A")
    rows = []
    for t in feature_data["tables"]:
        if t["instance"] not in dev:
            continue
        for slot in t["slots"]:
            rows.append({"instance": t["instance"], "step": t["step"], "slot": slot["slot"],
                         "candidate": slot["candidate"], "values": slot["values"],
                         "target": float(np.mean([r["z"] for r in lab[t["instance"], t["step"], slot["candidate"]]])),
                         "weight": 1 / len(dev) / len(c["state_steps"]) / c["slots"]})
    return rows, states_data["split"]


def fit(run, c, max_fits):
    planned = len(specs(c)) * 4 + 4
    if max_fits is None or max_fits < planned:
        raise ValueError(f"Explicit --max-fits >= {planned} required")
    with lock(run):
        allowed(run, "fit", c)
        rows, split = development_rows(run, c)
        with stage(run, "fit", {**lineage(run, c, ["features", "collect-dev-a"]), "max_fits": max_fits}) as directory:
            result = train(rows, split, c)
            immutable(directory / "data.json", seal({**result, "feature_schema": SCHEMA,
                      "fit_environment": {"numpy": np.__version__, "python": sys.version.split()[0],
                                          "platform": sys.platform, "algorithm": "deterministic weighted Ridge/CART"},
                      "training_rows_digest": digest(rows), "code_identity": code_identity()}))
    return {"main_method": result["main_method"], "fit_count": result["fit_count"], "evidence": "development A only"}


def policy_load(path):
    p = read(path)
    for field in ("min_gain_pp", "max_online_repair_ratio"):
        if not isinstance(p.get(field), (float, int)) or isinstance(p[field], bool) or not math.isfinite(p[field]):
            raise ValueError("Freeze finite engineering threshold: " + field)
    if p["schema"] != 1 or p["min_gain_pp"] < 0 or p["max_online_repair_ratio"] <= 0:
        raise ValueError("Invalid continuation policy")
    return p


def online_decision(state, c, i, step, fitted):
    start = time.perf_counter(); t = pool(state, c, i, step); generation = time.perf_counter() - start
    tic = time.perf_counter(); cc = cache(state)
    vectors = [fast(cc, t["unique"][s["candidate"]], s["center"]) for s in t["slots"]]
    rs = [rules(cc, t["unique"][s["candidate"]]) for s in t["slots"]]
    feature_seconds = time.perf_counter() - tic
    selected, timings = {}, {}
    score_arrays = {"constant": [0.] * c["slots"],
                    "contact_edges": [r["contact_edges"] for r in rs],
                    "release_concentration": [r["release_concentration"] for r in rs]}
    for name, model in fitted["models"].items():
        width = 5 if model["spec"]["schema"] == "geometry" else 12
        tic = time.perf_counter(); scores = predict(model, np.asarray(vectors)[:, :width])
        slot = pick(scores, c["root_seed"], i, step, name)
        timings[name] = {"generation_seconds": generation, "cache_features_rules_seconds": feature_seconds,
                         "predict_select_seconds": time.perf_counter() - tic}
        selected[name] = {"slot": slot, "candidate": t["slots"][slot]["candidate"]}
        score_arrays[name] = scores.tolist()
    for name in ("constant", "contact_edges", "release_concentration"):
        tic = time.perf_counter(); slot = pick(score_arrays[name], c["root_seed"], i, step, name)
        timings[name] = {"generation_seconds": generation, "cache_features_rules_seconds": feature_seconds,
                         "predict_select_seconds": time.perf_counter() - tic}
        selected[name] = {"slot": slot, "candidate": t["slots"][slot]["candidate"]}
    for tm in timings.values():
        tm["online_total_seconds"] = tm["generation_seconds"] + tm["cache_features_rules_seconds"] + tm["predict_select_seconds"]
    return t, vectors, selected, timings, score_arrays


def freeze_predictions(run, c, policy_path, benchmark_runs=()):
    p = policy_load(policy_path)
    with lock(run):
        allowed(run, "freeze-test-predictions", c)
        states_data, candidate_data, feature_data, fitted = (artifact(run, n) for n in ("states", "candidates", "features", "fit"))
        test = states_data["split"]["test"]
        benchmark_registry = []
        development_state_digests = {digest(s["state"]) for s in states_data["states"] if s["instance"] in states_data["split"]["development"]}
        for br in benchmark_runs:
            manifest(br, c)
            bd = artifact(br, "benchmark"); be = artifact(br, "benchmark", "export.json")
            benchmark_registry.append({"benchmark_digest": bd["digest"], "export_digest": be["digest"],
                                       "batch_size": bd["batch_size"],
                                       "matched_development_snapshot": bd["snapshot_digest"] in development_state_digests})
        tt = {(t["instance"], t["step"]): t for t in candidate_data["tables"]}
        ff = {(t["instance"], t["step"]): t for t in feature_data["tables"]}
        with stage(run, "freeze-test-predictions", lineage(run, c, ["fit", "features"])) as directory:
            frozen = []
            for s in states_data["states"]:
                i, step_no = s["instance"], s["step"]
                if i not in test:
                    continue
                runs = [online_decision(s["state"], c, i, step_no, fitted) for _ in range(7)]
                table, vectors, selected, timing, scores = runs[-1]
                if table != tt[i, step_no] or not np.allclose(vectors, [slot["values"] for slot in ff[i, step_no]["slots"]], rtol=1e-12, atol=1e-12):
                    raise RuntimeError("Fast online path differs from frozen candidate/feature path")
                if any(v[2] != selected for v in runs):
                    raise RuntimeError("Nondeterministic prediction/selection")
                frozen.append({"instance": i, "step": step_no, "selected": selected, "scores": scores,
                               "decision_timing": {"warmup": [r[3] for r in runs[:2]], "steady": [r[3] for r in runs[2:]],
                                                   "scope": "all 16 candidates, cache build, features/rules and per-method predict/select; host CPU; no audit/deepcopy/JSON"}})
            immutable(directory / "data.json", seal({"predictions": frozen, "main_method": fitted["main_method"],
                      "model_digest": fitted["digest"], "candidates_digest": candidate_data["digest"],
                      "feature_schema": SCHEMA, "policy": p, "policy_digest": digest(p),
                      "benchmark_registry": benchmark_registry,
                      "test_labels_read": False, "read_scope": ["states", "candidates", "features", "fit"]}))
    return {"test_states": len(frozen), "main_method": fitted["main_method"], "test_labels_read": False}


def freeze_reference(run, c):
    with lock(run):
        allowed(run, "freeze-test-reference", c)
        states_data, candidate_data = artifact(run, "states"), artifact(run, "candidates")
        tables = [t for t in candidate_data["tables"] if t["instance"] in states_data["split"]["test"]]
        a = artifact(run, "collect-test-a", "export.json")
        aa = labels(a, jobs(tables, c["A_repeats"]), c["root_seed"], "A")
        winners = []
        with stage(run, "freeze-test-reference", lineage(run, c, ["freeze-test-predictions", "collect-test-a"])) as directory:
            for t in tables:
                means = {cid: float(np.mean([r["z"] for r in aa[t["instance"], t["step"], cid]])) for cid in t["unique"]}
                # Reference original deterministic normalized-ID tie rule.
                cid = min(means, key=lambda cid: (-means[cid], cid))
                winners.append({"instance": t["instance"], "step": t["step"], "candidate": cid, "A_mean_z": means[cid]})
            immutable(directory / "data.json", seal({"winners": winners, "A_export_digest": a["digest"],
                      "B_read": False, "criterion": "mean per-rollout z; exact tie smallest normalized candidate ID",
                      "interpretation": "noisy expensive A selection reference, not true upper bound"}))
    return {"test_reference_states": len(winners), "B_read": False}


def benchmark(run, c, cap, batch_size, warmup, steady, snapshot_path=None):
    if batch_size < 1 or warmup < 1 or steady < 1:
        raise ValueError("Positive benchmark batch/warmup/steady required")
    planned = (warmup + steady) * batch_size
    cap = require_cap(c, "benchmark", cap, planned)
    run = inside(run)
    if (run / "manifest.json").exists():
        raise RuntimeError("Benchmark requires its own fresh directory")
    from .backend import Backend, generate, payload, restore, destroy_checked
    if snapshot_path:
        snap = verify(read(snapshot_path))
        state = snap["state"]
        scope = "user supplied complete state snapshot"
        instance_source = {"snapshot_path": str(snapshot_path), "snapshot_artifact_digest": snap["digest"]}
    else:
        # Distinct benchmark stream, never uses research instance stream.
        bc = {**c, "root_seed": int(rng(c["root_seed"], "benchmark", "instance").randint(1, 2**31))}
        state = payload(generate(bc, 0)); scope = "independent initial solution; not a representative evolved-state guarantee"
        instance_source = {"benchmark_instance_seed": words(c["root_seed"], "benchmark", "instance"),
                           "derived_root_seed": bc["root_seed"], "generation_seed": words(bc["root_seed"], "instance", 0)}
    ids, center = point(state, rng(c["root_seed"], "benchmark", "point"))
    cid = digest({"state": digest(state), "customers": ids})
    tasks = [{"instance": -1, "step": 0, "candidate": cid, "repeat": r} for r in range(planned)]

    def prepare(job):
        v = restore(state); destroy_checked(v, ids)
        return v, cost(state)

    with lock(run):
        initialize(run, c)
        with stage(run, "benchmark", {"cap": cap, "batch_size": batch_size, "warmup": warmup, "steady": steady}) as directory:
            immutable(directory / "representative.json", seal({"state": state, "customers": ids, "center": center,
                      "instance_source": instance_source, "point_seed": words(c["root_seed"], "benchmark", "point")}))
            ledger = Ledger(directory); start = time.perf_counter(); env = None
            try:
                tic = time.perf_counter(); backend = Backend(c); backend.sync()
                init_seconds = time.perf_counter() - tic; env = backend.environment
                execute(ledger, tasks, batch_size, cap, prepare, backend, c["root_seed"], "benchmark")
                immutable(directory / "data.json", seal({"scope": scope, "snapshot_digest": digest(state),
                          "warmup_batches": warmup, "steady_batches": steady, "batch_size": batch_size,
                          "initialization_seconds": init_seconds, "peak_memory": backend.peak(),
                          "timing_note": "CUDA synchronized around Repair; preparation separate; batch allocation not independent latency"}))
            finally:
                ledger.export_close({"environment": env, "wall_seconds": time.perf_counter() - start})
    return {"benchmark_rollouts": planned, "scope": scope}
