import time
from pathlib import Path
import numpy as np
from .identity import rng, words, seal, digest, read, verify, file_sha, inside
from .guards import manifest, artifact, completed, ORDER
from .store import immutable
from .collection import jobs, labels


def metrics(rows):
    y = np.asarray([r["y"] for r in rows]); z = np.asarray([r["z"] for r in rows])
    positive = y > 0
    return {"mean_z_pp": float(z.mean() * 100), "mean_y_pp": float(y.mean() * 100),
            "positive_probability": float(positive.mean()), "positive_events": int(positive.sum()),
            "label_occurrences": len(rows),
            "positive_conditional_y_pp": float(y[positive].mean() * 100) if positive.any() else None}


def interval(x, c, tag):
    x = np.asarray(x, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Complete finite instance vector required")
    stream = rng(c["root_seed"], "bootstrap", tag)
    means = [float(x[stream.randint(len(x), size=len(x))].mean()) for _ in range(c["bootstrap_repeats"])]
    return {"instances": len(x), "mean_pp": float(x.mean()),
            "exploratory_interval_pp": np.quantile(means, [.025, .975]).tolist(),
            "bootstrap_seed_words": words(c["root_seed"], "bootstrap", tag),
            "caveat": "instance cluster percentile bootstrap; exploratory, no power analysis or multiplicity correction"}


def cost_summary(export):
    # Every physical batch is counted once, not once per rollout allocation.
    batches = {}
    for row in export["rows"]:
        if row["status"] == "ok" and "batch" in row:
            batches.setdefault(row["batch"], row)
    return {**export["costs"], "wall_seconds": export["wall_seconds"],
            "prepare_seconds": sum(r.get("prepare_batch_seconds", 0) for r in batches.values()),
            "repair_seconds": (sum(r.get("repair_batch_seconds", 0) for r in batches.values()) if batches else
                               sum(r.get("repair_seconds", 0) for r in export["rows"] if r["status"] == "ok")),
            "batches": len(batches)}


def audit_run(run, c):
    m = manifest(run, c)
    results = []; costs = []
    for name in (*ORDER, "benchmark"):
        path = Path(run) / "stages" / name
        if not path.exists():
            results.append({"stage": name, "status": "absent"}); continue
        status = "complete"
        if (path / "failed.json").exists():
            status = "failed"
        elif not (path / "complete.json").exists():
            status = "started/uncertain"
        else:
            completed(run, name)
        results.append({"stage": name, "status": status})
        if (path / "export.json").exists():
            ex = verify(read(path / "export.json"))
            costs.append({"stage": name, **cost_summary(ex)})
        elif (path / "rollouts.sqlite").exists():
            costs.append({"stage": name, "status": "no sealed export; physical ledger requires independent recovery audit",
                          "uncertain_cost_upper_bound": verify(read(path / "started.json"))["lineage"].get("cap")})
    present = {r["stage"] for r in results if r["status"] != "absent"}
    for j, name in enumerate(ORDER):
        if name in present and any(p not in present for p in ORDER[:j]):
            raise RuntimeError("Missing predecessor stage")
    # Freeze ordering checked by receipts, not solely filesystem mtimes.
    for prior, later in zip(ORDER, ORDER[1:]):
        if later in present and prior in present:
            p = Path(run) / "stages" / prior / "complete.json"
            l = Path(run) / "stages" / later / "started.json"
            if not p.exists() or verify(read(p))["finished_utc_epoch"] > verify(read(l))["started_utc_epoch"]:
                raise RuntimeError("Invalid stage chronology")
    done = {r["stage"] for r in results if r["status"] == "complete"}
    if "states" in done:
        from .data import split_instances, pool
        sd = artifact(run, "states")
        if sd["split"] != split_instances(c):
            raise RuntimeError("Instance split/fold assignment changed")
        expected = {(i, step) for i in range(c["instances"]) for step in c["state_steps"]}
        if len(sd["states"]) != len(expected) or {(s["instance"], s["step"]) for s in sd["states"]} != expected:
            raise RuntimeError("Missing/duplicate state scope")
    if "candidates" in done:
        cd = artifact(run, "candidates")
        reconstructed = [pool(s["state"], c, s["instance"], s["step"]) for s in sd["states"]]
        if cd["states_digest"] != sd["digest"] or cd["tables"] != reconstructed:
            raise RuntimeError("Point pool / native slot mapping changed")
    if "features" in done:
        from .features import SCHEMA
        fd = artifact(run, "features")
        if fd["candidates_digest"] != cd["digest"] or fd["schema"] != SCHEMA:
            raise RuntimeError("Feature schema/candidate lineage mismatch")
    for name in ("collect-dev-a", "collect-test-a", "collect-test-b"):
        if name in done:
            group = "B" if name == "collect-test-b" else "A"
            ids = sd["split"]["development" if name == "collect-dev-a" else "test"]
            ts = [t for t in cd["tables"] if t["instance"] in ids]
            labels(artifact(run, name, "export.json"), jobs(ts, c[group + "_repeats"]), c["root_seed"], group)
    if "fit" in done:
        fit = artifact(run, "fit")
        if fit["folds"] != sd["split"]["folds"] or fit["feature_schema"] != fd["schema"] or fit["code_identity"] != m["identity"]:
            raise RuntimeError("Frozen model/fold/source identity mismatch")
    if "freeze-test-predictions" in done:
        pred = artifact(run, "freeze-test-predictions")
        if pred["model_digest"] != fit["digest"] or pred["candidates_digest"] != cd["digest"] or pred["test_labels_read"] is not False:
            raise RuntimeError("Test prediction freeze lineage mismatch")
        expected = {(i, step) for i in sd["split"]["test"] for step in c["state_steps"]}
        if len(pred["predictions"]) != len(expected) or {(r["instance"], r["step"]) for r in pred["predictions"]} != expected:
            raise RuntimeError("Test prediction scope changed")
        if pred["policy_digest"] != digest(pred["policy"]):
            raise RuntimeError("Continuation policy changed")
    if "freeze-test-reference" in done:
        rr = artifact(run, "freeze-test-reference")
        test_tables = [t for t in cd["tables"] if t["instance"] in sd["split"]["test"]]
        a = artifact(run, "collect-test-a", "export.json")
        aa = labels(a, jobs(test_tables, c["A_repeats"]), c["root_seed"], "A")
        rebuilt = []
        for t in test_tables:
            means = {cid: float(np.mean([r["z"] for r in aa[t["instance"], t["step"], cid]])) for cid in t["unique"]}
            cid = min(means, key=lambda cid: (-means[cid], cid))
            rebuilt.append({"instance": t["instance"], "step": t["step"], "candidate": cid, "A_mean_z": means[cid]})
        if rr["winners"] != rebuilt or rr["A_export_digest"] != a["digest"] or rr["B_read"] is not False:
            raise RuntimeError("A reference must reconstruct exclusively from A")
    # Every stage receipt binds the predecessor receipts it actually used.
    for name in done:
        started = verify(read(Path(run) / "stages" / name / "started.json"))
        for upstream, expected_digest in started["lineage"].get("upstream", {}).items():
            if digest(completed(run, upstream)) != expected_digest:
                raise RuntimeError("Stage predecessor receipt mismatch")
    return {"manifest_digest": m["digest"], "stages": results, "costs": costs,
            "fail_closed": all(r["status"] not in ("failed", "started/uncertain") for r in results),
            "history_scope": read(Path(__file__).resolve().parents[1] / "protocol/historical_instances.json")["scope"],
            "read_only": True, "scope": "sealed receipts/exports; no SQLite/WAL opens, no claims to recover unexported records"}


def analyze(run, c, output=None, benchmark_runs=()):
    start = time.perf_counter(); audit_result = audit_run(run, c)
    if not audit_result["fail_closed"]:
        raise RuntimeError("Failed/uncertain run cannot become an evaluation report")
    states, candidates, pred, ref, fitted = (artifact(run, n) for n in
            ("states", "candidates", "freeze-test-predictions", "freeze-test-reference", "fit"))
    test = states["split"]["test"]
    tables = [t for t in candidates["tables"] if t["instance"] in test]
    bb_export = artifact(run, "collect-test-b", "export.json")
    fixture = bool(bb_export["environment"].get("fixture", False))
    for name in ("states", "collect-dev-a", "collect-test-a"):
        if bool(artifact(run, name, "export.json")["environment"].get("fixture", False)) != fixture:
            raise ValueError("Synthetic/real evidence mixing forbidden")
    bb = labels(bb_export, jobs(tables, c["B_repeats"]), c["root_seed"], "B")
    frozen = {(r["instance"], r["step"]): r for r in pred["predictions"]}
    reference = {(r["instance"], r["step"]): r["candidate"] for r in ref["winners"]}
    expected_states = {(i, s) for i in test for s in c["state_steps"]}
    if set(frozen) != expected_states or set(reference) != expected_states:
        raise ValueError("Incomplete frozen predictions/reference")
    state_rows = []
    methods = ["native_expectation", "constant", "contact_edges", "release_concentration", *fitted["models"], "A_reference"]
    for t in tables:
        key = t["instance"], t["step"]
        native = [r for s in t["slots"] for r in bb[*key, s["candidate"]]]
        mm = {"native_expectation": metrics(native)}
        for name, chosen in frozen[key]["selected"].items():
            mm[name] = metrics(bb[*key, chosen["candidate"]])
        mm["A_reference"] = metrics(bb[*key, reference[key]])
        actual = [r for cid in t["unique"] for r in bb[*key, cid]]
        state_rows.append({"instance": key[0], "step": key[1], "methods": mm,
                           "unique_candidates": len(t["unique"]), "native_slots": len(t["slots"]),
                           "actual_unique_events": metrics(actual), "timing": frozen[key]["decision_timing"]})
    instance_rows = []
    for i in test:
        ss = [s for s in state_rows if s["instance"] == i]
        if {s["step"] for s in ss} != set(c["state_steps"]):
            raise ValueError("Incomplete instance; no silent exclusion")
        mm = {m: {k: float(np.mean([s["methods"][m][k] for s in ss]))
                  for k in ("mean_z_pp", "mean_y_pp", "positive_probability")} for m in methods}
        for m in methods:
            mm[m]["positive_conditional_y_pp"] = (mm[m]["mean_z_pp"] / mm[m]["positive_probability"]
                                                  if mm[m]["positive_probability"] > 0 else None)
        instance_rows.append({"instance": i, "states": len(ss), "methods": mm})
    means = {m: interval([r["methods"][m]["mean_z_pp"] for r in instance_rows], c, m) for m in methods}
    auxiliary = {}
    for m in methods:
        mean_y = float(np.mean([r["methods"][m]["mean_y_pp"] for r in instance_rows]))
        probability = float(np.mean([r["methods"][m]["positive_probability"] for r in instance_rows]))
        auxiliary[m] = {"mean_y_pp": mean_y, "positive_probability": probability,
                        "positive_conditional_y_pp": means[m]["mean_pp"] / probability if probability > 0 else None,
                        "native_weighted_event_occurrences": sum(s["methods"][m]["positive_events"] for s in state_rows),
                        "actual_unique_events_scope": "per_state.actual_unique_events; occurrences are not independent events"}
    paired = {f"{m}-{base}": interval([r["methods"][m]["mean_z_pp"] - r["methods"][base]["mean_z_pp"]
                                      for r in instance_rows], c, f"{m}-{base}")
              for m in methods if m != "native_expectation" for base in ("native_expectation", "contact_edges", "release_concentration") if m != base}
    structure = {family: interval([r["methods"][family + "_full"]["mean_z_pp"] - r["methods"][family + "_geometry"]["mean_z_pp"]
                                  for r in instance_rows], c, family + "/structure") for family in ("ridge", "tree")}
    strata = {str(step): {m: interval([s["methods"][m]["mean_z_pp"] for s in state_rows if s["step"] == step], c,
                                     f"step-{step}/{m}") for m in methods} for step in c["state_steps"]}
    # A scientifically interpretable latency ratio requires a separate batch-size-1 benchmark.
    calibration = []
    for br in benchmark_runs:
        manifest(br, c)
        bd = artifact(br, "benchmark"); be = artifact(br, "benchmark", "export.json")
        registration = next((r for r in pred["benchmark_registry"] if r["benchmark_digest"] == bd["digest"] and r["export_digest"] == be["digest"]), None)
        if registration is None:
            raise ValueError("Benchmark must be registered before test prediction freeze; post-test calibration selection forbidden")
        unique = {r["batch"]: r for r in be["rows"] if r["status"] == "ok"}
        steady = [r for b, r in unique.items() if b >= bd["warmup_batches"]]
        calibration.append({"run": str(br), "digest": bd["digest"], "batch_size": bd["batch_size"], "scope": bd["scope"],
                            "matched_development_snapshot": registration["matched_development_snapshot"],
                            "steady_repair_seconds": float(np.mean([r["repair_batch_seconds"] for r in steady])),
                            "environment": be["environment"], "costs": cost_summary(be)})
    same_environment = bb_export["environment"]
    singles = [b["steady_repair_seconds"] for b in calibration if b["batch_size"] == 1 and b["environment"] == same_environment]
    matched_singles = [b["steady_repair_seconds"] for b in calibration if b["batch_size"] == 1 and
                       b["environment"] == same_environment and b["matched_development_snapshot"]]
    online = {m: float(np.mean([t[m]["online_total_seconds"] for s in state_rows for t in s["timing"]["steady"]]))
              for m in methods if m not in ("native_expectation", "A_reference")}
    bcost = cost_summary(bb_export)
    latency_ratio = {m: v / np.mean(singles) for m, v in online.items()} if singles else None
    throughput_ratio = {m: v / (bcost["repair_seconds"] / bcost["completed"]) for m, v in online.items()}
    main = pred["main_method"]; policy = pred["policy"]

    def evidence(item):
        if item["mean_pp"] <= 0:
            return "no positive observed mean; pause expansion"
        if item["exploratory_interval_pp"][0] <= 0:
            return "insufficient independent evidence; pause expansion"
        return "exploratory positive evidence"

    a_status = evidence(paired["A_reference-native_expectation"])
    b_status = evidence(paired[main + "-native_expectation"])
    matched_ratio = {m: v / np.mean(matched_singles) for m, v in online.items()} if matched_singles else None
    d_pass = matched_ratio is not None and matched_ratio[main] <= policy["max_online_repair_ratio"]
    practical_pass = all(paired[main + "-" + base]["mean_pp"] > policy["min_gain_pp"] and
                         paired[main + "-" + base]["exploratory_interval_pp"][0] > 0
                         for base in ("native_expectation", "contact_edges", "release_concentration"))
    report = {"schema": 1, "label": "synthetic fixture integration only" if fixture else "exploratory real-backend one-step experiment",
              "evidence_level": "synthetic" if fixture else "real backend one-step exploratory",
              "independent_test_instances": len(test),
              "units": "mean relative improvement expressed in percent; paired differences in percentage points",
              "aggregation": "rollout -> native slot -> equal states per instance -> equal instances",
              "per_state": state_rows, "per_instance": instance_rows, "method_mean_z": means, "auxiliary": auxiliary, "paired_gains": paired,
              "structure_ablation": structure, "step_strata": strata, "development_cv": fitted["cv"],
              "event_caveat": "Native-weighted label occurrences reuse duplicates; actual unique events are separate. Zero events do not prove zero probability.",
              "timing": {"online_total_seconds": online, "single_repair_ratio": latency_ratio,
                         "matched_development_single_repair_ratio": matched_ratio,
                         "B_batch_amortized_throughput_ratio": throughput_ratio,
                         "ratio_caveat": "Batch-amortized ratios are throughput proxies, not single Repair latency; single benchmark representativeness still requires review.",
                         "benchmark_calibration": calibration},
              "costs": {"stages": audit_result["costs"], "independent_benchmarks": calibration,
                        "candidate_generation_seconds": candidates["generation_seconds"],
                        "offline_feature_audit_seconds": artifact(run, "features")["wall_seconds"],
                        "training_seconds": fitted["training_seconds"],
                        "fit_environment": fitted["fit_environment"],
                        "historical_restarts": "none allowed automatically; attach independently audited predecessor run costs for any manual replacement"},
              "interpretation": {"A_selection_space": a_status, "B_pre_repair_identifiability": b_status,
                                 "C_structure_increment": {k: evidence(v) for k, v in structure.items()},
                                 "D_cost": ("synthetic only; real cost not tested" if fixture else
                                            "engineering threshold met on supplied single-Repair calibration" if d_pass else
                                            "insufficient calibration or cost threshold not met"),
                                 "E_full_search": "not tested", "main_method": main, "frozen_policy": policy,
                                 "advance_to_separate_full_search_review": not fixture and a_status == "exploratory positive evidence" and practical_pass and d_pass,
                                 "caveat": "No automatic full search. Thresholds are engineering gates. B failure does not establish that models are too small."},
              "lineage": {"manifest": manifest(run, c)["digest"], "predictions": pred["digest"], "reference": ref["digest"], "B": bb_export["digest"]},
              "analysis_seconds": time.perf_counter() - start}
    result = seal(report)
    if output:
        immutable(inside(output), result)
    return result
