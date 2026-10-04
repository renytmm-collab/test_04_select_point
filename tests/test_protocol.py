import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from support import synthetic_state, TemporaryDirectory
from point_select.identity import ROOT, seal, words, rng, digest
from point_select.data import pool, split_instances, point
from point_select.config import load, budget
from point_select.collection import jobs, label_seed, execute, labels, job_key
from point_select.analysis import metrics
from point_select.store import Ledger
from point_select.pipeline import freeze_reference, freeze_predictions


class ProtocolTest(unittest.TestCase):
    def test_budget_and_instance_split(self):
        c = load(ROOT / "configs/pilot.template.json"); b = budget(c)
        self.assertEqual(b["total"], 71936)
        self.assertEqual(b["collect-dev-a"], 24576); self.assertEqual(b["collect-test-a"], 8192)
        s = split_instances(c)
        self.assertEqual(len(s["development"]), 24); self.assertEqual(len(s["test"]), 8)
        self.assertFalse(set(s["development"]) & set(s["test"]))
        self.assertEqual(s, split_instances(c))
        self.assertEqual(set(s["folds"]), set(map(str, s["development"])))

    def test_unapproved_caps_and_policy_block_before_backend_or_writes(self):
        from point_select.pipeline import states, policy_load
        c = load(ROOT / "configs/pilot.template.json")
        with patch("point_select.pipeline.lock") as mocked_lock:
            with self.assertRaises(ValueError):
                states(ROOT / "tests/_artifacts/must-not-exist", c, None)
            mocked_lock.assert_not_called()
        self.assertFalse((ROOT / "tests/_artifacts/must-not-exist").exists())
        with self.assertRaises(ValueError):
            policy_load(ROOT / "configs/decision.policy.template.json")

    def test_duplicate_slots_shared_labels_and_centers(self):
        c = load(ROOT / "configs/pilot.template.json"); state = synthetic_state()
        with patch("point_select.data.point", side_effect=[(list(range(1, 21)), [.1 + j / 100, .2]) for j in range(16)]):
            t = pool(state, c, 1, 50)
        self.assertEqual(len(t["slots"]), 16); self.assertEqual(len(t["unique"]), 1)
        self.assertEqual(len(jobs([t], 32)), 32)
        self.assertNotEqual(t["slots"][0]["center"], t["slots"][1]["center"])
        j = jobs([t], 32)[0]
        self.assertNotEqual(label_seed(c["root_seed"], "A", j), label_seed(c["root_seed"], "B", j))
        self.assertEqual(label_seed(c["root_seed"], "A", j), label_seed(c["root_seed"], "A", {**j, "method": "other"}))

    def test_per_rollout_truncation_sparse_events(self):
        r = metrics([{"y": .02, "z": .02}, {"y": -.04, "z": 0}])
        self.assertAlmostEqual(r["mean_z_pp"], 1.)
        self.assertLess(r["mean_y_pp"], 0)
        self.assertEqual(r["positive_events"], 1)
        zeros = metrics([{"y": 0., "z": 0.}] * 128)
        self.assertIsNone(zeros["positive_conditional_y_pp"])

    def test_reference_reads_a_never_b(self):
        from contextlib import contextmanager
        c = load(ROOT / "configs/pilot.template.json")
        seen = []; captured = []
        t = {"instance": 1, "step": 50, "unique": {"a": [1], "b": [2]}}
        def reader(run, name, filename="data.json"):
            seen.append(name)
            return {"states": {"split": {"test": [1]}}, "candidates": {"tables": [t]},
                    "collect-test-a": {"digest": "A"}}[name]
        @contextmanager
        def cm(*args):
            yield ROOT / "tests/_artifacts"
        aa = {(1, 50, "a"): [{"z": .1}], (1, 50, "b"): [{"z": .2}]}
        with patch("point_select.pipeline.lock", cm), patch("point_select.pipeline.stage", cm), \
             patch("point_select.pipeline.allowed"), patch("point_select.pipeline.lineage", return_value={}), \
             patch("point_select.pipeline.artifact", reader), patch("point_select.pipeline.labels", return_value=aa), \
             patch("point_select.pipeline.immutable", side_effect=lambda p, v: captured.append(v)):
            freeze_reference("unused", c)
        self.assertNotIn("collect-test-b", seen)
        self.assertEqual(captured[0]["winners"][0]["candidate"], "b")

    def test_prediction_path_never_reads_test_labels(self):
        from contextlib import contextmanager
        from point_select.features import cache, fast
        c = load(ROOT / "configs/pilot.template.json"); state = synthetic_state(); t = pool(state, c, 1, 50)
        seen = []; cc = cache(state)
        f = {"instance": 1, "step": 50, "slots": [{"values": fast(cc, t["unique"][s["candidate"]], s["center"])} for s in t["slots"]]}
        def reader(run, name, filename="data.json"):
            seen.append(name)
            return {"states": {"split": {"test": [1], "development": []}, "states": [{"instance": 1, "step": 50, "state": state}]},
                    "candidates": {"tables": [t], "digest": "c"}, "features": {"tables": [f]},
                    "fit": {"models": {}, "main_method": "constant", "digest": "fit"}}[name]
        @contextmanager
        def cm(*args):
            yield ROOT / "tests/_artifacts"
        with patch("point_select.pipeline.lock", cm), patch("point_select.pipeline.stage", cm), \
             patch("point_select.pipeline.allowed"), patch("point_select.pipeline.lineage", return_value={}), \
             patch("point_select.pipeline.policy_load", return_value={"min_gain_pp": 0, "max_online_repair_ratio": 1}), \
             patch("point_select.pipeline.artifact", reader), patch("point_select.pipeline.immutable"):
            freeze_predictions("unused", c, "unused")
        self.assertEqual(seen, ["states", "candidates", "features", "fit"])

    def test_failure_batch_not_silently_discarded(self):
        class Fake:
            def sync(self):
                pass
            def repair(self, instances, streams):
                raise RuntimeError("synthetic Repair failure")
        class V:
            def get_max_nb_input_points(self):
                return 4
        with TemporaryDirectory(dir=ROOT / "tests/_artifacts") as temp:
            ledger = Ledger(temp)
            tasks = [{"instance": 1, "step": 50, "candidate": "same", "repeat": j} for j in range(2)]
            with self.assertRaises(RuntimeError):
                execute(ledger, tasks, 2, 2, lambda j: (V(), 10.), Fake(), 7, "A")
            self.assertEqual(ledger.count(), 2)
            with self.assertRaises(RuntimeError):
                execute(ledger, tasks, 2, 4, lambda j: (V(), 10.), Fake(), 7, "A")
            ledger.export_close({"wall_seconds": 0})

    def test_full_staged_synthetic_integration_and_readonly_analysis(self):
        import copy
        import sys
        import types
        from point_select import pipeline as p
        from point_select.analysis import analyze, audit_run
        from point_select.identity import read, verify
        from point_select.store import immutable
        from support import actual_vrp_class
        cls = actual_vrp_class()
        def restore(state):
            v = cls(100, np.asarray(state["locations"]), np.asarray(state["original_locations"]), np.asarray(state["demand"]), 50)
            v.solution = copy.deepcopy(state["solution"])
            return v
        def payload(v):
            return {"locations": v.locations.tolist(), "original_locations": v.original_locations.tolist(),
                    "demand": v.demand.tolist(), "capacity": v.capacity, "solution": v.get_solution_copy()}
        def generate(c, i):
            state = synthetic_state(); xy = rng(c["root_seed"], "instance", i).rand(101, 2).tolist()
            state.update(locations=xy, original_locations=xy)
            return restore(state)
        def destroy(v, ids):
            v.destroy(sorted(ids)); return digest(v.solution)
        class Fake:
            def __init__(self, c):
                self.environment = {"fixture": True, "device": "synthetic-no-torch"}
            def sync(self):
                pass
            def peak(self):
                return {"allocated": None, "reserved": None}
            def repair(self, instances, streams):
                # Reconstruct a complete capacity-feasible solution with randomized tour order.
                # This is a deterministic-seeded synthetic replacement, NOT original Repair.
                for v, stream in zip(instances, streams):
                    tours = synthetic_state()["solution"]
                    for tour in tours:
                        if len(tour) > 3:
                            j, k = stream.choice(range(1, len(tour) - 1), 2, replace=False)
                            tour[j], tour[k] = tour[k], tour[j]
                    v.solution = tours
        fake = types.ModuleType("point_select.backend")
        fake.Backend, fake.restore, fake.payload, fake.generate, fake.destroy_checked = Fake, restore, payload, generate, destroy
        c = load(ROOT / "configs/pilot.template.json")
        c = {**c, "instances": 8, "development_instances": 4, "state_steps": [1, 2], "slots": 3,
             "A_repeats": 2, "B_repeats": 3, "batch_size": 3, "device": "cpu", "bootstrap_repeats": 20}
        with TemporaryDirectory(dir=ROOT / "tests/_artifacts") as temp, patch.dict(sys.modules, {"point_select.backend": fake}):
            run = Path(temp) / "run"; b = budget(c)
            benchmark_run = Path(temp) / "benchmark"
            p.benchmark(benchmark_run, c, 3, 1, 1, 2)
            p.states(run, c, b["states"])
            dev_id = verify(read(run / "stages/states/data.json"))["split"]["development"][0]
            matched_benchmark = Path(temp) / "benchmark-matched"
            p.benchmark(matched_benchmark, c, 3, 1, 1, 2, run / f"stages/states/snapshot-{dev_id}-1.json")
            p.candidates(run, c); p.features(run, c)
            p.collect(run, c, "collect-dev-a", b["collect-dev-a"])
            p.fit(run, c, 60)
            policy_path = Path(temp) / "policy.json"
            immutable(policy_path, {"schema": 1, "min_gain_pp": 0, "max_online_repair_ratio": 1})
            p.freeze_predictions(run, c, policy_path, [benchmark_run, matched_benchmark])
            p.collect(run, c, "collect-test-a", b["collect-test-a"])
            p.freeze_reference(run, c)
            p.collect(run, c, "collect-test-b", b["collect-test-b"])
            before_files = {str(f.relative_to(run)): (f.stat().st_size, f.stat().st_mtime_ns) for f in run.rglob("*") if f.is_file()}
            a = audit_run(run, c); report = analyze(run, c, benchmark_runs=[benchmark_run, matched_benchmark])
            after_files = {str(f.relative_to(run)): (f.stat().st_size, f.stat().st_mtime_ns) for f in run.rglob("*") if f.is_file()}
            self.assertEqual(before_files, after_files)
            self.assertTrue(a["fail_closed"]); self.assertEqual(len(report["per_instance"]), 4)
            self.assertEqual(report["evidence_level"], "synthetic")
            self.assertFalse(report["interpretation"]["advance_to_separate_full_search_review"])
            self.assertIn("ridge", report["structure_ablation"])
            self.assertIsNotNone(report["timing"]["single_repair_ratio"])
            self.assertIsNotNone(report["timing"]["matched_development_single_repair_ratio"])
            self.assertEqual(report["lineage"]["predictions"], verify(read(run / "stages/freeze-test-predictions/data.json"))["digest"])
            late_benchmark = Path(temp) / "benchmark-late"
            p.benchmark(late_benchmark, c, 3, 1, 1, 2)
            with self.assertRaises(ValueError):
                analyze(run, c, benchmark_runs=[late_benchmark])

    def test_cli_without_start_has_no_filesystem_side_effects(self):
        import subprocess
        import sys
        with TemporaryDirectory(dir=ROOT / "tests/_artifacts") as temp:
            run = Path(temp) / "must-not-exist"
            result = subprocess.run([sys.executable, "-B", str(ROOT / "point_select_cli.py"), "states", "--run", str(run)],
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Explicit --start", result.stderr)
            self.assertFalse(run.exists())
