from support import TemporaryDirectory
import unittest
from pathlib import Path
from unittest.mock import patch
from point_select.identity import ROOT, seal, read, verify
from point_select.store import stage, immutable, Ledger
from point_select.guards import allowed, ORDER, completed


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(dir=ROOT / "tests/_artifacts")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_started_failed_stage_cannot_repeat(self):
        with self.assertRaises(RuntimeError):
            with stage(self.root, "failed-stage", {}):
                raise RuntimeError("synthetic crash")
        with self.assertRaises(RuntimeError):
            with stage(self.root, "failed-stage", {}):
                pass
        immutable(self.root / "stages/uncertain/started.json", seal({"stage": "uncertain"}))
        with self.assertRaises(RuntimeError):
            with stage(self.root, "uncertain", {}):
                pass

    def test_task_cannot_overwrite_or_retry(self):
        d = self.root / "ledger"; d.mkdir()
        ledger = Ledger(d)
        ledger.begin("one", {"repair_invoked": False})
        with self.assertRaises(RuntimeError):
            ledger.begin("one", {})
        ledger.finish("one", "failed", {"repair_invoked": True})
        with self.assertRaises(RuntimeError):
            ledger.finish("one", "ok", {})
        ledger.begin("two", {"repair_invoked": False})
        ledger.export_close({"wall_seconds": 0})
        ex = verify(read(d / "export.json"))
        self.assertEqual(ex["costs"]["uncertain"], 1)
        self.assertEqual(ex["costs"]["repair_invocations_known"], 1)

    def test_test_labels_blocked_until_predictions_and_reference(self):
        c = {}
        with patch("point_select.guards.manifest", return_value={}):
            for name in ORDER[:5]:
                with stage(self.root, name, {}):
                    pass
            with self.assertRaises(FileNotFoundError):
                allowed(self.root, "collect-test-a", c)
            with stage(self.root, "freeze-test-predictions", {}):
                pass
            allowed(self.root, "collect-test-a", c)
            with stage(self.root, "collect-test-a", {}):
                pass
            with self.assertRaises(FileNotFoundError):
                allowed(self.root, "collect-test-b", c)
            with stage(self.root, "freeze-test-reference", {}):
                pass
            allowed(self.root, "collect-test-b", c)

    def test_completed_artifact_change_detected(self):
        with stage(self.root, "some", {}) as d:
            immutable(d / "data.json", seal({"x": 1}))
        completed(self.root, "some")
        (d / "data.json").write_text('{"x":2}', encoding="utf-8")
        with self.assertRaises(RuntimeError):
            completed(self.root, "some")
