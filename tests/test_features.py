import copy
import unittest
import numpy as np
from point_select.data import partial_solution, point, validate_deletion
from point_select.features import fast, audit, cache
from support import synthetic_state, actual_vrp_class


class FeaturesTest(unittest.TestCase):
    def test_point_matches_original_and_deletes_exactly_twenty(self):
        state = synthetic_state(); cls = actual_vrp_class()
        v = cls(100, np.asarray(state["locations"]), np.asarray(state["original_locations"]), np.asarray(state["demand"]), 50)
        v.solution = copy.deepcopy(state["solution"])
        stream = np.random.RandomState(42)
        ids, center = point(state, stream)
        old = np.random.get_state()
        try:
            np.random.seed(42); v.destroy_point_based(.2)
        finally:
            np.random.set_state(old)
        self.assertEqual(v.solution, partial_solution(state, ids))
        self.assertEqual(len(ids), 20); self.assertNotIn(0, ids)
        with self.assertRaises(ValueError):
            validate_deletion([0, *range(1, 20)])

    def test_fast_audit_and_actual_destroy_endpoints(self):
        state = synthetic_state(); before = copy.deepcopy(state); cls = actual_vrp_class()
        selections = [list(range(1, 21)), list(range(1, 41, 2)), list(range(81, 101))]
        selections += [point(state, np.random.RandomState(j))[0] for j in range(20)]
        for ids in selections:
            center = [.3, .8]; record = audit(state, ids, center)
            np.testing.assert_allclose(fast(cache(state), ids, center), record["values"], rtol=1e-12, atol=1e-12)
            v = cls(100, np.asarray(state["locations"]), np.asarray(state["original_locations"]), np.asarray(state["demand"]), 50)
            v.solution = copy.deepcopy(state["solution"]); v.destroy(sorted(ids))
            self.assertEqual(v.solution, record["partial_solution"])
            self.assertEqual(v.get_max_nb_input_points(), record["input_points"])
        self.assertEqual(state, before)

    def test_pre_repair_only(self):
        from point_select.identity import digest
        state = synthetic_state(); ids = list(range(1, 21))
        baseline = fast(cache(state), ids, [.4, .6])
        state["Repair_result"] = {"z": 100, "instance_id": 97}
        self.assertEqual(baseline, fast(cache(state), ids, [.4, .6]))
