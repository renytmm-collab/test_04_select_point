import unittest
import numpy as np
from point_select.models import fit_model, predict, preprocess_fit, transform, train, selection_metric
from point_select.data import split_instances, pick
from point_select.config import load
from point_select.identity import ROOT


class ModelTest(unittest.TestCase):
    def test_known_ranking_and_zero_targets(self):
        x = np.linspace(-2, 2, 80)[:, None]
        y = .01 + .002 * x[:, 0]; w = np.ones(80)
        for spec in ({"family": "ridge", "alpha": .01}, {"family": "tree", "depth": 3, "min_leaf": 4}):
            m = fit_model(x, y, w, spec)
            p = predict(m, x)
            self.assertGreater(p[-1], p[0])
            self.assertTrue(np.all(np.diff(p) >= -1e-12))
            z = fit_model(x, np.zeros(80), w, spec)
            np.testing.assert_array_equal(predict(z, x), np.zeros(80))

    def test_fold_local_preprocessing(self):
        x = np.array([[1., np.nan], [3., 4.]])
        p = preprocess_fit(x, np.ones(2)); heldout = [[1e9, 1e9]]
        transform(heldout, p)
        self.assertEqual(p["mean"], [2., 4.])
        self.assertEqual(p["median"], [2., 4.])

    def test_train_never_accepts_test_instance(self):
        c = load(ROOT / "configs/pilot.template.json"); split = split_instances(c)
        row = {"instance": split["test"][0], "step": 50, "values": [0.] * 12, "target": 0., "weight": 1.}
        with self.assertRaises(ValueError):
            train([row], split, c)

    def test_actual_four_fold_grouping_and_known_selection(self):
        c = load(ROOT / "configs/pilot.template.json"); c = {**c, "instances": 8, "development_instances": 4}
        split = split_instances(c); rows = []
        for i in split["development"]:
            for step in (50, 200):
                for slot in range(16):
                    rows.append({"instance": i, "step": step, "values": [float(slot)] * 12,
                                 "target": .001 * slot, "weight": 1 / 128})
        result = train(rows, split, c)
        self.assertEqual(result["fit_count"], 60)
        for r in result["cv"]:
            for f in r["folds"]:
                self.assertFalse(set(f["train_instances"]) & set(f["validation_instances"]))
        self.assertGreater(max(r["selection_gain"] for r in result["cv"]), .006)
        for model in result["models"].values():
            width = len(model["preprocessing"]["mean"])
            scores = predict(model, np.array([r["values"][:width] for r in rows]))
            gains = selection_metric(rows, scores, c["root_seed"], "synthetic")
            self.assertGreater(np.mean(list(gains.values())), 0)

    def test_tie_is_deterministic_and_native_weighted(self):
        self.assertEqual(pick([0, 0, 0], 7, 1, 50, "constant"), pick([0, 0, 0], 7, 1, 50, "constant"))
        counts = np.zeros(3)
        for i in range(600):
            counts[pick([0, 0, 0], 7, i, 50, "constant")] += 1
        self.assertTrue((counts > 150).all())
