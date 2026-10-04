"""Weighted Ridge and restricted CART, portable JSON models (no pickle)."""
import time
import numpy as np
from .data import pick
from .identity import digest, words


def preprocess_fit(x, weights):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0.)
    filled = np.where(np.isfinite(x), x, med)
    mean = np.average(filled, axis=0, weights=weights)
    scale = np.sqrt(np.average((filled - mean) ** 2, axis=0, weights=weights))
    scale = np.where(scale > 1e-12, scale, 1.)
    return {"median": med.tolist(), "mean": mean.tolist(), "scale": scale.tolist()}


def transform(x, p):
    a = np.asarray(x, dtype=float)
    return (np.where(np.isfinite(a), a, p["median"]) - p["mean"]) / p["scale"]


def tree_fit(x, y, w, depth, leaf):
    value = float(np.average(y, weights=w))
    node = {"value": value, "rows": len(y)}
    if depth == 0 or len(y) < 2 * leaf or np.max(y) == np.min(y):
        return node
    base = float(np.sum(w * (y - value) ** 2))
    best = None
    # Cumulative weighted SSE avoids quadratic enumeration of thresholds.
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind="stable")
        a, b, ww = x[order, j], y[order], w[order]
        sw, sy, sy2 = np.cumsum(ww), np.cumsum(ww * b), np.cumsum(ww * b * b)
        for k in range(leaf - 1, len(y) - leaf):
            if a[k] == a[k + 1]:
                continue
            rw = sw[-1] - sw[k]
            loss = sy2[k] - sy[k] ** 2 / sw[k] + sy2[-1] - sy2[k] - (sy[-1] - sy[k]) ** 2 / rw
            threshold = float(a[k] + (a[k + 1] - a[k]) / 2)
            if threshold >= a[k + 1]:
                threshold = float(a[k])
            key = (float(loss), j, threshold)
            if best is None or key < best[0]:
                best = (key, j, threshold)
    if best is None or best[0][0] >= base - 1e-18:
        return node
    _, j, threshold = best
    left = x[:, j] <= threshold
    return {**node, "feature": j, "threshold": threshold,
            "left": tree_fit(x[left], y[left], w[left], depth - 1, leaf),
            "right": tree_fit(x[~left], y[~left], w[~left], depth - 1, leaf)}


def fit_model(x, y, weights, spec):
    w = np.asarray(weights, dtype=float)
    w = w * len(w) / w.sum()  # alpha has consistent mean-row scale across folds.
    y = np.asarray(y, dtype=float)
    p = preprocess_fit(x, w)
    xx = transform(x, p)
    model = {"spec": spec, "preprocessing": p}
    if spec["family"] == "ridge":
        intercept = float(np.average(y, weights=w))
        a = xx.T @ (w[:, None] * xx) + spec["alpha"] * np.eye(xx.shape[1])
        b = xx.T @ (w * (y - intercept))
        model.update(intercept=intercept, coefficients=np.linalg.solve(a, b).tolist())
    else:
        model["tree"] = tree_fit(xx, y, w, spec["depth"], spec["min_leaf"])
    return model


def predict(model, x):
    xx = transform(x, model["preprocessing"])
    if model["spec"]["family"] == "ridge":
        return xx @ np.asarray(model["coefficients"]) + model["intercept"]
    result = []
    for row in xx:
        node = model["tree"]
        while "feature" in node:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        result.append(node["value"])
    return np.asarray(result)


def specs(c):
    return [{"schema": schema, "family": "ridge", "alpha": alpha}
            for schema in ("geometry", "full") for alpha in c["model_grid"]["ridge_alpha"]] + [
            {"schema": schema, "family": "tree", "depth": d, "min_leaf": leaf}
            for schema in ("geometry", "full") for d in c["model_grid"]["tree_depth"]
            for leaf in c["model_grid"]["tree_min_leaf"]]


def simplicity(spec):
    return (0 if spec["family"] == "ridge" else 1, 0 if spec["schema"] == "geometry" else 1,
            -spec.get("alpha", 0), spec.get("depth", 0), -spec.get("min_leaf", 0))


def best_result(results):
    best = max(r["selection_gain"] for r in results)
    return min((r for r in results if best - r["selection_gain"] <= 1e-12), key=lambda r: simplicity(r["spec"]))


def selection_metric(rows, scores, root, tag):
    by_state = {}
    for j, row in enumerate(rows):
        by_state.setdefault((row["instance"], row["step"]), []).append(j)
    by_instance = {}
    for (i, step), indices in by_state.items():
        chosen = indices[pick([scores[j] for j in indices], root, i, step, tag)]
        gain = rows[chosen]["target"] - np.mean([rows[j]["target"] for j in indices])
        by_instance.setdefault(i, []).append(float(gain))
    return {i: float(np.mean(v)) for i, v in by_instance.items()}


def train(rows, split, c):
    start = time.perf_counter()
    if set(r["instance"] for r in rows) != set(split["development"]):
        raise ValueError("Training input must contain exactly development instances")
    y = np.asarray([r["target"] for r in rows])
    weights = np.asarray([r["weight"] for r in rows])
    folds = np.asarray([split["folds"][str(r["instance"])] for r in rows])
    results = []
    for spec in specs(c):
        width = 5 if spec["schema"] == "geometry" else 12
        x = np.asarray([r["values"][:width] for r in rows])
        oof = np.empty(len(rows)); fold_info = []
        # Same deployed method tie stream across all its hyperparameters:
        # grid search must not optimize a different random tie lottery per alpha/depth.
        tag = f"{spec['family']}_{spec['schema']}"
        for f in range(4):
            train_idx, valid_idx = folds != f, folds == f
            model = fit_model(x[train_idx], y[train_idx], weights[train_idx], spec)
            oof[valid_idx] = predict(model, x[valid_idx])
            fold_info.append({"fold": f, "train_instances": sorted(set(rows[j]["instance"] for j in np.flatnonzero(train_idx))),
                              "validation_instances": sorted(set(rows[j]["instance"] for j in np.flatnonzero(valid_idx))),
                              "preprocessing": model["preprocessing"]})
        per_instance = selection_metric(rows, oof, c["root_seed"], tag)
        results.append({"spec": spec, "selection_gain": float(np.mean(list(per_instance.values()))),
                        "per_instance_gain": {str(k): v for k, v in per_instance.items()},
                        "mse": float(np.average((oof - y) ** 2, weights=weights)), "folds": fold_info})
    fitted = {}
    for schema in ("geometry", "full"):
        for family in ("ridge", "tree"):
            r = best_result([r for r in results if r["spec"]["schema"] == schema and r["spec"]["family"] == family])
            width = 5 if schema == "geometry" else 12
            fitted[f"{family}_{schema}"] = fit_model([r["values"][:width] for r in rows], y, weights, r["spec"])
    winner = best_result(results)["spec"]
    return {"models": fitted, "main_method": f"{winner['family']}_{winner['schema']}",
            "cv": results, "folds": split["folds"], "model_seed_words": words(c["root_seed"], "model"),
            "model_rng_usage": "deterministic algorithms; no random model fitting",
            "fit_count": len(specs(c)) * 4 + 4, "training_seconds": time.perf_counter() - start,
            "selection_rule": "instance-equal held-out A selection gain; ties within 1e-12 prefer Ridge, geometry, stronger regularization, shallower tree"}
