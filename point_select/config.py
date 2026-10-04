from .identity import read, WEIGHT, WEIGHT_SHA


def load(path):
    c = read(path)
    if c.get("schema") != 1 or c.get("blueprint") != "ALTR_100" or c.get("K") != 20:
        raise ValueError("Only frozen ALTR_100 / K=20 protocol supported")
    if c["checkpoint"] != WEIGHT or c["checkpoint_sha256"] != WEIGHT_SHA:
        raise ValueError("Official checkpoint specification changed")
    for k in ("root_seed", "instances", "development_instances", "slots", "A_repeats", "B_repeats",
              "batch_size", "torch_threads", "bootstrap_repeats"):
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError("Positive integer required: " + k)
    if c["root_seed"] in (2026100203,):
        raise ValueError("Old Point pilot seed is forbidden")
    if not 4 <= c["development_instances"] < c["instances"] or c["development_instances"] % 4:
        raise ValueError("Development count must be divisible by four; nonempty test required")
    if c["state_steps"] != sorted(set(c["state_steps"])) or any(type(s) is not int or s < 1 for s in c["state_steps"]):
        raise ValueError("Distinct increasing positive state steps required")
    if not c["state_steps"] or c["pointer_hidden_size"] != 128:
        raise ValueError("State steps required / official hidden size must be 128")
    if c["device"] != "cpu" and not c["device"].startswith("cuda"):
        raise ValueError("Device must be cpu or cuda[:index]")
    if c["folds"] != 4 or c["model_grid"] != {"ridge_alpha": [0.01, 0.1, 1.0], "tree_depth": [2, 3], "tree_min_leaf": [16, 32]}:
        raise ValueError("Frozen four-fold small model grid changed; version protocol first")
    return c


def budget(c):
    ns = len(c["state_steps"])
    n, dev, slots = c["instances"], c["development_instances"], c["slots"]
    b = {"states": n * max(c["state_steps"]),
         "collect-dev-a": dev * ns * slots * c["A_repeats"],
         "collect-test-a": (n - dev) * ns * slots * c["A_repeats"],
         "collect-test-b": (n - dev) * ns * slots * c["B_repeats"]}
    return {**b, "total": sum(b.values()), "benchmark_and_failed_attempts": "additional",
            "scope": "exploratory upper bounds before candidate deduplication; no power analysis"}


def require_cap(c, stage, cap, planned):
    if cap is None or cap < planned:
        raise ValueError(f"Explicit --cap >= {planned} required before any write or model load")
    approved = c["caps"].get(stage)
    if approved is not None and cap > approved:
        raise ValueError("CLI cap exceeds frozen configuration cap")
    return int(cap)
