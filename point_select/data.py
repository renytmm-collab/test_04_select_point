import copy
import numpy as np
from .identity import digest, candidate_id, rng, words


def validate_deletion(customers):
    if len(customers) != 20 or len(set(customers)) != 20 or not set(customers) <= set(range(1, 101)):
        raise ValueError("Exactly 20 distinct non-depot customers required")


def routes(state):
    rr = []
    for tour in state["solution"]:
        if not tour or tour[0][0] != 0 or tour[-1][0] != 0:
            raise ValueError("Complete depot-closed solution required")
        ids = [e[0] for e in tour[1:-1]]
        if 0 in ids:
            raise ValueError("Internal depot")
        if ids:
            rr.append(ids)
    if sorted(i for r in rr for i in r) != list(range(1, 101)):
        raise ValueError("Every customer must occur exactly once")
    if any(sum(state["demand"][i] for i in r) > state["capacity"] for r in rr):
        raise ValueError("Capacity infeasible snapshot")
    return rr


def cost(state):
    xy = np.asarray(state["original_locations"])
    return float(sum(np.linalg.norm(xy[a[0]] - xy[b[0]]) for tour in state["solution"]
                     for a, b in zip(tour, tour[1:])))


def point(state, stream):
    p = stream.rand(1, 2)
    xy = np.asarray(state["locations"])
    ids = (np.argsort(np.sum((xy[1:] - p) ** 2, axis=1))[:20] + 1).tolist()
    validate_deletion(ids)
    return sorted(ids), p[0].tolist()


def partial_solution(state, customers):
    """Independent serialization specification, including unchanged empty depot tours."""
    validate_deletion(customers)
    chosen = set(customers)
    out = []
    for tour in state["solution"]:
        last = 0
        for j in range(1, len(tour) - 1):
            if tour[j][0] in chosen:
                if j > last and j > 1:
                    out.append(copy.deepcopy(tour[last:j]))
                out.append([[tour[j][0], int(state["demand"][tour[j][0]]), None]])
                last = j + 1
        if last:
            if last < len(tour) - 1:
                out.append(copy.deepcopy(tour[last:]))
        else:
            out.append(copy.deepcopy(tour))
    return out


def pool(state, c, i, step):
    sd = digest(state)
    slots, unique = [], {}
    for j in range(c["slots"]):
        ids, center = point(state, rng(c["root_seed"], "candidate_point", i, step, j))
        cid = candidate_id(sd, ids)
        unique[cid] = ids
        slots.append({"slot": j, "candidate": cid, "center": center,
                      "seed_words": words(c["root_seed"], "candidate_point", i, step, j)})
    return {"instance": i, "step": step, "state_digest": sd, "slots": slots, "unique": unique}


def split_instances(c):
    order = rng(c["root_seed"], "split").permutation(c["instances"]).tolist()
    dev = sorted(order[:c["development_instances"]])
    test = sorted(order[c["development_instances"]:])
    folded = rng(c["root_seed"], "development_folds").permutation(dev).tolist()
    folds = {str(i): j % 4 for j, i in enumerate(folded)}
    return {"development": dev, "test": test, "folds": folds,
            "split_seed": words(c["root_seed"], "split"),
            "fold_seed": words(c["root_seed"], "development_folds")}


def pick(scores, root, i, step, method):
    a = np.asarray(scores, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Finite nonempty scores required")
    tied = np.flatnonzero(a == a.max())
    return int(rng(root, "tie", i, step, method).choice(tied))
