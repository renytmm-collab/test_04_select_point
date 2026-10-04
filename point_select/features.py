"""Small pre-Repair feature schema; fast path never restores/deep-copies an instance."""
import numpy as np
from .data import routes, validate_deletion, partial_solution
from .identity import digest

GEOMETRY = ("rms_radius", "center_mean", "center_max", "depot_mean", "contact_edge_mean")
STRUCTURE = ("involved_routes", "released_blocks", "release_concentration", "released_demand_capacity",
             "involved_slack_capacity_mean", "retained_singletons", "connectable_customer_endpoints")
NAMES = GEOMETRY + STRUCTURE
SCHEMA = {"version": 1, "geometry": list(GEOMETRY), "full": list(NAMES), "unit": "unit ALTR coordinates / normalized capacity"}


def cache(state):
    rr = routes(state)
    xy = np.asarray(state["locations"], dtype=float)
    demands = np.asarray(state["demand"], dtype=float)
    edge_a, edge_b = [], []
    for r in rr:
        seq = [0, *r, 0]
        edge_a.extend(seq[:-1]); edge_b.extend(seq[1:])
    a, b = np.asarray(edge_a), np.asarray(edge_b)
    return {"routes": rr, "xy": xy, "demand": demands, "capacity": state["capacity"],
            "a": a, "b": b, "edge_lengths": np.linalg.norm(xy[a] - xy[b], axis=1),
            "loads": [float(demands[r].sum()) for r in rr]}


def fast(c, customers, center):
    validate_deletion(customers)
    d = np.asarray(sorted(customers))
    selected = np.zeros(101, dtype=bool); selected[d] = True
    xy = c["xy"]; q = xy[d]
    dist = np.linalg.norm(q - np.asarray(center), axis=1)
    mask = selected[c["a"]] | selected[c["b"]]
    involved, blocks, counts, slack, singletons, endpoints = 0, 0, [], [], 0, 20
    for r, load in zip(c["routes"], c["loads"]):
        flags = selected[r]
        k = int(flags.sum())
        if not k:
            continue
        involved += 1; counts.append(k); slack.append((c["capacity"] - load) / c["capacity"])
        blocks += int(np.sum(flags & ~np.r_[False, flags[:-1]]))
        start = None
        for j in range(len(r) + 1):
            kept = j < len(r) and not flags[j]
            if kept and start is None:
                start = j
            if not kept and start is not None:
                length = j - start
                depot_left, depot_right = start == 0, j == len(r)
                singletons += int(length == 1 and not depot_left and not depot_right)
                endpoints += (1 if length == 1 and not depot_left and not depot_right else
                              int(not depot_left) + int(not depot_right))
                start = None
    values = [np.sqrt(np.mean(np.sum((q - q.mean(axis=0)) ** 2, axis=1))), dist.mean(), dist.max(),
              np.linalg.norm(q - xy[0], axis=1).mean(), c["edge_lengths"][mask].mean(),
              involved, blocks, sum((k / 20) ** 2 for k in counts), c["demand"][d].sum() / c["capacity"],
              np.mean(slack), singletons, endpoints]
    return [float(v) for v in values]


def rules(c, customers):
    selected = np.zeros(101, dtype=bool); selected[customers] = True
    mask = selected[c["a"]] | selected[c["b"]]
    return {"contact_edges": float(c["edge_lengths"][mask].sum() / c["edge_lengths"].sum()),
            "release_concentration": float(sum((selected[r].sum() / 20) ** 2 for r in c["routes"]))}


def audit(state, customers, center):
    """Independent scalar implementation + complete serialization, no Repair access."""
    validate_deletion(customers)
    rr = routes(state); selected = set(customers)
    xy = np.asarray(state["locations"]); q = xy[sorted(selected)]
    partial = partial_solution(state, customers)
    retained, incomplete, per_route = [], [], []
    for t in partial:
        released = len(t) == 1 and t[0][0] in selected
        if not released:
            retained.append(t)
        if t[0][0] != 0 or t[-1][0] != 0:
            incomplete.append(t)
    for r in rr:
        ks = [x for x in r if x in selected]
        if ks:
            block_count = sum(x in selected and (j == 0 or r[j - 1] not in selected) for j, x in enumerate(r))
            per_route.append({"released_count": len(ks), "blocks": block_count,
                              "load": sum(state["demand"][x] for x in r)})
    contact = [float(np.linalg.norm(xy[a[0]] - xy[b[0]])) for t in state["solution"]
               for a, b in zip(t, t[1:]) if a[0] in selected or b[0] in selected]
    centers = [float(np.linalg.norm(xy[x] - center)) for x in sorted(selected)]
    end_count = sum(1 if len(t) == 1 else int(t[0][0] != 0) + int(t[-1][0] != 0) for t in incomplete)
    values = [float(np.sqrt(np.mean(np.sum((q - q.mean(axis=0)) ** 2, axis=1)))),
              sum(centers) / 20, max(centers), sum(np.linalg.norm(xy[x] - xy[0]) for x in selected) / 20,
              sum(contact) / len(contact), len(per_route), sum(r["blocks"] for r in per_route),
              sum((r["released_count"] / 20) ** 2 for r in per_route),
              sum(state["demand"][x] for x in selected) / state["capacity"],
              sum((state["capacity"] - r["load"]) / state["capacity"] for r in per_route) / len(per_route),
              sum(len(t) == 1 and t[0][0] != 0 for t in retained), end_count]
    return {"values": [float(v) for v in values], "partial_solution": partial,
            "partial_digest": digest(partial), "retained_fragments": retained, "per_route": per_route,
            "input_points": 1 + end_count}
