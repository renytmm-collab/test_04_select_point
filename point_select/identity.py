import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHT = "trained_models/cvrp/altr_C_100/model_incumbent_vrpRl13_C_ALTR_100_P_0.2_80419.pt"
WEIGHT_SHA = "777178c30819f73b387674c29a2430cc06b0fda876792d5e08137d5ea3f94cf8"
STREAMS = ("instance", "split", "state_destroy", "state_repair", "candidate_point",
           "development_folds", "A", "B", "model", "bootstrap", "benchmark", "tie")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def seal(value):
    return {**value, "digest": digest(value)}


def verify(value):
    if value.get("digest") != digest({k: v for k, v in value.items() if k != "digest"}):
        raise ValueError("Artifact digest mismatch")
    return value


def words(root, stream, *parts):
    if stream not in STREAMS:
        raise ValueError("Unregistered RNG stream")
    raw = hashlib.sha256(canonical([root, "point-select-v1", stream, *parts]).encode()).digest()[:16]
    return [int.from_bytes(raw[j:j + 4], "little") for j in range(0, 16, 4)]


def rng(root, stream, *parts):
    import numpy as np
    return np.random.RandomState(words(root, stream, *parts))


def inside(path):
    p = Path(path).resolve()
    if p == ROOT or not p.is_relative_to(ROOT):
        raise ValueError("Output must stay under project root (including resolved symlinks)")
    return p


def code_identity():
    paths = [*ROOT.glob("*.py"), *(ROOT / "point_select").glob("*.py"),
             *(ROOT / "vrp").glob("*.py"), *(ROOT / "vrp/dataset_blueprints").glob("*.py"),
             *(ROOT / "tests").glob("*.py"), ROOT / "protocol/POINT_SELECTION_PROTOCOL.md",
             ROOT / "protocol/UPSTREAM_BASELINE.json", ROOT / "requirements.txt"]
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(
        p.read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest() for p in sorted(paths)}
    return {"files": files, "digest": digest(files),
            "history_sha256": file_sha(ROOT / "protocol/historical_instances.json"),
            "checkpoint_sha256": file_sha(ROOT / WEIGHT)}


def content_hash(state):
    # Exactly the reference project's canonical content identity.
    return digest({k: state[k] for k in ("original_locations", "demand", "capacity")})


def candidate_id(state_digest, customers):
    return digest({"state": state_digest, "customers": sorted(customers)})
