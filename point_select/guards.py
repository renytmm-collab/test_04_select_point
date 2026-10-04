from pathlib import Path
from .identity import read, verify, digest, code_identity, file_sha

ORDER = ("states", "candidates", "features", "collect-dev-a", "fit", "freeze-test-predictions",
         "collect-test-a", "freeze-test-reference", "collect-test-b")


def manifest(run, c):
    import numpy as np
    m = verify(read(Path(run) / "manifest.json"))
    if m["config"] != c or m["identity"] != code_identity():
        raise RuntimeError("Frozen config/code/history/checkpoint changed; new version/run required")
    if m["runtime"]["numpy"] != np.__version__:
        raise RuntimeError("Run NumPy version changed; Point sorting/analysis reproducibility requires matching version")
    return m


def completed(run, name):
    d = Path(run) / "stages" / name
    if (d / "failed.json").exists():
        raise RuntimeError("Failed stage: " + name)
    receipt = verify(read(d / "complete.json"))
    for filename, sha in receipt["files"].items():
        if file_sha(d / filename) != sha:
            raise RuntimeError("Completed stage artifact changed: " + name + "/" + filename)
    return receipt


def artifact(run, name, filename="data.json"):
    completed(run, name)
    return verify(read(Path(run) / "stages" / name / filename))


def allowed(run, name, c):
    manifest(run, c)
    pos = ORDER.index(name)
    for prior in ORDER[:pos]:
        completed(run, prior)
    for later in ORDER[pos:]:
        if (Path(run) / "stages" / later).exists():
            raise RuntimeError("Existing current/future stage blocks mutation: " + later)


def dependencies(run, names):
    return {name: digest(completed(run, name)) for name in names}
