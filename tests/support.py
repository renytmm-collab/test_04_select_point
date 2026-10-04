import numpy as np
from point_select.identity import ROOT


class TemporaryDirectory:
    """Avoid Python 3.12 Windows tempfile's user-only ACL under the sandbox token."""
    def __init__(self, dir):
        from pathlib import Path
        from uuid import uuid4
        self.path = Path(dir) / ("test-" + uuid4().hex)
        self.path.mkdir()
        self.name = str(self.path)

    def __enter__(self):
        return self.name

    def cleanup(self):
        import shutil
        path = self.path.resolve()
        if not path.is_relative_to((ROOT / "tests/_artifacts").resolve()) or path == (ROOT / "tests/_artifacts").resolve():
            raise RuntimeError("Test cleanup escaped its allowed directory")
        shutil.rmtree(path)

    def __exit__(self, *args):
        self.cleanup()


def synthetic_state():
    stream = np.random.RandomState(918)
    xy = stream.rand(101, 2).tolist()
    return {"locations": xy, "original_locations": xy, "demand": [0] + [1] * 100,
            "capacity": 50, "solution": [[[0, 0, 0]], [[0, 0, 0]], *[
                [[0, 0, 0], *[[j, 1, None] for j in range(k, k + 10)], [0, 0, 0]] for k in range(1, 101, 10)]]}


def actual_vrp_class():
    # Run the actual copied VRP class for structural checks without importing torch.
    # Tensor methods are never called. This is NOT a real model/Repair test.
    import ast
    path = ROOT / "vrp/vrp_problem.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VRPInstance")
    ns = {"np": np}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), ns)
    return ns["VRPInstance"]
