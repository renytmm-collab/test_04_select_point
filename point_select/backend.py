"""Lazy real backend; imported only after explicit start/cap checks."""
import copy
import os
from types import SimpleNamespace
import numpy as np
# Set before importing torch / initializing CUDA. Avoid driver cache writes outside the project.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["CUDA_CACHE_DISABLE"] = "1"
import torch
from vrp.data_utils import create_dataset
from vrp.vrp_problem import VRPInstance
from actor import VrpActorModel
import repair
from .identity import ROOT, WEIGHT, WEIGHT_SHA, file_sha, words, digest


def payload(instance):
    return {"locations": instance.locations.tolist(), "original_locations": instance.original_locations.tolist(),
            "demand": instance.demand.tolist(), "capacity": int(instance.capacity),
            "solution": instance.get_solution_copy()}


def restore(state):
    v = VRPInstance(100, np.asarray(state["locations"]), np.asarray(state["original_locations"]),
                    np.asarray(state["demand"], dtype=int), state["capacity"])
    v.solution = copy.deepcopy(state["solution"])
    return v


def generate(c, i):
    old = np.random.get_state()
    try:
        np.random.seed(words(c["root_seed"], "instance", i))
        return create_dataset(1, SimpleNamespace(instance_blueprint=c["blueprint"]), create_solution=True)[0]
    finally:
        np.random.set_state(old)


def destroy_checked(v, customers):
    from .data import validate_deletion, partial_solution
    validate_deletion(customers)
    expected = partial_solution(payload(v), customers)
    v.destroy(sorted(customers))
    if v.solution != expected:
        raise RuntimeError("Original Destroy serialization mismatch")
    return digest(v.solution)


class Backend:
    def __init__(self, c):
        if np.__version__ != "1.26.4":
            raise RuntimeError("Real experiments require NumPy==1.26.4 to preserve Point sorting semantics")
        if file_sha(ROOT / WEIGHT) != WEIGHT_SHA:
            raise RuntimeError("Checkpoint hash mismatch")
        self.config = SimpleNamespace(device=torch.device(c["device"]), split_delivery=False, round_distances=False)
        if self.config.device.type == "cuda" and os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
            raise RuntimeError("Invalid deterministic CUBLAS_WORKSPACE_CONFIG")
        if self.config.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; no CPU fallback")
            if torch.cuda.get_device_capability(self.config.device) == (7, 0) and "sm_70" not in torch.cuda.get_arch_list():
                raise RuntimeError("PyTorch wheel lacks V100 sm_70")
        torch.set_num_threads(c["torch_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        data = torch.load(ROOT / WEIGHT, map_location=self.config.device, weights_only=False)
        if data.get("destroy_operation") != "P" or float(data.get("p_destruction", -1)) != .2:
            raise RuntimeError("Checkpoint P/0.2 metadata mismatch")
        if data.get("ots_mode") != "random-seq":
            raise RuntimeError("Unexpected official Repair start semantics metadata")
        self.actor = VrpActorModel(self.config.device, 128).to(self.config.device)
        self.actor.load_state_dict(data["parameters"], strict=True)
        self.actor.eval()
        self.environment = {"torch": torch.__version__, "numpy": np.__version__, "device": str(self.config.device),
                            "cuda_runtime": torch.version.cuda, "checkpoint_sha256": WEIGHT_SHA,
                            "checkpoint_metadata": {k: v for k, v in data.items() if k != "parameters" and
                                                    isinstance(v, (str, float, int, bool, type(None)))},
                            "parameter_shapes": {k: list(v.shape) for k, v in data["parameters"].items()},
                            "training_distribution_evidence": "official upstream blob/hash and ALTR_C_100 path; no explicit distribution field in known metadata",
                            "deterministic": True, "tf32": False, "threads": c["torch_threads"]}
        self.environment.update(cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"], cuda_driver_cache_disabled=True)
        if self.config.device.type == "cuda":
            self.environment.update(gpu=torch.cuda.get_device_name(self.config.device),
                                    capability=list(torch.cuda.get_device_capability(self.config.device)),
                                    wheel_arch_list=torch.cuda.get_arch_list())

    def sync(self):
        if self.config.device.type == "cuda":
            torch.cuda.synchronize(self.config.device)

    def repair(self, instances, streams):
        with torch.no_grad():
            repair.repair(instances, self.actor, self.config, rngs=streams)
        for v in instances:
            v.verify_solution(self.config)
            v.get_costs(False)

    def peak(self):
        if self.config.device.type != "cuda":
            return {"allocated": None, "reserved": None}
        return {"allocated": torch.cuda.max_memory_allocated(self.config.device),
                "reserved": torch.cuda.max_memory_reserved(self.config.device)}
