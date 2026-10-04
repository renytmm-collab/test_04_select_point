import argparse
import ast
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from .identity import ROOT, WEIGHT, WEIGHT_SHA, file_sha, canonical
from .config import load, budget


def main():
    parser = argparse.ArgumentParser(description="Staged fixed-Repair Point selection. No all-in-one experiment command.")
    parser.add_argument("command", choices=["check", "budget", "audit", "benchmark", "states", "candidates", "features",
         "collect-dev-a", "fit", "freeze-test-predictions", "collect-test-a", "freeze-test-reference", "collect-test-b", "analyze"])
    parser.add_argument("--config", default=str(ROOT / "configs/pilot.template.json"))
    parser.add_argument("--run")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--cap", type=int)
    parser.add_argument("--max-fits", type=int)
    parser.add_argument("--policy")
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steady", type=int, default=5)
    parser.add_argument("--snapshot")
    parser.add_argument("--benchmark-run", action="append", default=[])
    args = parser.parse_args()
    try:
        c = load(args.config)
        cmd = args.command
        if cmd == "budget":
            result = budget(c)
        elif cmd == "check":
            versions = {}
            for pkg in ("numpy", "torch"):
                try:
                    versions[pkg] = importlib.metadata.version(pkg)
                except importlib.metadata.PackageNotFoundError:
                    versions[pkg] = None
            sources = [*ROOT.glob("*.py"), *(ROOT / "point_select").glob("*.py"), *(ROOT / "vrp").glob("*.py"),
                       *(ROOT / "vrp/dataset_blueprints").glob("*.py"), *(ROOT / "tests").glob("test_*.py")]
            for p in sources:
                ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
            result = {"read_only": True, "python": sys.version.split()[0], "dependencies": versions,
                      "numpy_pin_ok": versions["numpy"] == "1.26.4",
                      "checkpoint_hash_ok": file_sha(ROOT / WEIGHT) == WEIGHT_SHA, "parsed_python_files": len(sources),
                      "budget": budget(c), "caps": c["caps"],
                      "not_checked": ["CUDA availability/architecture", "checkpoint deserialization", "real Repair", "server timing"],
                      "note": "No dependency imports, checkpoint load, installations or output files."}
        else:
            if not args.run:
                raise ValueError("--run required")
            if cmd in ("analyze", "audit"):
                from .analysis import analyze, audit_run
                result = analyze(args.run, c, args.output, args.benchmark_run) if cmd == "analyze" else audit_run(args.run, c)
            else:
                if not args.start:
                    raise ValueError("Explicit --start required for every writing stage")
                import numpy as np
                if np.__version__ != "1.26.4":
                    raise ValueError("Real CLI writing stages require NumPy==1.26.4; no auto-install or fallback")
                from . import pipeline as p
                if cmd == "states":
                    result = p.states(args.run, c, args.cap)
                elif cmd == "benchmark":
                    result = p.benchmark(args.run, c, args.cap, args.batch_size, args.warmup, args.steady, args.snapshot)
                elif cmd.startswith("collect-"):
                    result = p.collect(args.run, c, cmd, args.cap)
                elif cmd == "fit":
                    result = p.fit(args.run, c, args.max_fits)
                elif cmd == "freeze-test-predictions":
                    if not args.policy:
                        raise ValueError("--policy required before freezing test predictions")
                    result = p.freeze_predictions(args.run, c, args.policy, args.benchmark_run)
                elif cmd == "freeze-test-reference":
                    result = p.freeze_reference(args.run, c)
                else:
                    result = getattr(p, cmd)(args.run, c)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
