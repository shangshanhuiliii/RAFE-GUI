"""Run and analyze RAFE-GUI experiments."""

import argparse
import json
import os
from pathlib import Path

from .config import load_config, load_env, versions
from .storage import read


def check(config, full=True):
    from .data import registry
    from .execution.prepare import validate_prepared

    root = config.path("dataset")
    tasks = read(root / "eval_pairs.jsonl")
    registry.validate(root, tasks)
    output = {
        "project": str(config.root),
        "tasks": len(tasks),
        "ready": sum(t["status"] == "ready" for t in tasks),
        "missing": sum(t["status"] == "missing_comparison" for t in tasks),
    }
    if (config.path("prepared") / "variants.jsonl").exists():
        images, schemes = validate_prepared(config.path("prepared"))
        output.update(
            images=len(images), schemes={s: len(t) for s, t in schemes.items()}
        )
    if config.data.get("experiment_type", "encoders") == "encoders":
        from .metrics.encoders import ENCODERS, verify_model

        lock = json.loads(config.path("models").read_text(encoding="utf-8"))
        for name in config.data["metrics"]:
            if name not in ENCODERS:
                raise ValueError(f"Unknown encoder: {name}")
            verify_model(config.root, lock[name], full=full)
        output["weights_verified"] = full
        output["packages"] = versions(
            ("torch", "torchvision", "transformers", "safetensors", "numpy", "Pillow")
        )
        missing = [k for k, v in output["packages"].items() if v is None]
        if missing:
            raise ValueError(f"Missing encoder dependencies: {missing}")
        import torch

        output["cuda_available"] = torch.cuda.is_available()
        output["gpu_count"] = torch.cuda.device_count()
    else:
        output["api_key_configured"] = bool(os.environ.get("RAFE_API_KEY"))
        output["judge_model"] = os.environ.get("RAFE_JUDGE_MODEL") or None
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = (
        "check",
        "prepare",
        "run",
        "status",
        "analyze",
        "export",
        "_encode",
    )
    parsers = {}
    for command in commands:
        p = sub.add_parser(command)
        parsers[command] = p
        p.add_argument("--project", type=Path)
        p.add_argument("--config", type=Path)
    parsers["run"].add_argument("--run-id", help="Default: a new timestamped run")
    for command in ("status", "analyze"):
        parsers[command].add_argument("--run-id", required=True)
    parsers["run"].add_argument("--resume", action="store_true")
    parsers["run"].add_argument("--dry-run", action="store_true")
    parsers["check"].add_argument(
        "--quick", action="store_true", help="Skip multi-GiB weight hashes"
    )
    p = parsers["export"]
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--with-models", action="store_true")
    p = parsers["_encode"]
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, args.project)
        load_env(config.root)
        if args.command == "check":
            result = check(config, not args.quick)
        elif args.command == "prepare":
            from .execution.prepare import prepare

            result = {"prepared": str(prepare(config))}
        elif args.command == "run":
            from .execution.pipeline import run

            result = run(config, args.run_id, args.resume, args.dry_run)
        elif args.command == "status":
            from .execution.pipeline import run_directory

            directory = run_directory(config, args.run_id)
            path = directory / "status.json"
            result = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.exists()
                else {"state": "not_started"}
            )
            if result["state"] == "running":
                from .execution.state import run_lock

                try:
                    with run_lock(directory):
                        result["state"] = "interrupted"
                        result["reason"] = (
                            "Last state was running, but no process holds the run lock"
                        )
                except ValueError:
                    pass
            result["feature_progress"] = {}
            for path in directory.glob("*/features/*/index.jsonl"):
                rows = read(path)
                result["feature_progress"][path.relative_to(directory).as_posix()] = {
                    "ok": sum(r["status"] == "ok" for r in rows),
                    "errors": sum(r["status"] == "error" for r in rows),
                }
        elif args.command == "analyze":
            from .execution.pipeline import analyze_run

            analyze_run(config, args.run_id)
            result = {"analyzed": args.run_id}
        elif args.command == "export":
            from .packaging import export_project

            result = export_project(config.root, args.output, args.with_models)
        else:
            from .execution.encode import encode

            return (
                0
                if encode(
                    config, args.work, args.model, args.mode, args.device, args.limit
                )
                else 1
            )
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        from .execution.pipeline import redact

        print(f"{type(error).__name__}: {redact(str(error))}")
        return 1
