"""Run experiments and stream worker progress to the current terminal."""

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from ..data.experiment import load
from ..storage import fingerprint, write
from .prepare import prepare
from .state import now, run_lock, status


def run_directory(config, name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Invalid run ID")
    return config.path("outputs") / name


def clone_prepared(config, source, target):
    if (target / "experiment.json").exists():
        load(target)
        return
    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "features",
            "analysis",
            "source_snapshot",
            "data_snapshot",
            "preprocess_preview",
        ),
    )
    experiment = json.loads((source / "experiment.json").read_text(encoding="utf-8"))
    experiment["dataset_root"] = Path(
        os.path.relpath(config.path("dataset"), target)
    ).as_posix()
    write(target / "experiment.json", experiment)
    load(target)


def run(config, name, resume=False, dry_run=False):
    if name is None:
        if resume:
            raise ValueError("--resume requires --run-id for this project")
        name = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%fZ")
    directory = run_directory(config, name)
    kind = config.data.get("experiment_type", "encoders")
    plan = {
        "experiment_type": kind,
        "variants": config.data.get("variants", ["original"]),
        "preprocessing": config.data.get("preprocessing", []),
        "metrics": config.data.get("metrics", []),
        "directory": str(directory),
    }
    if dry_run:
        return plan
    with run_lock(directory):
        recipe = fingerprint({"configuration": config.data, "run_schema": 2})
        record = directory / "run.json"
        if record.exists():
            if not resume:
                raise ValueError("Run exists; use --resume or another run ID")
            if json.loads(record.read_text(encoding="utf-8"))["fingerprint"] != recipe:
                raise ValueError("Experiment configuration changed")
        else:
            write(
                record,
                {
                    "fingerprint": recipe,
                    "configuration": config.data,
                    "started_at_utc": now(),
                    "kind": kind,
                },
            )
        children = []
        try:
            print(f"[RAFE-GUI] Run {name}\n[RAFE-GUI] Output: {directory}", flush=True)
            if kind == "vlm":
                from .judge import run_judge

                status(directory, "running", stage="judge", plan=plan)
                print("[RAFE-GUI] Validating VLM inputs", flush=True)
                run_judge(config, directory, resume=resume)
                print(
                    f"[RAFE-GUI] Report: {directory / 'analysis/report.md'}", flush=True
                )
            else:
                status(directory, "running", stage="prepare", plan=plan)
                print("[RAFE-GUI] Checking dataset and prepared images", flush=True)
                prepared = prepare(config)
                modes = config.data["preprocessing"]
                for number, mode in enumerate(modes, 1):
                    print(
                        f"[RAFE-GUI] {number}/{len(modes)} {mode}: preparing inputs",
                        flush=True,
                    )
                    work = directory / mode
                    clone_prepared(config, prepared, work)
                    status(
                        directory,
                        "running",
                        stage="encode",
                        mode=mode,
                        models=config.data["metrics"],
                    )
                    children = []
                    for model in config.data["metrics"]:
                        device = config.data["devices"].get(model, "cpu")
                        print(
                            f"[RAFE-GUI] {mode}: starting {model} on {device}",
                            flush=True,
                        )
                        env = os.environ.copy()
                        env["PYTHONIOENCODING"] = "utf-8"
                        env["PYTHONUNBUFFERED"] = "1"
                        if device.startswith("cuda:"):
                            env["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]
                            device = "cuda"
                        command = [
                            sys.executable,
                            "-m",
                            "rafe_gui",
                            "_encode",
                            "--config",
                            str(config.file),
                            "--project",
                            str(config.root),
                            "--work",
                            str(work),
                            "--model",
                            model,
                            "--mode",
                            mode,
                            "--device",
                            device,
                        ]
                        children.append(
                            subprocess.Popen(
                                command,
                                env=env,
                                stderr=subprocess.STDOUT,
                            )
                        )
                    codes = [p.wait() for p in children]
                    if any(codes):
                        raise RuntimeError(
                            f"Encoder workers failed in {mode} (exit codes {codes}). See the terminal output above."
                        )
                    status(directory, "running", stage="analyze", mode=mode)
                    print(f"[RAFE-GUI] {mode}: summarizing scores", flush=True)
                    analyze_work(config, work)
                if {"square-pad", "stretch"} <= set(modes):
                    from ..analysis.compare import compare

                    status(directory, "running", stage="compare")
                    print("[RAFE-GUI] Comparing the two input modes", flush=True)
                    compare(
                        SimpleNamespace(
                            pad=directory / "square-pad",
                            stretch=directory / "stretch",
                            output=directory / "comparison",
                        )
                    )
                    print(
                        f"[RAFE-GUI] Comparison report: {directory / 'comparison/report.md'}",
                        flush=True,
                    )
            status(directory, "completed", stage="finished", plan=plan)
            print(f"[RAFE-GUI] Completed {name}", flush=True)
        except BaseException as error:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                child.wait()
            status(
                directory,
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=redact(str(error)),
            )
            raise
    return directory


def redact(text):
    key = os.environ.get("RAFE_API_KEY")
    return text.replace(key, "[REDACTED]") if key else text


def analyze_work(config, work):
    from ..analysis.summary import analyze

    analyze(
        SimpleNamespace(
            work=work, models=config.data["metrics"], methods=config.data["variants"]
        )
    )


def analyze_run(config, name):
    directory = run_directory(config, name)
    if config.data.get("experiment_type") == "vlm":
        from ..analysis.judge import summarize_judge

        return summarize_judge(directory)
    for mode in config.data["preprocessing"]:
        analyze_work(config, directory / mode)
    if {"square-pad", "stretch"} <= set(config.data["preprocessing"]):
        from ..analysis.compare import compare

        compare(
            SimpleNamespace(
                pad=directory / "square-pad",
                stretch=directory / "stretch",
                output=directory / "comparison",
            )
        )
