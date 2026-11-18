"""Read dataset metadata and build page pair manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from ..storage import sha as sha256

SCHEMA_VERSION = "1.0"
ENV_CODES = {
    "eleme_app_1": "native",
    "eleme_app_2": "native_account2",
    "eleme_miniapp": "miniapp",
    "wechat_1": "account1",
    "wechat_2": "account2",
    "AppSim": "appsim",
    "MobileGym++": "mobilegympp",
    "PhoneWorld": "phoneworld",
    "ScaleWob": "scalewob",
}


def fingerprint(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def image_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Expected a dataset-relative POSIX path: {relative}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"Image is missing or outside the dataset: {relative}")
    return resolved


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{number}: invalid JSON; repair the line before resuming"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(row)
    return rows


def environment_id(record: dict) -> str:
    leaf = PurePosixPath(record["当前文件路径"]).parent.name
    return record["APP类别"] + "_" + record["Real/Sim"].lower() + "_" + ENV_CODES[leaf]


def image_record(record: dict, root: Path) -> dict:
    relative = record["当前文件路径"]
    return {
        "environment_id": environment_id(record),
        "environment_name": record["环境名称"],
        "path": relative,
        "page_id": record["页面ID"],
        "page_name": record["页面名称"],
        "page_state": record["页面状态"],
        "page_type": record["页面类型"],
        "sha256": sha256(image_path(root, relative)),
    }


def build(root: Path, destination: Path, include_real_real: bool = False) -> list[dict]:
    metadata_path = root / "metadata.json"
    if (
        destination.suffix.lower() != ".jsonl"
        or destination.resolve() == metadata_path.resolve()
    ):
        raise ValueError("Manifest destination must be a separate .jsonl file")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = metadata["截图"]
    paths = [row["当前文件路径"] for row in records]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate image paths in metadata.json")
    images = {r["当前文件路径"]: image_record(r, root) for r in records}
    reals = sorted(
        (r for r in records if r["Real/Sim"] == "Real"),
        key=lambda r: (environment_id(r), r["页面ID"]),
    )
    sims = [r for r in records if r["Real/Sim"] == "Sim"]
    sim_environments = sorted({(r["APP类别"], environment_id(r)) for r in sims})
    index = {}
    real_keys = {(r["APP类别"], r["页面ID"]) for r in reals}
    for sim in sims:
        for page_id in sim["对应Real页面ID"]:
            if (sim["APP类别"], page_id) not in real_keys:
                raise ValueError(
                    f"No real reference for {sim['当前文件路径']}: {page_id}"
                )
            key = (sim["APP类别"], environment_id(sim), page_id)
            if key in index:
                raise ValueError(
                    f"Ambiguous comparison: {key}; define the preferred image in metadata"
                )
            index[key] = sim
    metadata_hash = sha256(metadata_path)
    tasks = []

    def task(
        reference: dict,
        comparison: dict | None,
        comparator_environment: str,
        task_type: str,
    ):
        app, page_id = reference["APP类别"], reference["页面ID"]
        real = images[reference["当前文件路径"]]
        comparator = images[comparison["当前文件路径"]] if comparison else None
        notes = {
            "real": reference["特殊情况说明"],
            "comparison": comparison["特殊情况说明"] if comparison else None,
        }
        flags = []
        if any(
            "待核实" in (note or "") or "待复核" in (note or "")
            for note in notes.values()
        ):
            flags.append("manual_review_pending")
        shared = bool(
            comparison
            and task_type == "real_sim"
            and len(comparison["对应Real页面ID"]) > 1
        )
        if shared:
            flags.append("sim_image_reused_for_multiple_views")
        partial = bool(
            comparison
            and task_type == "real_sim"
            and page_id != comparison["页面ID"]
            and "部分范围对应" in (comparison["特殊情况说明"] or "")
        )
        if partial:
            flags.append("partial_view_coverage")
        partial_confirmed = bool(
            partial and page_id in comparison.get("已确认部分对应Real页面ID", [])
        )
        if partial_confirmed:
            flags.append("partial_view_coverage_confirmed")
        if task_type == "real_real":
            flags.append("real_account_variation")
        status = (
            "missing_comparison"
            if comparator is None
            else "review_required"
            if "manual_review_pending" in flags or (partial and not partial_confirmed)
            else "ready"
        )
        # Group the two store views / checkout views inside each fixed real anchor and simulator.
        group_page_id = (
            7
            if app == "eleme" and page_id in (7, 8)
            else 11
            if app == "eleme" and page_id in (11, 12)
            else page_id
        )
        comparison_page = f"p{comparator['page_id']:03d}" if comparator else "missing"
        tasks.append(
            {
                "schema_version": SCHEMA_VERSION,
                "task_id": f"{task_type}__{real['environment_id']}__p{page_id:03d}__{comparator_environment}__{comparison_page}",
                "task_type": task_type,
                "app": app,
                "page_id": page_id,
                "page_name": reference["页面名称"],
                "status": status,
                "relation": "missing"
                if comparator is None
                else "partial_view"
                if partial
                else "shared_sim_view"
                if shared
                else "direct",
                "real": real,
                "comparison_environment_id": comparator_environment,
                "comparison": comparator,
                "review_flags": flags,
                "notes": notes,
                "aggregation_group_id": f"{task_type}__{real['environment_id']}__{comparator_environment}__p{group_page_id:03d}",
                "comparison_image_reuse_key": comparator["path"]
                if comparator
                else None,
                "metadata_sha256": metadata_hash,
            }
        )

    for real in reals:
        for app, env in sim_environments:
            if app == real["APP类别"]:
                task(real, index.get((app, env, real["页面ID"])), env, "real_sim")
    if include_real_real:
        accounts = {
            r["页面ID"]: r for r in reals if environment_id(r) == "wechat_real_account2"
        }
        for real in reals:
            if (
                environment_id(real) == "wechat_real_account1"
                and real["页面ID"] in accounts
            ):
                other = accounts[real["页面ID"]]
                task(real, other, environment_id(other), "real_real")
    if len({r["task_id"] for r in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in tasks:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temporary.replace(destination)
    return tasks


def validate(root: Path, tasks: list[dict]) -> None:
    current_metadata_hash = sha256(root / "metadata.json")
    hashes = {}
    ids = set()
    for task in tasks:
        if task["schema_version"] != SCHEMA_VERSION or task["task_id"] in ids:
            raise ValueError("Unsupported schema version or duplicate task ID")
        ids.add(task["task_id"])
        if task["metadata_sha256"] != current_metadata_hash:
            raise ValueError(
                "metadata.json changed; rebuild eval_pairs.jsonl before running"
            )
        if (task["comparison"] is None) != (task["status"] == "missing_comparison"):
            raise ValueError(f"Invalid missing-image state: {task['task_id']}")
        for image in (task["real"], task["comparison"]):
            if image is None:
                continue
            relative = image["path"]
            if relative not in hashes:
                hashes[relative] = sha256(image_path(root, relative))
            if hashes[relative] != image["sha256"]:
                raise ValueError(
                    f"Image content changed: {relative}; rebuild the manifest"
                )
