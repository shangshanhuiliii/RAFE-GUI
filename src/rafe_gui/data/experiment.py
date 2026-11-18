import json
from pathlib import Path

from ..storage import read, sha, write, write_rows

METHODS = ("original", "gray", "local", "gaussian", "texture")


def load(work, root_override=None):
    work = Path(work).resolve()
    exp = json.loads((work / "experiment.json").read_text(encoding="utf-8"))
    root = (
        Path(root_override).resolve()
        if root_override is not None
        else (work / exp["dataset_root"]).resolve()
    )
    for path, checksum in exp["source_files"].items():
        if sha(root / path) != checksum:
            raise ValueError(f"Frozen dataset changed: {path}")
    if sha(work / "sources.jsonl") != exp["sources_sha256"]:
        raise ValueError("Frozen sources changed")
    tasks = {}
    for scheme in ("A", "B"):
        path = work / f"{scheme}.jsonl"
        if sha(path) != exp["tasks_sha256"][scheme]:
            raise ValueError("Frozen tasks changed")
        tasks[scheme] = read(path)
    images = read(work / "sources.jsonl")
    for image in images:
        if sha(root / image["path"]) != image["sha256"]:
            raise ValueError("Source image changed")
    return work, root, images, tasks, exp


def build_real_real(args):
    """Direct Real account comparison, distinct from Real–Sim account score gaps."""
    methods = tuple(getattr(args, "methods", METHODS))
    work, root, images, tasks, exp = load(args.work)
    pairs = []
    for app, first, second in [
        ("eleme", "eleme_real_native", "eleme_real_native_account2"),
        ("wechat", "wechat_real_account1", "wechat_real_account2"),
    ]:
        a = {i["page_id"]: i for i in images if i["environment_id"] == first}
        b = {i["page_id"]: i for i in images if i["environment_id"] == second}
        for page in sorted(a.keys() & b.keys()):
            group = (
                7
                if app == "eleme" and page in (7, 8)
                else 11
                if app == "eleme" and page in (11, 12)
                else page
            )
            pairs.append(
                dict(
                    schema_version="1.0",
                    task_type="real_real",
                    task_id=f"real_real__{first}__p{page:03d}__{second}__p{page:03d}",
                    app=app,
                    page_id=page,
                    page_name=a[page]["page_name"],
                    status="ready",
                    relation="direct",
                    real=a[page],
                    comparison=b[page],
                    comparison_environment_id=second,
                    aggregation_group_id=f"real_real__{first}__{second}__p{group:03d}",
                    metadata_sha256=exp["source_files"]["metadata.json"],
                    notes="Same page ID across two Real accounts; data, region, collection date and configuration may differ.",
                )
            )
    allowed = {(t["app"], t["page_id"]) for t in tasks["B"]}
    schemes = {
        "A": pairs,
        "B": [p for p in pairs if (p["app"], p["page_id"]) in allowed],
    }
    rows = {r["image_id"]: r for r in read(work / "variants.jsonl")}
    manifest_hashes = {}
    for scheme, selected in schemes.items():
        path = work / "real_real" / f"{scheme}.jsonl"
        write_rows(path, selected)
        manifest_hashes[scheme] = sha(path)
        for method in methods:
            prepared = []
            for task in selected:
                record = dict(
                    task_id=task["task_id"],
                    task_type="real_real",
                    app=task["app"],
                    page_id=task["page_id"],
                    method=method,
                    reference_environment=task["real"]["environment_id"],
                    comparison_environment=task["comparison"]["environment_id"],
                )
                for side in ("real", "comparison"):
                    image = task[side]
                    artifact = rows[image["image_id"]]["artifacts"][method]
                    record[side] = dict(
                        source_path=image["path"],
                        source_sha256=image["sha256"],
                        prepared_path=artifact["path"],
                        prepared_sha256=artifact["sha256"],
                    )
                prepared.append(record)
            write_rows(
                work / "prepared_pairs/real_real" / scheme / (method + ".jsonl"),
                prepared,
            )
    counts = {
        s: {app: sum(t["app"] == app for t in selected) for app in ("eleme", "wechat")}
        for s, selected in schemes.items()
    }
    write(
        work / "real_real/schemes.json",
        dict(
            counts=counts,
            manifests_sha256=manifest_hashes,
            source_metadata_sha256=exp["source_files"]["metadata.json"],
            B_rule="Align page set with Real–Sim B; excluding store/checkout does not imply Real–Real has many-to-one mappings.",
        ),
    )
    print(
        json.dumps(dict(real_real=counts, conditions=list(methods)), ensure_ascii=False)
    )
