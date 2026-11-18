"""Strict pairwise and logical-page judges share transport, not scoring protocols."""

import json
import os

from ..data import registry
from ..data.groups import (
    build_groups,
    labels,
    messages,
    validate_groups,
    validate_response,
)
from ..metrics.vlm import VLMJudge
from ..storage import fingerprint, read, write, write_rows
from .pipeline import redact


def run_judge(config, directory, resume=False, client=None):
    root = config.path("dataset")
    protocol = config.data["vlm_protocol"]
    tasks = read(root / config.data.get("manifest", "pairing_schemes/A_full.jsonl"))
    registry.validate(root, tasks)
    if protocol not in ("strict", "grouped"):
        raise ValueError("Unknown VLM protocol")
    if protocol == "grouped":
        manual = config.data.get("group_manifest")
        tasks = (
            read((config.root / manual).resolve()) if manual else build_groups(tasks)
        )
        validate_groups(root, tasks, registry)
    write_rows(directory / "input_manifest.jsonl", tasks)
    tasks = [t for t in tasks if t["status"] == "ready"]
    if config.data.get("limit"):
        tasks = tasks[: config.data["limit"]]
    model = os.environ.get("RAFE_JUDGE_MODEL", "").strip()
    if not model:
        raise ValueError("Set RAFE_JUDGE_MODEL explicitly in .env")
    if client is None:
        from openai import OpenAI

        if not os.environ.get("RAFE_API_KEY"):
            raise ValueError("Set RAFE_API_KEY in .env")
        client = OpenAI(
            api_key=os.environ["RAFE_API_KEY"],
            base_url=os.environ.get("RAFE_API_BASE_URL") or "https://api.openai.com/v1",
            max_retries=0,
        )
    extra = json.loads(os.environ.get("RAFE_JUDGE_EXTRA_BODY_JSON") or "{}")
    if not isinstance(extra, dict):
        raise ValueError("Extra body must be a JSON object")
    judge = VLMJudge(client, model, max_tokens=4096, extra_body=extra)
    prompt_names = (
        ["s_ele_lay_system.md", "s_ele_lay_user.md"]
        if protocol == "strict"
        else ["rafe_group_system.md", "rafe_group_user.md"]
    )
    from ..metrics.vlm import prompt

    recipe = {
        "protocol": protocol,
        "model": model,
        "base_url": os.environ.get("RAFE_API_BASE_URL") or "https://api.openai.com/v1",
        "extra_body": extra,
        "max_tokens": 4096,
        "temperature": 0.0,
        "image_encoding": {"longest": 1280, "image_format": "JPEG", "jpeg_quality": 88},
        "prompts": {name: fingerprint(prompt(name)) for name in prompt_names},
        "tasks": fingerprint(tasks),
    }
    key = fingerprint(recipe)
    runfile = directory / "judge.run.json"
    if (
        runfile.exists()
        and json.loads(runfile.read_text(encoding="utf-8"))["fingerprint"] != key
    ):
        raise ValueError("Judge protocol changed")
    write(runfile, {"fingerprint": key, "configuration": recipe})
    output = directory / "scores.jsonl"
    rows = {r["task_id"]: r for r in read(output)} if output.exists() else {}
    if not output.exists():
        write_rows(output, [])
    print(f"[VLM | {protocol}] Judging {len(tasks)} tasks", flush=True)
    for n, task in enumerate(tasks, 1):
        identifier = task["task_id"] if protocol == "strict" else task["group_id"]
        if rows.get(identifier, {}).get("status") in ("ok", "needs_review"):
            continue
        try:
            if protocol == "strict":
                result = judge.s_ele_lay(
                    registry.image_path(root, task["real"]["path"]),
                    registry.image_path(root, task["comparison"]["path"]),
                )
                result = {
                    "metrics": {m: result[m] for m in ("S_ele", "S_lay")},
                    "raw_scores": result["raw_scores"],
                    "judge": result["judge"],
                }
            else:
                if len(task["real_views"]) + len(task["sim_views"]) > config.data.get(
                    "max_group_images", 16
                ):
                    raise ValueError("Group exceeds image limit")
                parsed = judge._request(
                    "S_ele_group/S_lay_group", messages(task, root, registry)
                )
                result = validate_response(parsed, task)
            needs_review = protocol == "grouped" and any(
                v is None for v in result["metrics"].values()
            )
            row = {
                "task_id": identifier,
                "protocol": key,
                "status": "needs_review" if needs_review else "ok",
                "result": result,
            }
            if protocol == "grouped":
                row["input_views"] = labels(task)
        except Exception as error:
            row = {
                "task_id": identifier,
                "protocol": key,
                "status": "error",
                "error": redact(str(error)),
            }
        rows[identifier] = row
        write_rows(output, rows.values())
        print(f"VLM {protocol} {n}/{len(tasks)} {row['status']}", flush=True)
    if any(
        rows.get(t.get("task_id", t.get("group_id")), {}).get("status")
        not in ("ok", "needs_review")
        for t in tasks
    ):
        raise RuntimeError("Some VLM tasks failed; successful results retained")
    from ..analysis.judge import summarize_judge

    summarize_judge(directory)
