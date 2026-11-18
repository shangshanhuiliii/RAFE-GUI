import json
from collections import defaultdict
from pathlib import Path

from ..metrics.vlm import prompt
from ..storage import sha as digest

SCHEMA = "rafe_logical_page_group_v1"


def build_groups(tasks: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for task in tasks:
        key = (
            task["real"]["environment_id"],
            task["comparison_environment_id"],
            task["aggregation_group_id"],
        )
        buckets[key].append(task)
    groups = []
    for (real_env, sim_env, group_id), members in sorted(buckets.items()):

        def views(side):
            unique = {t[side]["path"]: t[side] for t in members if t[side] is not None}
            return sorted(unique.values(), key=lambda v: (v["page_id"], v["path"]))

        real, sim = views("real"), views("comparison")
        # Review is determined after grouping: never quietly drop a pending sibling.
        if not sim:
            status = "missing_comparison"
        elif any(t["status"] != "ready" for t in members):
            status = "review_required"
        else:
            status = "ready"
        groups.append(
            {
                "schema_version": SCHEMA,
                "group_id": "group__" + group_id,
                "app": members[0]["app"],
                "real_environment_id": real_env,
                "comparison_environment_id": sim_env,
                "page_ids": sorted({t["page_id"] for t in members}),
                "page_names": sorted({t["page_name"] for t in members}),
                "real_views": real,
                "sim_views": sim,
                "status": status,
                "source_task_ids": sorted(t["task_id"] for t in members),
                "metadata_sha256": members[0]["metadata_sha256"],
                "review_flags": sorted(
                    {f for t in members for f in t.get("review_flags", [])}
                ),
            }
        )
    if len({g["group_id"] for g in groups}) != len(groups):
        raise ValueError("Group IDs are reused across different real/sim environments")
    return groups


def validate_groups(root: Path, groups: list[dict], batch) -> None:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    registry = {
        r["当前文件路径"]: batch.image_record(r, root) for r in metadata["截图"]
    }
    seen = set()
    for group in groups:
        if group["schema_version"] != SCHEMA or group["group_id"] in seen:
            raise ValueError("Unsupported group schema or duplicate group ID")
        seen.add(group["group_id"])
        if (
            not isinstance(group.get("page_ids"), list)
            or not group["page_ids"]
            or any(type(page) is not int or page < 1 for page in group["page_ids"])
            or not isinstance(group.get("page_names"), list)
            or not group["page_names"]
            or any(
                not isinstance(name, str) or not name.strip()
                for name in group["page_names"]
            )
            or not isinstance(group.get("source_task_ids"), list)
        ):
            raise ValueError(
                "Group page IDs, page names and source task IDs are required"
            )
        if group["metadata_sha256"] != digest(root / "metadata.json"):
            raise ValueError("Group metadata is stale; rebuild the group manifest")
        if group["status"] not in ("ready", "review_required", "missing_comparison"):
            raise ValueError("Invalid group status")
        if not group["real_views"] or (
            (not group["sim_views"]) != (group["status"] == "missing_comparison")
        ):
            raise ValueError("Invalid empty-side group state")
        for side, env_key, category in (
            ("real_views", "real_environment_id", "Real"),
            ("sim_views", "comparison_environment_id", "Sim"),
        ):
            paths = set()
            for view in group[side]:
                expected = registry.get(view["path"])
                if expected is None or any(
                    view.get(k) != expected[k]
                    for k in ("environment_id", "sha256", "page_id")
                ):
                    raise ValueError(f"Invalid or stale group image: {view['path']}")
                if view["environment_id"] != group[env_key] or view["path"] in paths:
                    raise ValueError("A group mixes environments or repeats an image")
                # Canonical registry IDs encode category and real/sim role.
                if not view["environment_id"].startswith(
                    group["app"] + "_" + category.lower() + "_"
                ):
                    raise ValueError("A group mixes apps or real/sim roles")
                paths.add(view["path"])


def labels(group):
    return {
        label: view
        for prefix, side in (("R", "real_views"), ("S", "sim_views"))
        for label, view in ((prefix + str(i), v) for i, v in enumerate(group[side], 1))
    }


def messages(group: dict, root: Path, batch) -> list[dict]:
    from ..metrics.vlm import step_image_url

    content = [
        {"type": "text", "text": "目标逻辑页面：" + " / ".join(group["page_names"])}
    ]
    for label, view in labels(group).items():
        # Do not send simulator names, file names, human difference notes or review flags.
        content.append(
            {
                "type": "text",
                "text": f"{label} | page_id={view['page_id']} | state={view['page_state']}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": step_image_url(batch.image_path(root, view["path"]))
                },
            }
        )
    content.append({"type": "text", "text": prompt("rafe_group_user.md")})
    return [
        {"role": "system", "content": prompt("rafe_group_system.md")},
        {"role": "user", "content": content},
    ]


def validate_response(parsed: dict, group: dict) -> dict:
    from ..metrics.vlm import _number, _required_text

    _required_text(parsed, "reasoning")
    comparable = parsed.get("comparability")
    if comparable not in ("comparable", "partial", "not_comparable", "uncertain"):
        raise ValueError("Invalid comparability verdict")
    refs = labels(group)

    def evidence(items, prefix=None):
        if not isinstance(items, list) or any(
            not isinstance(x, str)
            or x not in refs
            or (prefix and not x.startswith(prefix))
            for x in items
        ):
            raise ValueError("Judge cited an unknown or wrong-side image label")

    regions = parsed.get("regions")
    if not isinstance(regions, list) or not 1 <= len(regions) <= 8:
        raise ValueError("regions must contain 1 to 8 evidence records")
    for region in regions:
        _required_text(region, "region")
        _required_text(region, "observation")
        evidence(region.get("real_views"), "R")
        evidence(region.get("sim_views"), "S")
        if not region["real_views"] and not region["sim_views"]:
            raise ValueError("A region needs at least one visible image reference")
        if region.get("assessment") not in (
            "matched",
            "different",
            "not_observed",
            "uncertain",
        ):
            raise ValueError("Invalid region assessment")
    for key in ("data_variations", "state_differences"):
        if not isinstance(parsed.get(key), list):
            raise ValueError(f"{key} must be an array")
        for item in parsed[key]:
            _required_text(item, "observation")
            evidence(item.get("views"))
            if not item["views"]:
                raise ValueError("A variation needs image evidence")
    if not isinstance(parsed.get("uncertainties"), list) or any(
        not isinstance(x, str) or not x.strip() for x in parsed["uncertainties"]
    ):
        raise ValueError("uncertainties must be an array of nonempty strings")
    scores = {}
    for field, metric in (
        ("element_alignment_score", "S_ele_group"),
        ("structural_fidelity_score", "S_lay_group"),
    ):
        if comparable == "comparable":
            scores[metric] = _number(parsed, field, 1, 10)
        else:
            if field not in parsed or parsed[field] is not None:
                raise ValueError("Non-comparable groups must return null scores")
            scores[metric] = None
    return {
        "metrics": {
            k: (v - 1) / 9 if v is not None else None for k, v in scores.items()
        },
        "raw_scores": scores,
        "judge": parsed,
    }
