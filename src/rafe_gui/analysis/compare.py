"""Compare square-padding and direct-resize runs on the same 20-score success set."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..data.experiment import METHODS, load
from ..storage import read, sha, write, write_rows
from .summary import stats

MODELS = ("siglip", "dino")


def aggregate(rows, grouped=False, models=MODELS, methods=METHODS):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"] if grouped else row["task_id"]].append(row)
    result = {}
    for model in models:
        result[model] = {}
        for method in methods:
            values = {
                mode: [
                    statistics.mean(r["scores"][mode][model][method] for r in members)
                    for members in groups.values()
                ]
                for mode in ("square-pad", "stretch")
            }
            result[model][method] = {mode: stats(v) for mode, v in values.items()}
            result[model][method]["stretch_minus_pad"] = stats(
                [b - a for a, b in zip(values["square-pad"], values["stretch"])]
            )
    return result


def paired_rows(tasks, pad_rows, stretch_rows, models=MODELS, methods=METHODS):
    indexed = []
    for rows in (pad_rows, stretch_rows):
        by_id = {r["task_id"]: r for r in rows}
        if len(by_id) != len(rows) or set(by_id) != {t["task_id"] for t in tasks}:
            raise ValueError("Analysis task IDs do not match frozen tasks")
        indexed.append(by_id)
    output = []
    for task in tasks:
        row = dict(
            task_id=task["task_id"],
            app=task["app"],
            page_id=task["page_id"],
            group=task["aggregation_group_id"],
            real_environment=task["real"]["environment_id"],
            comparison_environment=task["comparison_environment_id"],
        )
        row["scores"] = {
            mode: indexed[n][task["task_id"]]["scores"]
            for n, mode in enumerate(("square-pad", "stretch"))
        }
        values = [
            row["scores"][mode][model][method]
            for mode in ("square-pad", "stretch")
            for model in models
            for method in methods
        ]
        row["complete"] = all(v is not None and np.isfinite(v) for v in values)
        row["deltas"] = {
            model: {
                m: (
                    row["scores"]["stretch"][model][m]
                    - row["scores"]["square-pad"][model][m]
                )
                if row["complete"]
                else None
                for m in methods
            }
            for model in models
        }
        output.append(row)
    return output


def summarize(rows, real_sim, models=MODELS, methods=METHODS):
    valid = [r for r in rows if r["complete"]]
    report = dict(
        expected=len(rows),
        complete=len(valid),
        excluded_incomplete=[r["task_id"] for r in rows if not r["complete"]],
        scores=aggregate(valid, models=models, methods=methods),
        by_app={},
        common_environments=[],
    )
    for app in sorted({r["app"] for r in rows}):
        members = [r for r in valid if r["app"] == app]
        report["by_app"][app] = dict(
            expected=sum(r["app"] == app for r in rows),
            complete=len(members),
            scores=aggregate(members, models=models, methods=methods),
            group_balanced=aggregate(
                members, grouped=True, models=models, methods=methods
            ),
        )
        if not real_sim:
            continue
        envs = sorted({r["comparison_environment"] for r in rows if r["app"] == app})
        if not envs:
            continue
        common = set.intersection(
            *[
                {
                    (r["real_environment"], r["page_id"])
                    for r in members
                    if r["comparison_environment"] == e
                }
                for e in envs
            ]
        )
        for env in envs:
            selected = [
                r
                for r in members
                if r["comparison_environment"] == env
                and (r["real_environment"], r["page_id"]) in common
            ]
            report["common_environments"].append(
                dict(
                    app=app,
                    environment=env,
                    pairs=len(selected),
                    groups=len({r["group"] for r in selected}),
                    scores=aggregate(
                        selected, grouped=True, models=models, methods=methods
                    ),
                )
            )
    return report


def compare(args):
    pad, root, images, tasks, exp = load(args.pad)
    stretch, _, _, other_tasks, other_exp = load(args.stretch)
    if pad == stretch:
        raise ValueError("Specify two separate work directories")
    for name in (
        "sources.jsonl",
        "A.jsonl",
        "B.jsonl",
        "variants.jsonl",
        "render.run.json",
    ):
        if sha(pad / name) != sha(stretch / name):
            raise ValueError(f"Input or text treatment differs: {name}")
    if exp["source_files"] != other_exp["source_files"]:
        raise ValueError("Dataset snapshots differ")
    summaries = {
        mode: json.loads((work / "analysis/summary.json").read_text(encoding="utf-8"))
        for mode, work in (("square-pad", pad), ("stretch", stretch))
    }
    models = tuple(summaries["square-pad"]["encoder_audit"])
    methods = tuple(
        summaries["square-pad"]["encoder_audit"][models[0]]["run"]["configuration"].get(
            "methods", METHODS
        )
    )
    for model in models:
        configurations = []
        for mode, work in (("square-pad", pad), ("stretch", stretch)):
            run = json.loads(
                (work / "features" / model / "run.json").read_text(encoding="utf-8")
            )
            if summaries[mode]["encoder_audit"][model]["run"] != run:
                raise ValueError("Analysis is stale; rerun analyze")
            cfg = dict(run["configuration"])
            spec = cfg.pop("encoder_preprocessing")
            if spec["mode"] != mode or spec["output_size_wh"] != [560, 560]:
                raise ValueError("Unexpected geometry protocol")
            configurations.append(cfg)
        if configurations[0] != configurations[1]:
            raise ValueError(f"Encoder protocols differ beyond geometry: {model}")
    result = {}
    destination = Path(args.output).resolve()
    if (
        destination in (pad, stretch)
        or destination.is_relative_to(pad)
        or destination.is_relative_to(stretch)
    ):
        raise ValueError(
            "Comparison output must be separate from both input work directories"
        )
    for kind in ("real_sim", "real_real"):
        result[kind] = {}
        for scheme in ("A", "B"):
            relative = (
                Path("analysis")
                / (scheme if kind == "real_sim" else "real_real/" + scheme)
                / "paired_scores.jsonl"
            )
            if kind == "real_sim":
                selected = tasks[scheme]
            else:
                if sha(pad / "real_real" / f"{scheme}.jsonl") != sha(
                    stretch / "real_real" / f"{scheme}.jsonl"
                ):
                    raise ValueError("Real–Real mappings differ")
                selected = read(pad / "real_real" / f"{scheme}.jsonl")
            rows = paired_rows(
                selected,
                read(pad / relative),
                read(stretch / relative),
                models=models,
                methods=methods,
            )
            result[kind][scheme] = summarize(
                rows, real_sim=kind == "real_sim", models=models, methods=methods
            )
            write_rows(destination / kind / scheme / "paired_scores.jsonl", rows)
    write(
        destination / "summary.json",
        dict(
            conditions=list(methods),
            success_rule="Intersection of 2 geometries x 2 models x 5 conditions",
            inputs={"square-pad": str(pad), "stretch": str(stretch)},
            results=result,
        ),
    )
    lines = [
        "# 补白与直接缩放对比",
        "",
        "square-pad：整页等比缩至252×560，白色补齐560×560；stretch：整页直接缩放至560×560，宽高比改变。",
        "每张表使用两种几何、两个模型、五种图像条件的20套分数共同成功任务。Δ=stretch−square-pad；分数升高不等于评价更准确。",
        "",
    ]

    def table(title, entry):
        lines.extend(
            [
                f"## {title}",
                "",
                "| 方法 | 模型 | 补白均值 | 直接缩放均值 | Δ | 配对/组数 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )

        def fmt(v):
            return f"{v:.4f}" if v is not None else "缺失"

        for method in methods:
            for model in models:
                values = entry[model][method]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            method,
                            model,
                            fmt(values["square-pad"]["mean"]),
                            fmt(values["stretch"]["mean"]),
                            fmt(values["stretch_minus_pad"]["mean"]),
                            str(values["square-pad"]["n"]),
                        ]
                    )
                    + " |"
                )
        lines.append("")

    for kind, schemes in result.items():
        for scheme, report in schemes.items():
            lines.extend(
                [
                    f"{kind} / {scheme}：共同成功 {report['complete']}/{report['expected']}。",
                    "",
                ]
            )
            for app, entry in report["by_app"].items():
                if entry["expected"]:
                    table(f"{kind} / {scheme} / {app}（逐对等权）", entry["scores"])
            for entry in report["common_environments"]:
                table(
                    f"{scheme} / {entry['environment']}（共同页面、页面组等权）",
                    entry["scores"],
                )
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(destination),
                "complete": {
                    k: {s: r["complete"] for s, r in v.items()}
                    for k, v in result.items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pad", type=Path, required=True)
    parser.add_argument("--stretch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    compare(parser.parse_args())
