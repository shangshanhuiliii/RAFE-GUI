"""Common-set comparisons; scores are not a quality ranking of transforms."""

import json
import statistics
from collections import Counter, defaultdict

import numpy as np

from ..data.experiment import METHODS, load
from ..storage import inside, read, sha, write, write_rows


def stats(values):
    if not values:
        return dict(n=0, mean=None, median=None)
    return dict(
        n=len(values),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        q10=float(np.quantile(values, 0.1)),
        q90=float(np.quantile(values, 0.9)),
    )


def analyze(args):
    models = tuple(getattr(args, "models", ("siglip", "dino")))
    methods = tuple(getattr(args, "methods", METHODS))
    work, root, images, tasks, exp = load(args.work)
    variants = {r["image_id"]: r for r in read(work / "variants.jsonl")}
    by_path = {i["path"]: i for i in images}
    features = {}
    audit = {}
    for model in models:
        folder = work / "features" / model
        run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        if run["configuration"].get("render_run_sha256"):
            if run["configuration"]["render_run_sha256"] != sha(
                work / "render.run.json"
            ):
                raise ValueError("Render protocol changed after encoding")
        elif run["configuration"]["variants_sha256"] != sha(work / "variants.jsonl"):
            raise ValueError("Render index changed after encoding")
        parity = json.loads((folder / "parity.json").read_text(encoding="utf-8"))
        if parity["absolute_difference"] > parity["tolerance"]:
            raise ValueError("Pairwise parity check failed")
        index = read(folder / "index.jsonl")
        features[model] = {}
        for row in index:
            if row["status"] != "ok":
                continue
            if (
                row["protocol"] != run["fingerprint"]
                or sha(inside(work, row["feature_path"])) != row["feature_sha256"]
            ):
                raise ValueError("Stale feature cache")
            features[model][row["key"]] = np.load(inside(work, row["feature_path"]))
        audit[model] = dict(
            run=run,
            parity=parity,
            successful=len(features[model]),
            expected=len(images) * len(methods),
        )
    geometries = [
        r["run"]["configuration"].get("encoder_preprocessing") for r in audit.values()
    ]
    if any(g != geometries[0] for g in geometries):
        raise ValueError(
            "SigLIP and DINOv2 must use the same preprocessing within one work directory"
        )

    def score(model, method, a, b, raw=False):
        keys = [by_path[p]["image_id"] + "__" + method for p in (a, b)]
        if not all(k in features[model] for k in keys):
            return None
        value = float(features[model][keys[0]] @ features[model][keys[1]])
        return value if raw else float(np.clip(value, 0, 1))

    scheme_reports = {}
    for scheme, selected in tasks.items():
        rows = []
        for t in selected:
            row = dict(
                task_id=t["task_id"],
                app=t["app"],
                page_id=t["page_id"],
                real_environment=t["real"]["environment_id"],
                sim_environment=t["comparison_environment_id"],
                group=t["aggregation_group_id"],
                relation=t["relation"],
                real_path=t["real"]["path"],
                sim_path=t["comparison"]["path"],
            )
            row["scores"] = {
                model: {
                    m: score(model, m, row["real_path"], row["sim_path"])
                    for m in methods
                }
                for model in features
            }
            row["raw_cosines"] = {
                model: {
                    m: score(model, m, row["real_path"], row["sim_path"], raw=True)
                    for m in methods
                }
                for model in features
            }
            row["complete"] = all(
                v is not None
                for values in row["scores"].values()
                for v in values.values()
            )
            row["mask_ratios"] = [
                variants.get(by_path[p]["image_id"], {}).get("mask_ratio")
                for p in (row["real_path"], row["sim_path"])
            ]
            rows.append(row)
        valid = [r for r in rows if r["complete"]]
        report = dict(
            expected=len(rows),
            complete=len(valid),
            excluded_incomplete=[r["task_id"] for r in rows if not r["complete"]],
            global_scores={
                model: {
                    m: stats([r["scores"][model][m] for r in valid]) for m in methods
                }
                for model in features
            },
            deltas={
                model: {
                    m: stats(
                        [
                            r["scores"][model][m] - r["scores"][model]["original"]
                            for r in valid
                        ]
                    )
                    for m in methods[1:]
                }
                for model in features
            },
            common_environments=[],
            account_sensitivity=[],
            controls=[],
        )
        for app in sorted({t["app"] for selected in tasks.values() for t in selected}):
            envs = sorted({r["sim_environment"] for r in rows if r["app"] == app})
            if not envs:
                continue
            common = set.intersection(
                *[
                    {
                        (r["real_environment"], r["page_id"])
                        for r in valid
                        if r["app"] == app and r["sim_environment"] == e
                    }
                    for e in envs
                ]
            )
            for env in envs:
                members = [
                    r for r in valid if r["app"] == app and r["sim_environment"] == env
                ]
                subset = [
                    r
                    for r in members
                    if (r["real_environment"], r["page_id"]) in common
                ]
                groups = defaultdict(list)
                for r in subset:
                    groups[r["group"]].append(r)
                report["common_environments"].append(
                    dict(
                        app=app,
                        environment=env,
                        pairs=len(subset),
                        groups=len(groups),
                        scores={
                            model: {
                                m: stats(
                                    [
                                        statistics.mean(
                                            r["scores"][model][m] for r in group
                                        )
                                        for group in groups.values()
                                    ]
                                )
                                for m in methods
                            }
                            for model in features
                        },
                    )
                )
                ref_names = (
                    ("eleme_real_native", "eleme_real_native_account2")
                    if app == "eleme"
                    else ("wechat_real_account1", "wechat_real_account2")
                )
                accounts = [
                    {r["page_id"]: r for r in members if r["real_environment"] == ref}
                    for ref in ref_names
                ]
                pages = accounts[0].keys() & accounts[1].keys()
                report["account_sensitivity"].append(
                    dict(
                        app=app,
                        environment=env,
                        paired_pages=len(pages),
                        absolute_gap={
                            model: {
                                m: stats(
                                    [
                                        abs(
                                            accounts[0][p]["scores"][model][m]
                                            - accounts[1][p]["scores"][model][m]
                                        )
                                        for p in pages
                                    ]
                                )
                                for m in methods
                            }
                            for model in features
                        },
                    )
                )
        # Fixed semantic page-type mismatch stress check, not human-ground-truth accuracy.
        controls = []
        for env in sorted({r["sim_environment"] for r in rows}):
            seen = set()
            for r in valid:
                if r["sim_environment"] != env or r["page_id"] in seen:
                    continue
                t = next(t for t in selected if t["task_id"] == r["task_id"])
                candidates = [
                    i
                    for i in images
                    if i["environment_id"] == env
                    and i["page_type"] != t["comparison"]["page_type"]
                    and i["page_id"] != r["page_id"]
                    and (
                        r["app"] != "eleme"
                        or {i["page_id"], r["page_id"]} not in ({7, 8}, {11, 12})
                    )
                ]
                if not candidates:
                    continue
                other = candidates[-1]
                control = dict(
                    positive_task_id=r["task_id"],
                    real_path=r["real_path"],
                    wrong_sim_path=other["path"],
                    provenance="automatic_different_page_type_not_human_validated",
                    scores={},
                )
                for model in features:
                    control["scores"][model] = {
                        m: dict(
                            positive=r["scores"][model][m],
                            negative=score(model, m, r["real_path"], other["path"]),
                        )
                        for m in methods
                    }
                controls.append(control)
                seen.add(r["page_id"])
                if len(seen) == 4:
                    break
        for model in features:
            for m in methods:
                gaps = [
                    c["scores"][model][m]["positive"]
                    - c["scores"][model][m]["negative"]
                    for c in controls
                    if c["scores"][model][m]["negative"] is not None
                ]
                report["controls"].append(
                    dict(
                        model=model,
                        method=m,
                        provenance="automatic_stress_check_not_accuracy",
                        gap=stats(gaps),
                        positive_win_fraction=float(np.mean(np.asarray(gaps) > 0))
                        if gaps
                        else None,
                    )
                )
        dest = work / "analysis" / scheme
        dest.mkdir(parents=True, exist_ok=True)
        write_rows(dest / "paired_scores.jsonl", rows)
        write_rows(dest / "stress_controls.jsonl", controls)
        write(dest / "statistics.json", report)
        scheme_reports[scheme] = report
    real_real = {}
    rr_registry = work / "real_real/schemes.json"
    if rr_registry.exists():
        registry = json.loads(rr_registry.read_text(encoding="utf-8"))
        if registry["source_metadata_sha256"] != exp["source_files"]["metadata.json"]:
            raise ValueError("Stale Real–Real mapping")
        for scheme in ("A", "B"):
            path = work / "real_real" / f"{scheme}.jsonl"
            if sha(path) != registry["manifests_sha256"][scheme]:
                raise ValueError("Real–Real manifest changed")
            rows = []
            for task in read(path):
                row = dict(
                    task_id=task["task_id"],
                    app=task["app"],
                    page_id=task["page_id"],
                    group=task["aggregation_group_id"],
                    reference_environment=task["real"]["environment_id"],
                    comparison_environment=task["comparison"]["environment_id"],
                )
                row["scores"] = {
                    model: {
                        m: score(
                            model, m, task["real"]["path"], task["comparison"]["path"]
                        )
                        for m in methods
                    }
                    for model in features
                }
                row["raw_cosines"] = {
                    model: {
                        m: score(
                            model,
                            m,
                            task["real"]["path"],
                            task["comparison"]["path"],
                            raw=True,
                        )
                        for m in methods
                    }
                    for model in features
                }
                row["complete"] = all(
                    v is not None
                    for values in row["scores"].values()
                    for v in values.values()
                )
                rows.append(row)
            summary = {}
            for app in sorted(
                {t["app"] for selected in tasks.values() for t in selected}
            ):
                valid = [r for r in rows if r["app"] == app and r["complete"]]
                groups = defaultdict(list)
                for r in valid:
                    groups[r["group"]].append(r)
                summary[app] = dict(
                    expected=sum(r["app"] == app for r in rows),
                    complete=len(valid),
                    excluded_incomplete=[
                        r["task_id"]
                        for r in rows
                        if r["app"] == app and not r["complete"]
                    ],
                    scores={
                        model: {
                            m: stats([r["scores"][model][m] for r in valid])
                            for m in methods
                        }
                        for model in features
                    },
                    deltas={
                        model: {
                            m: stats(
                                [
                                    r["scores"][model][m]
                                    - r["scores"][model]["original"]
                                    for r in valid
                                ]
                            )
                            for m in methods[1:]
                        }
                        for model in features
                    },
                    group_balanced={
                        model: {
                            m: stats(
                                [
                                    statistics.mean(r["scores"][model][m] for r in g)
                                    for g in groups.values()
                                ]
                            )
                            for m in methods
                        }
                        for model in features
                    },
                )
            real_real[scheme] = summary
            write_rows(
                work / "analysis/real_real" / scheme / "paired_scores.jsonl", rows
            )
            write(work / "analysis/real_real" / scheme / "statistics.json", summary)
    quality = dict(
        detections=Counter(r["status"] for r in read(work / "detections.jsonl")),
        render_statuses=Counter(r["status"] for r in variants.values()),
        human_QA="not_exhaustively_labeled",
        local_complex_regions=sum(
            r.get("details", {}).get("local", {}).get("complex_regions", 0)
            for r in variants.values()
        ),
        texture_low_contrast=sum(
            d.get("low_contrast_fallback", False)
            for r in variants.values()
            for d in r.get("details", {}).get("texture", {}).get("regions", [])
        ),
    )
    quality["pairing_coverage"] = {}
    for scheme, relative in {
        "A": "pairing_schemes/A_full.jsonl",
        "B": "pairing_schemes/B_single_view.jsonl",
    }.items():
        if relative in exp.get("source_files", {}):
            candidates = read(root / relative)
            quality["pairing_coverage"][scheme] = {
                "candidate": len(candidates),
                "statuses": dict(Counter(t["status"] for t in candidates)),
            }
    write(
        work / "analysis/summary.json",
        dict(
            schemes=scheme_reports,
            real_real=real_real,
            quality=quality,
            encoder_audit=audit,
            encoder_preprocessing=geometries[0],
        ),
    )
    report_markdown(work, scheme_reports, quality, models, methods)
    with (work / "analysis/report.md").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n## 编码器几何预处理\n\n"
            + json.dumps(geometries[0], ensure_ascii=False)
            + "\n"
        )
    if real_real:
        lines = [
            "",
            "## Real–Real：两个真实账号直接对比",
            "",
            "这是两张Real图片直接计算相似度，区别于同一Sim使用两个Real参照时的评分差距。原图与四种处理分别比较；不能从账号对照单独推断差异均由文字造成。",
            "",
            "| 方案 | APP | 有效/候选 | 模型 | " + " | ".join(methods) + " |",
            "|---|---|---:|---|" + "---:|" * len(methods),
        ]
        for scheme, apps in real_real.items():
            for app, entry in apps.items():
                for model in features:
                    vals = [entry["scores"][model][m]["mean"] for m in methods]
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                scheme,
                                app,
                                f"{entry['complete']}/{entry['expected']}",
                                model,
                                *[
                                    f"{v:.4f}" if v is not None else "缺失"
                                    for v in vals
                                ],
                            ]
                        )
                        + " |"
                    )
        with (work / "analysis/report.md").open("a", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
    plots(work, scheme_reports, models, methods)
    print(
        json.dumps(
            {
                s: dict(expected=r["expected"], complete=r["complete"])
                for s, r in scheme_reports.items()
            }
        ),
        flush=True,
    )


def report_markdown(work, reports, quality, models, methods):
    lines = [
        "# 五条件视觉保真度实验",
        "",
        f"质量统计：{dict(quality)}。",
        "使用选定模型与条件的共同成功任务，不将失败补零。均分不是处理方法准确性的排名。",
        "OCR误检漏检尚未完成全量人工标注；自动不同类型页面压力对照不是金标准。",
        "",
    ]

    def fmt(value):
        return f"{value:.4f}" if value is not None else "缺失"

    for scheme, report in reports.items():
        lines.extend(
            [
                f"## 方案 {scheme}",
                "",
                f"共同成功：{report['complete']}/{report['expected']}。",
                "",
                "| 条件 | 模型 | 均值 | 相对原图 |",
                "|---|---|---:|---:|",
            ]
        )
        for method in methods:
            for model in models:
                delta = (
                    report["deltas"][model][method]["mean"]
                    if method != "original"
                    else 0.0
                )
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            method,
                            model,
                            fmt(report["global_scores"][model][method]["mean"]),
                            fmt(delta),
                        ]
                    )
                    + " |"
                )
        lines.extend(
            [
                "",
                "### 共同页面、页面组等权",
                "",
                "| 环境 | 配对/组 | 条件 | 模型 | 均值 |",
                "|---|---:|---|---|---:|",
            ]
        )
        for entry in report["common_environments"]:
            for method in methods:
                for model in models:
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                entry["environment"],
                                f"{entry['pairs']}/{entry['groups']}",
                                method,
                                model,
                                fmt(entry["scores"][model][method]["mean"]),
                            ]
                        )
                        + " |"
                    )
        lines.extend(
            [
                "",
                "### 同一Sim对两个Real账号的平均绝对分差",
                "",
                "| 环境 | 页数 | 条件 | 模型 | 平均绝对分差 |",
                "|---|---:|---|---|---:|",
            ]
        )
        for entry in report["account_sensitivity"]:
            for method in methods:
                for model in models:
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                entry["environment"],
                                str(entry["paired_pages"]),
                                method,
                                model,
                                fmt(entry["absolute_gap"][model][method]["mean"]),
                            ]
                        )
                        + " |"
                    )
        lines.append("")
    lines.extend(
        [
            "## 解释边界",
            "",
            "保留内容数量、头像图片、背景与滚动差异。模糊可能仍可读，伪笔画可能引入共同纹理，局部修补可能涂抹。",
            "原图基线与其他条件使用同一几何协议；A/B分数变化同时包含页面范围差异。原尺寸图片预览见../gallery.html。",
        ]
    )
    (work / "analysis/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plots(work, reports, models, methods):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(reports),
        len(models),
        figsize=(6 * len(models), 3.5 * len(reports)),
        squeeze=False,
        layout="constrained",
    )
    for i, (scheme, report) in enumerate(reports.items()):
        for j, model in enumerate(models):
            values = [report["global_scores"][model][m]["mean"] for m in methods]
            axes[i, j].bar(methods, [v if v is not None else np.nan for v in values])
            axes[i, j].set(
                title=f"{scheme} / {model} (common valid tasks)", ylim=(0, 1)
            )
    fig.savefig(work / "analysis/method_means.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(
        len(reports),
        len(models),
        figsize=(6 * len(models), 3.5 * len(reports)),
        squeeze=False,
        layout="constrained",
    )
    for i, scheme in enumerate(reports):
        rows = [
            r
            for r in read(work / "analysis" / scheme / "paired_scores.jsonl")
            if r["complete"]
        ]
        for j, model in enumerate(models):
            values = [
                [r["scores"][model][m] - r["scores"][model]["original"] for r in rows]
                for m in methods
                if m != "original"
            ]
            if rows and values:
                axes[i, j].boxplot(
                    values,
                    tick_labels=[m for m in methods if m != "original"],
                    showfliers=False,
                )
            axes[i, j].axhline(0, color="gray", linestyle="--")
            axes[i, j].set(title=f"{scheme} / {model}", ylabel="Processed - Original")
    fig.savefig(work / "analysis/paired_deltas.png", dpi=160)
    plt.close(fig)
