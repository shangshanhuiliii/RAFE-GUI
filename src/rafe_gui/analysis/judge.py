"""Summarize the current VLM run from its own score file."""

from collections import Counter, defaultdict
from statistics import mean

from ..storage import read, write


def summarize_judge(directory):
    rows = read(directory / "scores.jsonl")
    values = defaultdict(list)
    statuses = Counter(row["status"] for row in rows)
    for row in rows:
        if row["status"] != "ok":
            continue
        for metric, value in row["result"]["metrics"].items():
            if value is not None:
                values[metric].append(value)
    scores = {
        metric: {"n": len(items), "mean": mean(items)}
        for metric, items in values.items()
    }
    result = {"records": len(rows), "statuses": dict(statuses), "metrics": scores}
    output = directory / "analysis"
    write(output / "summary.json", result)
    status_text = "，".join(
        f"{name} {count}" for name, count in sorted(statuses.items())
    )
    lines = [
        "# VLM Judge 结果",
        "",
        f"共 {len(rows)} 条结果。{status_text}。",
        "",
        "| 指标 | 有效样本 | 均值 |",
        "|---|---:|---:|",
    ]
    for metric, score in sorted(scores.items()):
        lines.append(f"| {metric} | {score['n']} | {score['mean']:.4f} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
