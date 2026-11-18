import json
from types import SimpleNamespace

import pytest
from helpers import fixture

from rafe_gui.analysis.compare import compare
from rafe_gui.data.experiment import METHODS, build_real_real
from rafe_gui.execution.pipeline import clone_prepared
from rafe_gui.storage import read, write


def comparison_fixture(tmp_path):
    source, root, images = fixture(tmp_path)
    works = {}
    for mode, score in (("square-pad", 0.4), ("stretch", 0.7)):
        work = tmp_path / mode
        clone_prepared(SimpleNamespace(path=lambda key: root), source, work)
        build_real_real(SimpleNamespace(work=work))
        works[mode] = work
        audit = {}
        for model in ("siglip", "dino"):
            run = dict(
                fingerprint="test-" + mode,
                configuration=dict(
                    dtype="float32",
                    models="same-models",
                    encoder_preprocessing={"mode": mode, "output_size_wh": [560, 560]},
                ),
            )
            write(work / "features" / model / "run.json", run)
            audit[model] = {"run": run}
        write(work / "analysis/summary.json", {"encoder_audit": audit})
        from rafe_gui.storage import write_rows

        for kind in ("real_sim", "real_real"):
            for scheme in ("A", "B"):
                tasks = read(
                    work / (scheme + ".jsonl")
                    if kind == "real_sim"
                    else work / "real_real" / (scheme + ".jsonl")
                )
                rows = []
                for task in tasks:
                    scores = {
                        model: {m: score for m in METHODS}
                        for model in ("siglip", "dino")
                    }
                    # A bad score in one geometry/model/condition must remove the pair from all means.
                    if (
                        mode == "stretch"
                        and task["app"] == "eleme"
                        and task["page_id"] == 7
                    ):
                        scores["dino"]["texture"] = None
                    rows.append(dict(task_id=task["task_id"], scores=scores))
                folder = (
                    work
                    / "analysis"
                    / (scheme if kind == "real_sim" else "real_real/" + scheme)
                )
                write_rows(folder / "paired_scores.jsonl", rows)
    return works


def test_geometry_comparison_uses_joint_twenty_scores_and_direct_real_real(tmp_path):
    works = comparison_fixture(tmp_path)
    output = tmp_path / "comparison"
    compare(
        SimpleNamespace(
            pad=works["square-pad"], stretch=works["stretch"], output=output
        )
    )
    result = json.loads((output / "summary.json").read_text())["results"]
    a = result["real_sim"]["A"]
    assert a["expected"] == 6 and a["complete"] == 4
    assert len(a["excluded_incomplete"]) == 2
    for model in ("siglip", "dino"):
        for method in METHODS:
            scores = a["scores"][model][method]
            assert scores["square-pad"]["n"] == 4 and scores["stretch"]["n"] == 4
            assert scores["stretch_minus_pad"]["mean"] == pytest.approx(0.3)
    rr = result["real_real"]["A"]
    assert rr["expected"] == 3 and rr["complete"] == 2
    assert rr["by_app"]["eleme"]["complete"] == 1
    assert (output / "report.md").is_file()


def test_geometry_comparison_rejects_other_protocol_changes(tmp_path):
    works = comparison_fixture(tmp_path)
    work = works["stretch"]
    run = json.loads((work / "features/dino/run.json").read_text())
    run["configuration"]["dtype"] = "float16"
    write(work / "features/dino/run.json", run)
    summary = json.loads((work / "analysis/summary.json").read_text())
    summary["encoder_audit"]["dino"]["run"] = run
    write(work / "analysis/summary.json", summary)
    with pytest.raises(ValueError, match="beyond geometry"):
        compare(
            SimpleNamespace(
                pad=works["square-pad"], stretch=work, output=tmp_path / "comparison"
            )
        )
