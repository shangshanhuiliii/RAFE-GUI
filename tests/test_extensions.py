import json
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import fixture

from rafe_gui.analysis.summary import analyze
from rafe_gui.cli import main
from rafe_gui.execution.state import run_lock
from rafe_gui.storage import read, sha, write, write_rows


def test_analysis_accepts_new_encoder_and_original_only(tmp_path):
    work, root, images = fixture(tmp_path)
    folder = work / "features/custom_encoder"
    folder.mkdir(parents=True)
    write(
        folder / "run.json",
        {
            "fingerprint": "test",
            "configuration": {"render_run_sha256": sha(work / "render.run.json")},
        },
    )
    write(folder / "parity.json", {"absolute_difference": 0.0, "tolerance": 2e-5})
    rows = []
    for image in images:
        key = image["image_id"] + "__original"
        path = folder / (key + ".npy")
        np.save(path, np.array([1.0, 0.0], np.float32))
        rows.append(
            {
                "key": key,
                "status": "ok",
                "protocol": "test",
                "feature_path": path.relative_to(work).as_posix(),
                "feature_sha256": sha(path),
            }
        )
    write_rows(folder / "index.jsonl", rows)
    analyze(SimpleNamespace(work=work, models=["custom_encoder"], methods=["original"]))
    summary = json.loads((work / "analysis/summary.json").read_text())
    assert summary["schemes"]["A"]["complete"] == 6
    assert (
        summary["schemes"]["A"]["global_scores"]["custom_encoder"]["original"]["mean"]
        == 1.0
    )


def test_status_detects_abandoned_run_and_distinguishes_live_lock(tmp_path, capsys):
    config = tmp_path / "configs/test.json"
    write(
        config,
        {
            "schema_version": 1,
            "experiment_type": "vlm",
            "project_root": "..",
            "paths": {"outputs": "runs"},
        },
    )
    folder = tmp_path / "runs/example"
    write(folder / "status.json", {"state": "running"})
    assert main(["status", "--config", str(config), "--run-id", "example"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "interrupted"
    with run_lock(folder):
        assert main(["status", "--config", str(config), "--run-id", "example"]) == 0
        assert json.loads(capsys.readouterr().out)["state"] == "running"


def test_original_only_preparation_never_initializes_ocr(tmp_path, monkeypatch):
    from rafe_gui.config import Config
    from rafe_gui.execution.prepare import prepare
    from rafe_gui.transforms import detectors

    work, root, images = fixture(tmp_path)
    tasks = read(work / "A.jsonl")
    selected = read(work / "B.jsonl")
    for task in tasks + selected:
        task.update(schema_version="1.0", metadata_sha256=sha(root / "metadata.json"))
    write_rows(root / "eval_pairs.jsonl", tasks)
    write_rows(root / "pairing_schemes/A_full.jsonl", tasks)
    write_rows(root / "pairing_schemes/B_single_view.jsonl", selected)
    monkeypatch.setattr(
        detectors, "create", lambda *a: pytest.fail("Original-only must not load OCR")
    )
    config = Config(
        tmp_path,
        {
            "paths": {"dataset": "dataset", "prepared": "new_prepared"},
            "variants": ["original"],
        },
        tmp_path / "config.json",
    )
    output = prepare(config)
    assert len(read(output / "variants.jsonl")) == len(images)
    assert all(r["status"] == "not_required" for r in read(output / "detections.jsonl"))
    assert all(r["mask_ratio"] is None for r in read(output / "variants.jsonl"))
