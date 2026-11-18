import json
from pathlib import Path

import numpy as np
import pytest
from helpers import fixture

from rafe_gui.config import Config, load_config
from rafe_gui.execution.pipeline import run
from rafe_gui.execution.state import run_lock
from rafe_gui.storage import read, sha, write
from rafe_gui.transforms.text import DEFAULTS


def test_full_pipeline_separates_geometries_reports_and_resume(
    tmp_path, monkeypatch, capsys
):
    work, root, images = fixture(tmp_path)
    write(work / "render.run.json", {"configuration": {"parameters": DEFAULTS}})
    model = tmp_path / "models/stub.bin"
    model.parent.mkdir()
    model.write_bytes(b"stub")
    record = {
        "directory": "models",
        "files": {"stub.bin": {"bytes": 4, "sha256": sha(model)}},
    }
    write(tmp_path / "models/lock.json", {"siglip": record, "dino": record})
    data = {
        "paths": {
            "dataset": "dataset",
            "prepared": "prepared",
            "models": "models/lock.json",
            "outputs": "runs",
        },
        "preprocessing": ["square-pad", "stretch"],
        "metrics": ["siglip", "dino"],
        "variants": ["original", "gray", "local", "gaussian", "texture"],
        "devices": {"siglip": "cpu", "dino": "cpu"},
        "dtype": "float32",
    }
    config = Config(tmp_path, data, tmp_path / "config.json")
    from rafe_gui.execution import encode as worker
    from rafe_gui.execution import pipeline
    from rafe_gui.metrics.encoders import ENCODERS

    class FakeEncoder:
        def __init__(self, *args):
            pass

        def feature(self, path, spec):
            from rafe_gui.transforms.preprocessing import prepare_image

            image, geometry = prepare_image(path, spec)
            return np.array([1.0, 0.0], np.float32), geometry

        def cosine(self, *args):
            return 1.0

    monkeypatch.setitem(ENCODERS, "siglip", FakeEncoder)
    monkeypatch.setitem(ENCODERS, "dino", FakeEncoder)

    class Process:
        def __init__(self, command, **options):
            assert options.get("stdout") is None
            assert options["env"]["PYTHONUNBUFFERED"] == "1"

            def value(flag):
                return command[command.index(flag) + 1]

            self.code = (
                0
                if worker.encode(
                    config,
                    Path(value("--work")),
                    value("--model"),
                    value("--mode"),
                    value("--device"),
                )
                else 1
            )

        def wait(self):
            return self.code

        def poll(self):
            return self.code

    monkeypatch.setattr(pipeline.subprocess, "Popen", Process)
    output = run(config, None)
    assert output.name.startswith("run_")
    console = capsys.readouterr().out
    assert "[RAFE-GUI] Checking dataset and prepared images" in console
    assert "[square-pad | siglip] 10/45" in console
    assert "[stretch | dino] 45/45" in console
    assert f"[RAFE-GUI] Completed {output.name}" in console
    assert json.loads((output / "status.json").read_text())["state"] == "completed"
    summary = json.loads((output / "comparison/summary.json").read_text())["results"]
    assert summary["real_sim"]["A"]["complete"] == 6
    assert summary["real_real"]["A"]["complete"] == 3
    assert (output / "square-pad/features/dino/run.json").exists()
    assert (
        json.loads((output / "stretch/features/dino/run.json").read_text())[
            "configuration"
        ]["encoder_preprocessing"]["mode"]
        == "stretch"
    )
    with pytest.raises(ValueError, match="Run exists"):
        run(config, output.name)
    run(config, output.name, resume=True)
    assert len(read(output / "stretch/features/dino/index.jsonl")) == len(images) * 5


def test_runtime_failure_is_recorded_and_lock_released(tmp_path, monkeypatch):
    config = Config(
        tmp_path,
        {"paths": {"outputs": "runs"}, "metrics": []},
        tmp_path / "config.json",
    )
    from rafe_gui.execution import pipeline

    monkeypatch.setattr(
        pipeline,
        "prepare",
        lambda config: (_ for _ in ()).throw(ValueError("missing image")),
    )
    with pytest.raises(ValueError, match="missing image"):
        run(config, "failed")
    assert (
        json.loads((tmp_path / "runs/failed/status.json").read_text())["state"]
        == "failed"
    )
    with run_lock(tmp_path / "runs/failed"):
        pass


def test_paths_are_config_relative_not_working_directory(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    write(
        tmp_path / "configs/test.json",
        {
            "schema_version": 1,
            "experiment_type": "vlm",
            "project_root": "..",
            "paths": {"dataset": "data/dataset"},
        },
    )
    monkeypatch.chdir(tmp_path.parent)
    config = load_config(tmp_path / "configs/test.json")
    assert config.path("dataset") == tmp_path / "data/dataset"


def test_export_excludes_credentials_history_and_weights_unless_requested(tmp_path):
    import zipfile

    from rafe_gui.packaging import export_project

    (tmp_path / "src").mkdir()
    (tmp_path / "src/module.py").write_text("pass")
    (tmp_path / ".env").write_text("RAFE_API_KEY=secret")
    write(tmp_path / "models/models.lock.json", {})
    (tmp_path / "models/weight.bin").write_bytes(b"weight")
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive/old.json").write_text("{}")
    output = tmp_path / "dist/test.zip"
    export_project(tmp_path, output)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "RAFE-GUI/models/models.lock.json" in names
        assert not any(
            n.endswith(".env") or "weight.bin" in n or "/archive/" in n for n in names
        )
