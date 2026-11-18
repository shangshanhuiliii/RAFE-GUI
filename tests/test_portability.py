import ast
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from rafe_gui.packaging import export_project


def test_runtime_imports_have_no_legacy_package_dependencies():
    root = Path(__file__).resolve().parents[1] / "src/rafe_gui"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {
                    "rafe",
                    "offline",
                    "utils",
                }, path
            if isinstance(node, ast.Import):
                assert all(
                    alias.name.split(".")[0] not in {"rafe", "offline", "utils"}
                    for alias in node.names
                ), path


def test_utf8_archive_preserves_chinese_paths(tmp_path):
    source = tmp_path / "project"
    target = source / "data/dataset/Real/Eleme/独立APP/1_real_eleme.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"raw-image")
    output = tmp_path / "transfer.zip"
    export_project(source, output)
    name = "RAFE-GUI/data/dataset/Real/Eleme/独立APP/1_real_eleme.png"
    with zipfile.ZipFile(output) as archive:
        assert archive.getinfo(name).flag_bits & 0x800
        archive.extractall(tmp_path / "extracted")
    assert (tmp_path / "extracted" / name).read_bytes() == b"raw-image"


def test_isolated_cli_outside_project_does_not_import_gui_cc(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "rafe_gui",
            "run",
            "--project",
            str(root),
            "--run-id",
            "test_plan",
            "--dry-run",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["preprocessing"] == ["square-pad", "stretch"]
