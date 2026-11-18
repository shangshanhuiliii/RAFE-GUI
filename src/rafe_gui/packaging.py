"""UTF-8/Zip64 bundles containing only the selected independent project assets."""

import json
import zipfile
from pathlib import Path

from .storage import sha, write


def export_project(root, output, with_models=False):
    root = Path(root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Archive exists; choose another output filename")
    allowed = {
        "src",
        "configs",
        "docs",
        "tests",
        "tools",
        "third_party",
        "data",
        "models",
    }
    root_files = {".env.example", ".gitignore", "README.md", "pyproject.toml"}
    files = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file() or any(
            part in ("__pycache__", ".pytest_cache", "dist", ".git", ".tools")
            or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path.name in (".env", "rafe.env", "paths.env") or path.suffix in (
            ".pyc",
            ".tmp",
        ):
            continue
        if len(relative.parts) == 1 and relative.name not in root_files:
            continue
        if len(relative.parts) > 1 and relative.parts[0] not in allowed:
            continue
        if (
            relative.parts[0] == "models"
            and not with_models
            and relative != Path("models/models.lock.json")
        ):
            continue
        if path == output:
            continue
        files["RAFE-GUI/" + relative.as_posix()] = path
    manifest = {
        name: {"sha256": sha(path), "bytes": path.stat().st_size}
        for name, path in files.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "x", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr(
            "RAFE-GUI/BUNDLE_MANIFEST.json",
            json.dumps(
                {
                    "with_models": with_models,
                    "files": manifest,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    write(
        output.with_suffix(".sha256.json"),
        {"file": output.name, "bytes": output.stat().st_size, "sha256": sha(output)},
    )
    return {
        "output": str(output),
        "files": len(files),
        "GiB": output.stat().st_size / 2**30,
        "with_models": with_models,
    }
