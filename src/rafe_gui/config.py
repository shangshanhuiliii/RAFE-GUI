"""Resolve assets from an explicit project/config root, never a GUI-CC checkout."""

import importlib.metadata
import json
import os
from dataclasses import dataclass
from pathlib import Path


def versions(packages):
    values = {}
    for name in packages:
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def project_root(explicit=None):
    root = (
        Path(
            explicit
            or os.environ.get("RAFE_PROJECT_ROOT")
            or Path(__file__).resolve().parents[2]
        )
        .expanduser()
        .resolve()
    )
    if not (root / "configs").is_dir():
        raise ValueError(
            "Specify --project or RAFE_PROJECT_ROOT for the independent project"
        )
    return root


@dataclass(frozen=True)
class Config:
    root: Path
    data: dict
    file: Path

    def path(self, key):
        return (self.root / Path(self.data["paths"][key]).expanduser()).resolve()


def load_config(path=None, project=None):
    if path and Path(path).is_absolute():
        file = Path(path)
    else:
        file = project_root(project) / (path or "configs/text_fidelity.json")
    data = json.loads(file.read_text(encoding="utf-8"))
    root = (file.parent / data.get("project_root", "..")).resolve()
    if project:
        root = Path(project).expanduser().resolve()
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported experiment configuration")
    if data.get("experiment_type", "encoders") == "encoders":
        methods = data.get("variants", [])
        if (
            not methods
            or len(set(methods)) != len(methods)
            or "original" not in methods
        ):
            raise ValueError("Select unique variants including the original baseline")
        if not data.get("metrics") or not data.get("preprocessing"):
            raise ValueError("Select metrics and preprocessing")
    return Config(root, data, file.resolve())


def load_env(root):
    path = root / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.partition("=")
        if not sep:
            raise ValueError("Invalid .env assignment")
        value = value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name.strip(), value)
