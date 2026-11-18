"""Independent source freezing, OCR and image rendering stages."""

import json
from pathlib import Path
from types import SimpleNamespace

from ..config import versions
from ..data import registry
from ..data.experiment import build_real_real, load
from ..storage import fingerprint, inside, read, rgb, sha, write, write_rows


def freeze(config):
    root = config.path("dataset")
    work = config.path("prepared")
    if (work / "experiment.json").exists():
        return load(work)
    paths = {
        "A": "pairing_schemes/A_full.jsonl",
        "B": "pairing_schemes/B_single_view.jsonl",
    }
    selected = {}
    for scheme, path in paths.items():
        tasks = read(root / path)
        registry.validate(root, tasks)
        selected[scheme] = [t for t in tasks if t["status"] == "ready"]
        write_rows(work / (scheme + ".jsonl"), selected[scheme])
    sources = {
        i["path"]: dict(i, image_id=fingerprint([i["path"], i["sha256"]])[:20])
        for t in selected["A"]
        for i in (t["real"], t["comparison"])
    }
    write_rows(
        work / "sources.jsonl",
        sorted(sources.values(), key=lambda i: (i["environment_id"], i["page_id"])),
    )
    import os

    write(
        work / "experiment.json",
        dict(
            schema_version="four_text_methods_v1",
            dataset_root=Path(os.path.relpath(root, work)).as_posix(),
            source_files={
                p: sha(root / p)
                for p in ("metadata.json", "eval_pairs.jsonl", *paths.values())
            },
            tasks_sha256={s: sha(work / (s + ".jsonl")) for s in paths},
            sources_sha256=sha(work / "sources.jsonl"),
        ),
    )
    return load(work)


def validate_prepared(work, parameters=None, methods=None):
    _, root, images, tasks, exp = load(work)
    rows = read(work / "variants.jsonl")
    if len(rows) != len(images) or {r["image_id"] for r in rows} != {
        i["image_id"] for i in images
    }:
        raise ValueError("Incomplete prepared image index")
    for row in rows:
        if row["status"] != "ok":
            raise ValueError("Failed prepared image")
        if methods and not set(methods) <= row["artifacts"].keys():
            raise ValueError(
                "Requested transform is missing; choose a new prepared directory"
            )
        for artifact in row["artifacts"].values():
            if sha(inside(work, artifact["path"])) != artifact["sha256"]:
                raise ValueError("Prepared artifact checksum mismatch")
    recipe = json.loads((work / "render.run.json").read_text(encoding="utf-8"))[
        "configuration"
    ]
    if parameters is not None and recipe["parameters"] != parameters:
        raise ValueError("Text parameters changed; choose another prepared directory")
    return images, tasks


def detect(config, work, root, images):
    from ..transforms.detectors import create

    cfg = json.loads(
        (config.root / config.data["ocr"]["config"]).read_text(encoding="utf-8")
    )
    cache = config.root / "models/ocr" / cfg.get("backend", "easyocr")
    weights = {
        p.relative_to(cache).as_posix(): sha(p)
        for p in cache.rglob("*")
        if p.suffix in (".onnx", ".pth")
    }
    if not weights:
        raise ValueError(f"OCR weights missing: {cache}")
    recipe = dict(
        config=cfg,
        weights=weights,
        versions=versions(
            ("rapidocr", "onnxruntime", "numpy", "opencv-python-headless")
        ),
        implementation="detector-v1",
    )
    key = fingerprint(recipe)
    if (work / "detect.run.json").exists() and json.loads(
        (work / "detect.run.json").read_text(encoding="utf-8")
    )["fingerprint"] != key:
        raise ValueError("OCR protocol differs; use another prepared directory")
    write(work / "detect.run.json", {"fingerprint": key, "recipe": recipe})
    rows = (
        {r["image_id"]: r for r in read(work / "detections.jsonl")}
        if (work / "detections.jsonl").exists()
        else {}
    )
    pending = [i for i in images if rows.get(i["image_id"], {}).get("status") != "ok"]
    if not pending:
        return
    reader, _ = create(
        cfg,
        SimpleNamespace(
            reuse_boxes_from=None, model_cache=cache, device="cpu", allow_download=False
        ),
        root / pending[0]["path"],
    )
    for n, image in enumerate(pending, 1):
        try:
            boxes = reader(rgb(root / image["path"]), root / image["path"])
            if not boxes:
                raise ValueError("No text detected; manual review required")
            row = dict(
                image_id=image["image_id"],
                source_sha256=image["sha256"],
                status="ok",
                boxes=boxes,
                recipe=key,
            )
        except Exception as error:
            row = dict(
                image_id=image["image_id"],
                source_sha256=image["sha256"],
                status="error",
                error=str(error),
            )
        rows[image["image_id"]] = row
        write_rows(work / "detections.jsonl", rows.values())
        print(f"OCR {n}/{len(pending)} {row['status']}", flush=True)


def prepare(config):
    work, root, images, tasks, exp = freeze(config)
    from ..transforms.text import DEFAULTS

    parameters = config.data.get("text_parameters", DEFAULTS)
    need_text = any(m != "original" for m in config.data["variants"])
    if (work / "variants.jsonl").exists():
        validate_prepared(
            work, parameters if need_text else None, config.data["variants"]
        )
    elif not need_text:
        render_originals(work, root, images)
    else:
        detect(config, work, root, images)
        from ..transforms.render import render_all

        render_all(
            SimpleNamespace(
                work=work,
                limit=None,
                parameters=parameters,
                methods=config.data["variants"],
            )
        )
        validate_prepared(work, parameters, config.data["variants"])
    build_real_real(SimpleNamespace(work=work, methods=config.data["variants"]))
    return work


def render_originals(work, root, images):
    """An Original-only encoder experiment has no OCR dependency."""
    import html

    from PIL import Image

    rows = []
    cards = ['<meta charset="utf-8"><h1>Original images</h1>']
    for image in images:
        path = work / "images" / image["image_id"] / "original.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(root / image["path"]) as source:
            source.convert("RGB").save(path)
        relative = path.relative_to(work).as_posix()
        rows.append(
            {
                "image_id": image["image_id"],
                "status": "ok",
                "mask_ratio": None,
                "details": {"original": {"text_processing": "not_requested"}},
                "artifacts": {"original": {"path": relative, "sha256": sha(path)}},
            }
        )
        cards.append(
            f'<p>{html.escape(image["path"])}</p><img width="280" loading="lazy" src="{relative}">'
        )
    write_rows(work / "variants.jsonl", rows)
    write_rows(
        work / "detections.jsonl",
        [{"image_id": i["image_id"], "status": "not_required"} for i in images],
    )
    write(work / "detect.run.json", {"status": "not_required"})
    write(
        work / "render.run.json",
        {
            "configuration": {
                "parameters": None,
                "methods": ["original"],
                "implementation": "original-only-v1",
            }
        },
    )
    (work / "gallery.html").write_text("\n".join(cards), encoding="utf-8")
