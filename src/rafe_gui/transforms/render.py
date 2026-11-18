import html
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

from ..config import versions as environment
from ..data.experiment import METHODS, load
from ..storage import fingerprint, inside, read, rgb, sha, write, write_rows
from .text import DEFAULTS, apply


def render_all(args):
    BASE = Path(__file__).resolve().parent
    work, root, images, tasks, exp = load(args.work)
    methods = tuple(getattr(args, "methods", METHODS))
    detections = {r["image_id"]: r for r in read(work / "detections.jsonl")}
    config = dict(
        parameters=getattr(args, "parameters", DEFAULTS),
        detector_run_sha256=sha(work / "detect.run.json"),
        source_sha256={
            p: sha(BASE / p) for p in ("text.py", "background.py", "../storage.py")
        },
        packages=environment(
            ("numpy", "Pillow", "opencv-python-headless", "opencv-python")
        ),
    )
    key = fingerprint(config)
    sidecar = work / "render.run.json"
    if (
        sidecar.exists()
        and json.loads(sidecar.read_text(encoding="utf-8"))["fingerprint"] != key
    ):
        raise ValueError(
            "Detection results or render configuration changed; keep a new experiment directory"
        )
    write(sidecar, dict(fingerprint=key, configuration=config))
    index = work / "variants.jsonl"
    rows = {r["image_id"]: r for r in read(index)} if index.exists() else {}
    from .preprocessing import prepare_image, specification

    selected = images[: args.limit] if args.limit else images
    for n, image in enumerate(selected, 1):
        identifier = image["image_id"]
        previous = rows.get(identifier)
        detected = detections.get(identifier)
        detection_key = fingerprint(detected) if detected else None
        if previous and previous["status"] == "ok":
            if previous["recipe"] != key:
                raise ValueError("Stale variant recipe")
            if previous.get("detection_fingerprint") != detection_key:
                raise ValueError("Cached OCR boxes changed")
            for artifact in previous["artifacts"].values():
                if sha(inside(work, artifact["path"])) != artifact["sha256"]:
                    raise ValueError("Variant checksum mismatch")
            continue
        start = time.perf_counter()
        row = dict(
            image_id=identifier,
            source_path=image["path"],
            source_sha256=image["sha256"],
            recipe=key,
            detection_fingerprint=detection_key,
        )
        try:
            if detected is None:
                raise ValueError("OCR not processed yet")
            if detected["status"] != "ok":
                raise ValueError(
                    "Detection failed; all four methods excluded for this image"
                )
            pixels = rgb(root / image["path"])
            boxes = detected["boxes"]
            folder = work / "images" / identifier
            folder.mkdir(parents=True, exist_ok=True)
            local = apply(pixels, boxes, "local", config=config["parameters"])
            row.update(
                artifacts={},
                details={},
                boxes=boxes,
                box_count=len(boxes),
                mask_ratio=float((local[1] != 0).mean()),
            )
            Image.fromarray(local[1]).save(folder / "mask.png")
            overlay = Image.fromarray(pixels)
            draw = ImageDraw.Draw(overlay)
            for box in boxes:
                draw.line(
                    [tuple(p) for p in box] + [tuple(box[0])], fill="red", width=3
                )
            overlay.save(folder / "overlay.png")
            artifacts = {"mask": folder / "mask.png", "overlay": folder / "overlay.png"}
            for method in methods:
                if method == "original":
                    output = pixels
                    detail = {}
                elif method == "local":
                    output, mask, detail = local
                else:
                    output, mask, detail = apply(
                        pixels,
                        boxes,
                        method,
                        config=config["parameters"],
                        local_result=local,
                    )
                    if not __import__("numpy").array_equal(mask, local[1]):
                        raise ValueError("Methods must share exactly one binary mask")
                if not __import__("numpy").array_equal(
                    output[local[1] == 0], pixels[local[1] == 0]
                ):
                    raise ValueError(
                        "Transform modified pixels outside the shared mask"
                    )
                path = folder / f"{method}.png"
                Image.fromarray(output).save(path)
                artifacts[method] = path
                preview, _ = prepare_image(path, specification("square-pad"))
                path = folder / f"{method}_input.png"
                preview.save(path)
                artifacts[method + "_input"] = path
                row["details"][method] = detail
            row["artifacts"] = {
                m: dict(path=p.relative_to(work).as_posix(), sha256=sha(p))
                for m, p in artifacts.items()
            }
            row["status"] = "ok"
        except Exception as error:
            row.update(status="error", error=str(error))
        row["seconds"] = time.perf_counter() - start
        rows[identifier] = row
        write_rows(index, rows.values())
        print(
            f"{n}/{len(selected)} {image['path']} {row['status']} {row['seconds']:.2f}s",
            flush=True,
        )
    gallery(work, images, rows, methods)


def gallery(work, images, rows, methods):
    cards = [
        '<meta charset="utf-8"><title>Four text transforms</title>',
        "<style>body{font:15px sans-serif;background:#eee}section{margin:25px 8px}.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}img{width:100%}h3{overflow-wrap:anywhere}</style>",
        "<h1>Original / Gray / Local / Gaussian / Texture</h1><p>Exploratory automatic output; OCR false positives and missed text remain possible. Images load lazily.</p>",
    ]
    for image in images:
        row = rows.get(image["image_id"])
        if not row or row["status"] != "ok":
            continue
        cards.append(
            "<section><h2>"
            + html.escape(image["path"])
            + f'</h2><p>boxes={row["box_count"]}; mask={row["mask_ratio"]:.2%}</p><div class="grid">'
        )
        for method in methods:
            p = row["artifacts"][method]["path"]
            cards.append(
                f'<div><h3>{method}</h3><a href="{p}"><img loading="lazy" src="{p}"></a></div>'
            )
        overlay = row["artifacts"]["overlay"]["path"]
        cards.append(
            f'</div><details><summary>OCR detection overlay</summary><img style="width:450px" src="{overlay}" loading="lazy"></details></section>'
        )
    (work / "gallery.html").write_text("\n".join(cards), encoding="utf-8")
