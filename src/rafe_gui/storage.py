"""Pure image/provenance operations; no OCR, encoder or network initialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def read(path):
    return [
        json.loads(s)
        for s in Path(path).read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def unique(rows, key):
    result = {}
    for row in rows:
        if row[key] in result:
            raise ValueError(f"Duplicate {key}: {row[key]}")
        result[row[key]] = row
    return result


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Path outside allowed directory: {relative}")
    return path


def rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def mask_image(pixels, polygons, expand=2, color=(128, 128, 128)):
    import cv2

    if not isinstance(expand, int) or expand < 0:
        raise ValueError("expand must be a nonnegative integer")
    if len(color) != 3 or any(
        not isinstance(v, int) or not 0 <= v <= 255 for v in color
    ):
        raise ValueError("Invalid RGB fill")
    height, width = pixels.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=float)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Expected finite quadrilateral")
        center = points.mean(axis=0)
        points = np.where(points < center, np.floor(points), np.ceil(points))
        points[:, 0] = points[:, 0].clip(0, width - 1)
        points[:, 1] = points[:, 1].clip(0, height - 1)
        cv2.fillPoly(mask, [points.astype(np.int32)], 255)
    if expand:
        mask = cv2.dilate(mask, np.ones((2 * expand + 1,) * 2, np.uint8), iterations=1)
    output = pixels.copy()
    output[mask != 0] = color
    assert np.array_equal(output[mask == 0], pixels[mask == 0])
    assert np.all(output[mask != 0] == np.asarray(color))
    return output, mask


def verify_variant(work, row, source):
    if row["source_sha256"] != sha(source) or row["status"] not in (
        "ok",
        "no_text_detected",
    ):
        raise ValueError("Missing/failed/stale image variant")
    for field in ("masked", "mask", "overlay", "input"):
        if sha(inside(work, row[field + "_path"])) != row[field + "_sha256"]:
            raise ValueError(f"Variant checksum mismatch: {field}")
    return inside(work, row["masked_path"])
