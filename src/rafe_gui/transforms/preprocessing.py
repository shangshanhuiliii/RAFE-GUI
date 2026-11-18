"""Whole-page encoder inputs: resize once, then optionally pad with white."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def _builtin_specification(mode: str, width: int = 252, height: int = 560) -> dict:
    if mode == "baseline":
        return {"mode": "baseline"}
    if mode not in ("rectangle", "square-pad", "stretch"):
        raise ValueError(f"Unknown preprocessing mode: {mode}")
    if min(width, height) < 14 or width % 14 or height % 14:
        raise ValueError(
            "Content width and height must be positive multiples of patch size 14"
        )
    return {
        "mode": mode,
        "content_size_wh": [width, height],
        "output_size_wh": [max(width, height)] * 2
        if mode == "square-pad"
        else [width, height],
        "resize": "PIL LANCZOS, stretch entire image to content_size_wh"
        if mode == "stretch"
        else "PIL LANCZOS, aspect-preserving fit within content_size_wh",
        "padding_rgb": [255, 255, 255] if mode == "square-pad" else None,
        "padding_alignment": "center" if mode == "square-pad" else None,
        "processor_resize": False,
        "processor_center_crop": False,
        "normalization": "Each checkpoint's original rescale and normalization",
        "siglip_interpolate_pos_encoding": True,
    }


def _builtin_prepare_image(path: Path, spec: dict) -> tuple[Image.Image, dict]:
    with Image.open(path) as original:
        image = original.convert("RGB")
    source = image.size
    width, height = spec["content_size_wh"]
    scale = min(width / source[0], height / source[1])
    size = (
        (width, height)
        if spec["mode"] == "stretch"
        else (max(1, round(source[0] * scale)), max(1, round(source[1] * scale)))
    )
    # Avoid accidentally stretching an image with an unexpected aspect ratio.
    if spec["mode"] == "rectangle" and size != (width, height):
        raise ValueError(
            f"{path}: aspect ratio does not fit {width}x{height}; "
            "choose matching content dimensions or use square-pad"
        )
    resized = image.resize(size, Image.Resampling.LANCZOS)
    if spec["mode"] == "square-pad":
        side = max(width, height)
        left, top = (side - size[0]) // 2, (side - size[1]) // 2
        image = Image.new("RGB", (side, side), tuple(spec["padding_rgb"]))
        image.paste(resized, (left, top))
    else:
        image = resized
        left = top = 0
    return image, {
        "source_size_wh": list(source),
        "resized_content_size_wh": list(size),
        "input_size_wh": list(image.size),
        "content_box_ltrb": [left, top, left + size[0], top + size[1]],
    }


PREPROCESSORS = {
    name: (_builtin_specification, _builtin_prepare_image)
    for name in ("baseline", "rectangle", "square-pad", "stretch")
}


def specification(mode, width=252, height=560):
    if mode not in PREPROCESSORS:
        raise ValueError(f"Unknown preprocessing: {mode}")
    return PREPROCESSORS[mode][0](mode, width, height)


def prepare_image(path, spec):
    return PREPROCESSORS[spec["mode"]][1](path, spec)
