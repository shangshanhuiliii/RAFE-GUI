"""Gaussian suppression and deterministic nonsemantic stroke replacement."""

import copy

import numpy as np

from ..storage import mask_image
from .background import render as local_render

DEFAULTS = {
    "expand_pixels": 2,
    "fill_rgb": [128, 128, 128],
    "local": {
        "method": "local-background",
        "ring_pixels": 6,
        "dominant_fraction": 0.55,
        "inpaint_radius": 3,
    },
    "gaussian": {"sigma_height_ratio": 0.22, "min_sigma": 2.0, "max_sigma": 24.0},
    "texture": {
        "foreground_distance": 35.0,
        "min_density": 0.03,
        "max_density": 0.45,
        "cell_width_height_ratio": 0.85,
        "template": "three_disconnected_strokes_v1",
        "deterministic": True,
    },
}


def region(pixels, box, expand, pad=0):
    points = np.asarray(box, dtype=float)
    h, w = pixels.shape[:2]
    x0 = max(0, int(np.floor(points[:, 0].min())) - expand - pad)
    y0 = max(0, int(np.floor(points[:, 1].min())) - expand - pad)
    x1 = min(w, int(np.ceil(points[:, 0].max())) + expand + pad + 1)
    y1 = min(h, int(np.ceil(points[:, 1].max())) + expand + pad + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Out-of-frame text box")
    patch = pixels[y0:y1, x0:x1]
    shifted = (points - [x0, y0]).tolist()
    _, mask = mask_image(patch, [shifted], expand)
    return (slice(y0, y1), slice(x0, x1)), patch, mask, shifted


def _builtin_apply(pixels, boxes, method, config=None, local_result=None):
    import cv2

    cfg = copy.deepcopy(DEFAULTS if config is None else config)
    expand = cfg["expand_pixels"]
    gray, union = mask_image(pixels, boxes, expand, cfg["fill_rgb"])
    if method == "gray":
        return gray, union, dict(method=method)
    if method in ("local", "texture"):
        local_cfg = dict(
            expand_pixels=expand, fill_rgb=cfg["fill_rgb"], fill=cfg["local"]
        )
        base, _, local_details = (
            local_result
            if local_result is not None
            else local_render(pixels, boxes, local_cfg)
        )
        if method == "local":
            return base, union, local_details
    output = base.copy() if method == "texture" else pixels.copy()
    details = []
    for index, box in enumerate(boxes):
        points = np.asarray(box, dtype=float)
        box_height = max(1.0, float(np.linalg.norm(points[3] - points[0])))
        if method == "gaussian":
            opts = cfg["gaussian"]
            sigma = float(
                np.clip(
                    box_height * opts["sigma_height_ratio"],
                    opts["min_sigma"],
                    opts["max_sigma"],
                )
            )
            slices, patch, mask, shifted = region(
                pixels, box, expand, int(np.ceil(4 * sigma))
            )
            blurred = cv2.GaussianBlur(
                patch,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REFLECT_101,
            )
            output[slices][mask != 0] = blurred[mask != 0]
            details.append(
                dict(box_index=index, sigma_source_pixels=sigma, box_height=box_height)
            )
        elif method == "texture":
            opts = cfg["texture"]
            slices, patch, mask, shifted = region(pixels, box, expand)
            back = base[slices]
            _, core = mask_image(patch, [shifted], 0)
            distance = np.linalg.norm(patch.astype(float) - back.astype(float), axis=2)
            candidate = (distance > opts["foreground_distance"]) & (core != 0)
            density = float(
                np.clip(
                    candidate.sum() / max(1, (core != 0).sum()),
                    opts["min_density"],
                    opts["max_density"],
                )
            )
            samples = patch[candidate]
            if len(samples):
                color = np.median(samples, axis=0).round().astype(np.uint8)
            else:
                # Low contrast/failed color estimate: opposite luminance to local background.
                color = np.full(3, 40 if np.median(back) > 127 else 220, np.uint8)
            ph, pw = mask.shape
            prototype = np.zeros((ph, pw), np.uint8)
            count = max(1, round(pw / (box_height * opts["cell_width_height_ratio"])))
            cell = pw / count
            top = expand
            bottom = max(top + 1, ph - expand - 1)
            height = bottom - top
            for cell_id in range(count):
                left = cell_id * cell
                for a, b in [
                    ((0.15, 0.22), (0.57, 0.22)),
                    ((0.27, 0.40), (0.27, 0.78)),
                    ((0.55, 0.57), (0.83, 0.80)),
                ]:
                    p = (round(left + a[0] * cell), round(top + a[1] * height))
                    q = (round(left + b[0] * cell), round(top + b[1] * height))
                    cv2.line(prototype, p, q, 255, 1, lineType=cv2.LINE_8)
            distances = cv2.distanceTransform(
                (prototype == 0).astype(np.uint8), cv2.DIST_L2, 3
            )
            eligible = np.flatnonzero(core.ravel() != 0)
            amount = min(len(eligible), max(1, round(density * len(eligible))))
            # Stable tie breaking, identical geometry uses identical texture positions.
            order = np.argsort(distances.ravel()[eligible], kind="stable")
            strokes = np.zeros(ph * pw, bool)
            strokes[eligible[order[:amount]]] = True
            strokes = strokes.reshape(ph, pw)
            output[slices][strokes] = color
            details.append(
                dict(
                    box_index=index,
                    estimated_foreground_density=density,
                    rendered_density=float(strokes.sum() / max(1, len(eligible))),
                    color_rgb=color.tolist(),
                    cells=count,
                    low_contrast_fallback=not len(samples),
                    geometry="axis_aligned_clipped_to_polygon",
                )
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    if not np.array_equal(output[union == 0], pixels[union == 0]):
        raise RuntimeError("Transform changed pixels outside common mask")
    return (
        output,
        union,
        dict(
            method=method,
            regions=details,
            background_details=local_details if method == "texture" else None,
        ),
    )


def _fixed_method(name):
    def transform(pixels, boxes, config=None, local_result=None):
        return _builtin_apply(pixels, boxes, name, config, local_result)

    return transform


TEXT_TRANSFORMS = {
    name: _fixed_method(name) for name in ("gray", "local", "gaussian", "texture")
}


def apply(pixels, boxes, method, config=None, local_result=None):
    if method not in TEXT_TRANSFORMS:
        raise ValueError(f"Unknown text transform: {method}")
    return TEXT_TRANSFORMS[method](
        pixels, boxes, config=config, local_result=local_result
    )
