"""Deterministic local background estimation; only masked pixels are replaced."""

import numpy as np

from ..storage import mask_image


def render(pixels, boxes, config):
    import cv2

    gray, union = mask_image(
        pixels, boxes, config["expand_pixels"], config.get("fill_rgb", [128] * 3)
    )
    options = config.get("fill", {"method": "gray"})
    method = options.get("method", "gray")
    if method == "gray":
        return gray, union, dict(method="gray")
    if method != "local-background":
        raise ValueError(f"Unsupported fill method: {method}")
    output = pixels.copy()
    details = []
    ring_px = int(options.get("ring_pixels", 6))
    threshold = float(options.get("dominant_fraction", 0.55))
    radius = float(options.get("inpaint_radius", 3))
    if ring_px < 1 or not 0 < threshold <= 1 or radius <= 0:
        raise ValueError("Invalid local background configuration")
    height, width = union.shape
    for index, box in enumerate(boxes):
        points = np.asarray(box)
        pad = config["expand_pixels"] + ring_px + int(np.ceil(radius)) + 3
        x0 = max(0, int(np.floor(points[:, 0].min())) - pad)
        y0 = max(0, int(np.floor(points[:, 1].min())) - pad)
        x1 = min(width, int(np.ceil(points[:, 0].max())) + pad + 1)
        y1 = min(height, int(np.ceil(points[:, 1].max())) + pad + 1)
        if x0 >= x1 or y0 >= y1:
            raise ValueError("Text box does not intersect image")
        original = pixels[y0:y1, x0:x1]
        shifted = (points - [x0, y0]).tolist()
        _, region = mask_image(original, [shifted], config["expand_pixels"])
        global_mask = union[y0:y1, x0:x1]
        # Prefer the inner edge: an external ring may cross a colored button border.
        inner = (region != 0) & (cv2.erode(region, np.ones((5, 5), np.uint8)) == 0)
        expanded = cv2.dilate(region, np.ones((2 * ring_px + 1,) * 2, np.uint8))
        outer = (expanded != 0) & (global_mask == 0)
        samples = original[inner]
        if not len(samples):
            samples = original[outer]
        if not len(samples):
            raise ValueError("No pixels available for local background estimation")
        # A mode in 8-level RGB bins resists text strokes, anti-aliasing and thin borders.
        bins = samples.astype(np.int32) // 8
        keys = bins[:, 0] * 1024 + bins[:, 1] * 32 + bins[:, 2]
        unique, counts = np.unique(keys, return_counts=True)
        winner = unique[counts.argmax()]
        fraction = float(counts.max() / len(samples))
        region_pixels = region != 0
        target = output[y0:y1, x0:x1]
        if fraction >= threshold:
            color = np.median(samples[keys == winner], axis=0).round().astype(np.uint8)
            target[region_pixels] = color
            detail = dict(
                box_index=index,
                method="local-mode",
                color_rgb=color.tolist(),
                dominant_fraction=fraction,
            )
        else:
            # Classical local interpolation, not generative reconstruction.
            repaired = cv2.inpaint(original, global_mask, radius, cv2.INPAINT_TELEA)
            target[region_pixels] = repaired[region_pixels]
            detail = dict(
                box_index=index,
                method="inpaint-telea",
                dominant_fraction=fraction,
                review_reason="nonuniform background or foreground-contaminated samples",
            )
        details.append(detail)
    if not np.array_equal(output[union == 0], pixels[union == 0]):
        raise RuntimeError("Local fill modified pixels outside mask")
    return (
        output,
        union,
        dict(
            method=method,
            regions=details,
            complex_regions=sum(d["method"] == "inpaint-telea" for d in details),
        ),
    )
