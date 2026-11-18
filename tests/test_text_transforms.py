import numpy as np

from rafe_gui.transforms.text import apply

BOX = [[12, 12], [72, 12], [72, 42], [12, 42]]


def source():
    pixels = np.full((60, 90, 3), 250, np.uint8)
    pixels[20:34, 20:40] = 20
    return pixels


def test_all_methods_share_mask_and_preserve_outside():
    pixels = source()
    reference = None
    for method in ("gray", "local", "gaussian", "texture"):
        output, mask, details = apply(pixels, [BOX], method)
        if reference is None:
            reference = mask
        assert np.array_equal(mask, reference)
        assert np.array_equal(output[mask == 0], pixels[mask == 0])
        assert output.shape == pixels.shape and output.dtype == np.uint8


def test_texture_does_not_copy_glyph_location_and_is_deterministic():
    first = source()
    second = np.full_like(first, 250)
    second[20:34, 46:66] = 20
    a, _, details = apply(first, [BOX], "texture")
    b, _, _ = apply(second, [BOX], "texture")
    c, _, _ = apply(first, [BOX], "texture")
    assert np.array_equal(a, b) and np.array_equal(a, c)
    assert not np.array_equal(a, first)
    row = details["regions"][0]
    assert abs(row["rendered_density"] - row["estimated_foreground_density"]) < 0.001


def test_gaussian_reduces_sharp_edge_energy():
    import cv2

    pixels = source()
    output, mask, detail = apply(pixels, [BOX], "gaussian")
    before = cv2.Laplacian(pixels, cv2.CV_32F).var()
    after = cv2.Laplacian(output, cv2.CV_32F).var()
    assert after < before
    assert detail["regions"][0]["sigma_source_pixels"] > 0


def test_empty_masks_preserve_original_for_each_method():
    for method in ("gray", "local", "gaussian", "texture"):
        pixels = source()
        result, mask, _ = apply(pixels, [], method)
        assert not mask.any() and np.array_equal(result, pixels)
