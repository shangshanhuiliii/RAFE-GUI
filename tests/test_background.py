import numpy as np

from rafe_gui.transforms.background import render

CONFIG = dict(
    expand_pixels=2,
    fill_rgb=[128] * 3,
    fill=dict(
        method="local-background",
        dominant_fraction=0.55,
        ring_pixels=6,
        inpaint_radius=3,
    ),
)
BOX = [[15, 15], [45, 15], [45, 35], [15, 35]]


def test_local_color_on_white_and_colored_buttons():
    for color in ([255, 255, 255], [255, 102, 0], [240, 248, 252]):
        original = np.full((60, 70, 3), 255, np.uint8)
        original[8:44, 8:54] = color
        original[18:32, 20:40] = 20
        output, mask, details = render(original, [BOX], CONFIG)
        assert details["regions"][0]["method"] == "local-mode"
        assert details["regions"][0]["color_rgb"] == color
        assert np.all(output[mask != 0] == color)
        assert np.array_equal(output[mask == 0], original[mask == 0])


def test_nonuniform_background_uses_explicit_fallback():
    original = np.zeros((60, 70, 3), np.uint8)
    original[:, :, 0] = np.arange(70, dtype=np.uint8)[None, :] * 3
    original[:, :, 1] = np.arange(60, dtype=np.uint8)[:, None] * 4
    original[18:32, 20:40] = 0
    output, mask, details = render(original, [BOX], CONFIG)
    assert details["complex_regions"] == 1
    assert np.array_equal(output[mask == 0], original[mask == 0])
    assert output.dtype == np.uint8


def test_empty_background_fill_preserves_all_pixels():
    original = np.full((30, 30, 3), 241, np.uint8)
    output, mask, details = render(original, [], CONFIG)
    assert not mask.any() and np.array_equal(output, original)
