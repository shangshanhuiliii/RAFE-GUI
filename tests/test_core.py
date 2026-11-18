import numpy as np
import pytest
from PIL import Image

from rafe_gui.storage import inside, mask_image, sha, verify_variant


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_union_and_two_pixel_expansion():
    original = np.full((20, 20, 3), 240, np.uint8)
    one, binary = mask_image(original, [box(5, 5, 8, 8)], expand=2)
    duplicate, binary2 = mask_image(original, [box(5, 5, 8, 8)] * 2, expand=2)
    assert np.array_equal(one, duplicate)
    assert np.array_equal(binary, binary2)
    assert np.count_nonzero(binary) == 64
    assert np.all(one[3:11, 3:11] == 128)
    assert np.array_equal(one[binary == 0], original[binary == 0])


def test_empty_clipping_and_invalid_coordinates():
    original = np.arange(300, dtype=np.uint8).reshape(10, 10, 3)
    output, binary = mask_image(original, [])
    assert np.array_equal(output, original) and not binary.any()
    _, binary = mask_image(original, [box(-2, -2, 1, 1)], expand=2)
    assert np.count_nonzero(binary) == 16
    with pytest.raises(ValueError):
        mask_image(original, [[[float("nan"), 0]] * 4])


def test_variant_tampering_and_path_escape(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), "white").save(source)
    row = dict(source_sha256=sha(source), status="ok")
    for field in ("masked", "mask", "overlay", "input"):
        target = tmp_path / (field + ".png")
        Image.new("RGB", (8, 8), "gray").save(target)
        row[field + "_path"] = target.name
        row[field + "_sha256"] = sha(target)
    assert verify_variant(tmp_path, row, source) == tmp_path / "masked.png"
    (tmp_path / "masked.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_variant(tmp_path, row, source)
    with pytest.raises(ValueError, match="outside"):
        inside(tmp_path, "../escape.png")
    row["status"] = "error"
    with pytest.raises(ValueError, match="failed"):
        verify_variant(tmp_path, row, source)


def test_source_changes_invalidate_cached_variant(tmp_path):
    source = tmp_path / "original.png"
    Image.new("RGB", (8, 8), "white").save(source)
    row = dict(source_sha256=sha(source), status="ok")
    Image.new("RGB", (8, 8), "black").save(source)
    with pytest.raises(ValueError, match="stale"):
        verify_variant(tmp_path, row, source)
