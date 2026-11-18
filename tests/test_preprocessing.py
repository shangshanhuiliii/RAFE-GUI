from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor

from rafe_gui.metrics.encoders import Encoder
from rafe_gui.transforms.preprocessing import prepare_image, specification


@pytest.fixture
def screenshot(tmp_path):
    image = Image.new("RGB", (1260, 2800), "#909090")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1259, 99), fill="red")
    draw.rectangle((0, 2700, 1259, 2799), fill="blue")
    path = tmp_path / "page.png"
    image.save(path)
    return path


def test_padding_and_stretch_preserve_entire_page_with_different_geometry(screenshot):
    pad, a = prepare_image(screenshot, specification("square-pad"))
    stretch, b = prepare_image(screenshot, specification("stretch", 560, 560))
    rectangle, c = prepare_image(screenshot, specification("rectangle"))
    assert a["content_box_ltrb"] == [154, 0, 406, 560] and b["content_box_ltrb"] == [
        0,
        0,
        560,
        560,
    ]
    assert np.array_equal(np.asarray(pad)[:, 154:406], np.asarray(rectangle))
    assert np.array_equal(np.asarray(pad)[280, 0], [255, 255, 255])
    assert np.array_equal(np.asarray(stretch)[280, 0], [144, 144, 144])
    assert np.array_equal(np.asarray(stretch)[0, 0], [255, 0, 0])
    assert np.array_equal(np.asarray(stretch)[-1, -1], [0, 0, 255])


@pytest.mark.parametrize("width,height", ((251, 560), (252, 0), (0, 560), (252, 561)))
def test_patch_dimensions_validated(width, height):
    with pytest.raises(ValueError):
        specification("rectangle", width, height)


@pytest.mark.parametrize("mode", ("baseline", "rectangle", "square-pad", "stretch"))
@pytest.mark.parametrize(
    "name,leaf", (("siglip", "siglip-so400m-patch14-384"), ("dino", "dinov2-giant"))
)
def test_actual_checkpoint_processor_uses_protocol_geometry(
    screenshot, mode, name, leaf
):
    directory = Path(__file__).resolve().parents[1] / "models/encoders" / leaf
    if not directory.exists():
        pytest.skip("Local checkpoint config unavailable")
    processor = AutoImageProcessor.from_pretrained(
        str(directory), local_files_only=True, use_fast=False
    )
    observed = {}

    def model(pixel_values, **kwargs):
        observed.update(shape=tuple(pixel_values.shape), kwargs=kwargs)
        if mode != "baseline":
            center = pixel_values.shape[-1] // 2
            for row, rgb in ((0, [1.0, 0.0, 0.0]), (-1, [0.0, 0.0, 1.0])):
                expected = (
                    torch.tensor(rgb) - torch.tensor(processor.image_mean)
                ) / torch.tensor(processor.image_std)
                assert torch.allclose(
                    pixel_values[0, :, row, center], expected, atol=1e-5
                )
        features = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
        return SimpleNamespace(
            pooler_output=features, last_hidden_state=features[:, None, :]
        )

    encoder = Encoder(name, directory, "cpu", model=model, processor=processor)
    spec = specification(mode, 560, 560) if mode == "stretch" else specification(mode)
    score = encoder.cosine(screenshot, screenshot, spec)
    assert score == pytest.approx(0.6)
    if mode != "baseline":
        assert observed["shape"] == (2, 3, 560, 252 if mode == "rectangle" else 560)
    assert observed["kwargs"] == (
        {"interpolate_pos_encoding": True}
        if name == "siglip" and mode != "baseline"
        else {}
    )
