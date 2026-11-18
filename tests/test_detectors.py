import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from rafe_gui.storage import rgb, sha, write, write_rows
from rafe_gui.transforms.detectors import create


def test_cached_detections_require_exact_source(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), "white").save(source)
    write_rows(
        tmp_path / "images.jsonl",
        [
            dict(
                source_path="source.png",
                source_sha256=sha(source),
                status="ok",
                boxes=[],
            )
        ],
    )
    write(tmp_path / "mask.run.json", dict(recipe="test"))
    detect, weights = create({}, SimpleNamespace(reuse_boxes_from=tmp_path), source)
    assert detect(rgb(source), source) == []
    Image.new("RGB", (8, 8), "black").save(source)
    with pytest.raises(ValueError, match="stale"):
        detect(rgb(source), source)


def test_rapidocr_fixes_model_and_disables_recognition(tmp_path, monkeypatch):
    cache = tmp_path / "rapidocr"
    cache.mkdir()
    (cache / "ch_PP-OCRv5_det_server.onnx").write_bytes(b"model")
    source = tmp_path / "red.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(source)

    def engine(params):
        assert (
            params["Det.ocr_version"] == "v5" and params["Det.model_type"] == "server"
        )
        assert params["Det.mean"] == [0.485, 0.456, 0.406]
        assert params["Det.std"] == [0.229, 0.224, 0.225]
        assert params["EngineConfig.onnxruntime.intra_op_num_threads"] == 4
        assert not params["Global.use_cls"] and not params["Global.use_rec"]

        def infer(pixels, **kwargs):
            assert pixels[0, 0].tolist() == [0, 0, 255]
            assert kwargs == dict(use_det=True, use_cls=False, use_rec=False)
            return SimpleNamespace(boxes=np.array([[[1, 1], [6, 1], [6, 6], [1, 6]]]))

        return infer

    monkeypatch.setitem(
        sys.modules,
        "rapidocr",
        SimpleNamespace(
            RapidOCR=engine,
            EngineType=SimpleNamespace(ONNXRUNTIME="ort"),
            LangDet=SimpleNamespace(CH="ch"),
            ModelType=SimpleNamespace(SERVER="server"),
            OCRVersion=SimpleNamespace(PPOCRV5="v5"),
        ),
    )
    detect, weights = create(
        dict(backend="rapidocr", rapidocr={}),
        SimpleNamespace(
            reuse_boxes_from=None,
            model_cache=tmp_path,
            device="cpu",
            allow_download=False,
        ),
        source,
    )
    assert len(detect(rgb(source), source)) == 1 and len(weights) == 1
