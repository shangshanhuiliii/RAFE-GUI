"""Pinned local encoders. Each worker owns its model; no process-global backend."""

import hashlib
from pathlib import Path

from ..storage import sha
from ..transforms.preprocessing import prepare_image


class Encoder:
    def __init__(self, name, directory, device="cuda", model=None, processor=None):
        if name not in ENCODERS:
            raise ValueError(f"Unknown encoder: {name}")
        self.name = name
        self.device = device
        if model is None:
            from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel

            loader = SiglipVisionModel if name == "siglip" else AutoModel
            model = (
                loader.from_pretrained(str(directory), local_files_only=True)
                .to(device)
                .eval()
            )
            processor = AutoImageProcessor.from_pretrained(
                str(directory), local_files_only=True, use_fast=False
            )
        self.model = model
        self.processor = processor

    def features(self, paths, spec):
        import torch
        from PIL import Image

        if spec["mode"] == "baseline":
            prepared = []
            for path in paths:
                with Image.open(path) as im:
                    prepared.append(
                        (im.convert("RGB"), {"source_size_wh": list(im.size)})
                    )
            options = {}
        else:
            prepared = [prepare_image(Path(p), spec) for p in paths]
            options = {"do_resize": False}
            if self.name == "dino":
                options["do_center_crop"] = False
        inputs = self.processor(
            images=[p[0] for p in prepared], return_tensors="pt", **options
        )
        tensor = inputs["pixel_values"]
        if spec["mode"] != "baseline":
            w, h = spec["output_size_wh"]
            if tuple(tensor.shape) != (len(paths), 3, h, w):
                raise ValueError("Unexpected model input geometry")
        geometry = [
            dict(p[1], tensor_sha256=hashlib.sha256(t.numpy().tobytes()).hexdigest())
            for p, t in zip(prepared, tensor)
        ]
        inputs = inputs.to(self.device)
        kwargs = (
            {"interpolate_pos_encoding": True}
            if self.name == "siglip" and spec["mode"] != "baseline"
            else {}
        )
        with torch.no_grad():
            output = self.model(**inputs, **kwargs)
            feature = (
                getattr(output, "pooler_output", None)
                if self.name == "siglip"
                else None
            )
            if feature is None:
                feature = output.last_hidden_state[:, 0]
            if not torch.isfinite(feature).all():
                raise ValueError("Nonfinite model feature")
            feature = feature / feature.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return feature.cpu().numpy(), geometry

    def feature(self, path, spec):
        values, geometry = self.features([path], spec)
        return values[0], geometry[0]

    def cosine(self, a, b, spec):
        values, _ = self.features([a, b], spec)
        return float(values[0] @ values[1])


ENCODERS = {"siglip": Encoder, "dino": Encoder}


def verify_model(root, record, full=True):
    directory = (root / record["directory"]).resolve()
    for name, expected in record["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Missing model file: {path}")
        if full and sha(path) != expected["sha256"]:
            raise ValueError(f"Changed model weight: {path}")
    return directory
