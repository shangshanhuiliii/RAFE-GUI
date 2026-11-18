"""Small float32 CPU parity check against the preserved pairwise equations; not a dataset run."""

import gc
import json

import torch

from rafe_gui.config import load_config
from rafe_gui.metrics.encoders import Encoder, verify_model
from rafe_gui.storage import read, write
from rafe_gui.transforms.preprocessing import prepare_image, specification

config = load_config()
torch.set_num_threads(4)
task = next(
    t
    for t in read(config.path("dataset") / "eval_pairs.jsonl")
    if t["status"] == "ready"
)
paths = [config.path("dataset") / task[s]["path"] for s in ("real", "comparison")]
lock = json.loads(config.path("models").read_text(encoding="utf-8"))
reports = []
for name in ("siglip", "dino"):
    print("Validating one pair:", name, flush=True)
    directory = verify_model(config.root, lock[name])
    encoder = Encoder(name, directory, "cpu")
    spec = specification("square-pad")
    cached = [encoder.feature(path, spec)[0] for path in paths]
    # Independent original pairwise forward expression, using the same checkpoint processor.
    images = [prepare_image(path, spec)[0] for path in paths]
    options = {"do_resize": False}
    if name == "dino":
        options["do_center_crop"] = False
    inputs = encoder.processor(images=images, return_tensors="pt", **options)
    with torch.no_grad():
        output = (
            encoder.model(**inputs, interpolate_pos_encoding=True)
            if name == "siglip"
            else encoder.model(**inputs)
        )
        features = getattr(output, "pooler_output", None) if name == "siglip" else None
        if features is None:
            features = output.last_hidden_state[:, 0]
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        pair = float((features[0] @ features[1]).item())
    single = float(cached[0] @ cached[1])
    delta = abs(single - pair)
    if delta > 2e-5:
        raise ValueError("Parity tolerance exceeded")
    reports.append(
        {
            "model": name,
            "device": "cpu",
            "dtype": "float32",
            "pairs": 1,
            "single_image_cosine": single,
            "original_pairwise_cosine": pair,
            "absolute_difference": delta,
            "tolerance": 2e-5,
        }
    )
    print(reports[-1], flush=True)
    del encoder, output, features, inputs
    gc.collect()
write(config.root / "runs/validation/model_parity.json", reports)
