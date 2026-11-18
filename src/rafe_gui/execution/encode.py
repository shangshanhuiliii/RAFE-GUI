"""One model per worker, explicit inputs, portable content-based cache keys."""

import json
import os
import time
from pathlib import Path

import numpy as np

from ..config import versions
from ..data.experiment import load
from ..metrics.encoders import ENCODERS, verify_model
from ..storage import fingerprint, inside, read, sha, write, write_rows
from ..transforms.preprocessing import specification


def encode(config, work, model, mode, device, limit=None):
    label = f"[{mode} | {model}]"
    print(f"{label} Checking input files and model weights", flush=True)
    import torch

    work, root, images, tasks, exp = load(work)
    variants = {r["image_id"]: r for r in read(work / "variants.jsonl")}
    lock = json.loads(config.path("models").read_text(encoding="utf-8"))
    record = lock[model]
    directory = verify_model(config.root, record, full=True)
    width = 560 if mode == "stretch" else config.data.get("content_width", 252)
    spec = specification(mode, width, config.data.get("content_height", 560))
    cfg = dict(
        execution="unique_image_features_float32_v1",
        implementation="encoder-v1",
        metric=model,
        source_sha256={
            "encoder": sha(Path(__file__).resolve().parents[1] / "metrics/encoders.py"),
            "preprocessing": sha(
                Path(__file__).resolve().parents[1] / "transforms/preprocessing.py"
            ),
            "execution": sha(Path(__file__)),
        },
        models=record,
        encoder_preprocessing=spec,
        dtype="float32",
        device=device,
        torch_threads=4,
        render_run_sha256=sha(work / "render.run.json"),
        methods=config.data["variants"],
        packages=versions(
            (
                "torch",
                "torchvision",
                "transformers",
                "huggingface-hub",
                "numpy",
                "Pillow",
                "safetensors",
            )
        ),
        hardware=dict(
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            torch_cuda_runtime=torch.version.cuda,
            gpu_names=[
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            if device.startswith("cuda")
            else [],
        ),
    )
    if config.data.get("dtype", "float32") != "float32":
        raise ValueError("This protocol requires float32")
    key = fingerprint(cfg)
    folder = work / "features" / model
    if (folder / "run.json").exists() and json.loads(
        (folder / "run.json").read_text(encoding="utf-8")
    )["fingerprint"] != key:
        raise ValueError("Encoder protocol changed; choose a new run ID")
    rows = (
        {r["key"]: r for r in read(folder / "index.jsonl")}
        if (folder / "index.jsonl").exists()
        else {}
    )
    torch.set_num_threads(4)
    print(f"{label} Loading model", flush=True)
    encoder = ENCODERS[model](model, directory, device)
    write(folder / "run.json", dict(fingerprint=key, configuration=cfg))
    queue = [(i, m) for i in images for m in config.data["variants"]]
    selected = queue[:limit] if limit else queue
    print(f"{label} Encoding {len(selected)} images", flush=True)
    for n, (image, method) in enumerate(selected, 1):
        identifier = image["image_id"] + "__" + method
        artifact = variants[image["image_id"]]["artifacts"][method]
        path = inside(work, artifact["path"])
        if sha(path) != artifact["sha256"]:
            raise ValueError("Changed input image")
        old = rows.get(identifier)
        if old and old["status"] == "ok":
            if (
                old["protocol"] != key
                or old["image_sha256"] != artifact["sha256"]
                or sha(inside(work, old["feature_path"])) != old["feature_sha256"]
            ):
                raise ValueError("Stale cached feature")
            continue
        start = time.perf_counter()
        row = dict(
            key=identifier,
            image_id=image["image_id"],
            method=method,
            protocol=key,
            image_sha256=artifact["sha256"],
        )
        try:
            value, geometry = encoder.feature(path, spec)
            destination = folder / (identifier + ".npy")
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, value)
            row.update(
                status="ok",
                feature_path=destination.relative_to(work).as_posix(),
                feature_sha256=sha(destination),
                geometry=geometry,
            )
        except Exception as error:
            row.update(status="error", error=str(error))
        row["seconds"] = time.perf_counter() - start
        rows[identifier] = row
        write_rows(folder / "index.jsonl", rows.values())
        if n == 1 or n % 10 == 0 or n == len(selected) or row["status"] == "error":
            detail = f"error: {row['error']}" if row["status"] == "error" else "ok"
            print(
                f"{label} {n}/{len(selected)} {method} {detail} ({row['seconds']:.2f}s)",
                flush=True,
            )
    by_path = {i["path"]: i for i in images}
    for task in tasks["A"]:
        keys = [
            by_path[task[s]["path"]]["image_id"] + "__original"
            for s in ("real", "comparison")
        ]
        if all(rows.get(k, {}).get("status") == "ok" for k in keys):
            print(f"{label} Checking pairwise consistency", flush=True)
            values = [np.load(inside(work, rows[k]["feature_path"])) for k in keys]
            cached = float(values[0] @ values[1])
            pairwise = encoder.cosine(
                root / task["real"]["path"], root / task["comparison"]["path"], spec
            )
            delta = abs(cached - pairwise)
            if delta > 2e-5:
                raise ValueError(f"Original pairwise parity failed: {delta}")
            write(
                folder / "parity.json",
                dict(
                    task_id=task["task_id"],
                    absolute_difference=delta,
                    tolerance=2e-5,
                    cached_raw_cosine=cached,
                    pairwise_raw_cosine=pairwise,
                ),
            )
            break
    expected = {i["image_id"] + "__" + m for i, m in queue}
    successful = sum(rows.get(k, {}).get("status") == "ok" for k in expected)
    report = dict(model=model, mode=mode, successful=successful, expected=len(expected))
    write(folder / "completion.json", report)
    print(f"{label} Finished: {successful}/{len(expected)} features ready", flush=True)
    return successful == len(expected)
