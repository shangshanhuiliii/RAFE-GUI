"""Detection adapters; all models and downloads remain in the experiment cache."""

import numpy as np

from ..storage import read, rgb, sha, unique


def _builtin_create(config, args, sample_path):
    if getattr(args, "reuse_boxes_from", None):
        source = args.reuse_boxes_from.resolve()
        rows = unique(read(source / "images.jsonl"), "source_path")

        def detect(pixels, path):
            matches = [r for r in rows.values() if r["source_sha256"] == sha(path)]
            if len(matches) != 1 or matches[0]["status"] not in (
                "ok",
                "no_text_detected",
            ):
                raise ValueError("Cached detection missing, ambiguous, stale or failed")
            return matches[0]["boxes"]

        return detect, dict(
            reused_index_sha256=sha(source / "images.jsonl"),
            reused_recipe_sha256=sha(source / "mask.run.json"),
        )
    backend = config.get("backend", "easyocr")
    cache = args.model_cache.resolve() / backend
    cache.mkdir(parents=True, exist_ok=True)
    if backend == "easyocr":
        import easyocr

        # Backward-compatible location for the original CRAFT cache.
        if not list(cache.glob("*.pth")) and list(args.model_cache.glob("*.pth")):
            cache = args.model_cache.resolve()
        reader = easyocr.Reader(
            config["languages"],
            gpu=False if args.device == "cpu" else args.device,
            model_storage_directory=str(cache),
            download_enabled=args.allow_download,
            detect_network=config["detector"],
            recognizer=False,
            verbose=False,
        )

        def detect(pixels, path):
            horizontal, free = reader.detect(pixels, **config["detect"])
            boxes = [
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                for x0, x1, y0, y1 in horizontal[0]
            ]
            boxes.extend(free[0])
            return np.asarray(boxes).tolist() if boxes else []

        weights = {p.name: sha(p) for p in sorted(cache.glob("*.pth"))}
    elif backend == "rapidocr":
        if args.device != "cpu":
            raise ValueError(
                "RapidOCR adapter currently fixes ONNX CPU; use --device cpu"
            )
        from rapidocr import EngineType, LangDet, ModelType, OCRVersion, RapidOCR

        models = list(cache.rglob("*v5*det*server*.onnx")) + list(
            cache.rglob("*v5*server*det*.onnx")
        )
        if not args.allow_download and not models:
            raise ValueError(
                "PP-OCRv5 server detection weights missing; first use --allow-download"
            )
        params = {
            "Global.model_root_dir": str(cache),
            "Global.use_det": True,
            "Global.use_cls": False,
            "Global.use_rec": False,
            "Global.max_side_len": 4000,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType.SERVER,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            # RapidOCR 3.10 defaults target newer models; v5 requires its own normalization.
            "Det.mean": [0.485, 0.456, 0.406],
            "Det.std": [0.229, 0.224, 0.225],
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
        params.update(config["rapidocr"])
        if models:
            params["Det.model_path"] = str(sorted(set(models))[0])
        reader = RapidOCR(params=params)

        def detect(pixels, path):
            result = reader(
                np.ascontiguousarray(pixels[:, :, ::-1]),
                use_det=True,
                use_cls=False,
                use_rec=False,
            )
            return np.asarray(result.boxes).tolist() if result.boxes is not None else []

        # Versions with lazy loading only download detection weights at first inference.
        first_boxes = detect(rgb(sample_path), sample_path)
        first_hash = sha(sample_path)
        underlying = detect

        def detect(pixels, path):
            return first_boxes if sha(path) == first_hash else underlying(pixels, path)

        weights = {
            str(p.relative_to(cache)): sha(p) for p in sorted(cache.rglob("*.onnx"))
        }
    else:
        raise ValueError(f"Unsupported OCR backend: {backend}")
    if not weights:
        raise ValueError("No detector weights found to fingerprint")
    return detect, weights


DETECTORS = {"easyocr": _builtin_create, "rapidocr": _builtin_create}


def create(config, args, sample_path):
    name = config.get("backend", "easyocr")
    if name not in DETECTORS:
        raise ValueError(f"Unknown detector: {name}")
    return DETECTORS[name](config, args, sample_path)
