from types import SimpleNamespace

import numpy as np
from PIL import Image

from rafe_gui.analysis.summary import analyze
from rafe_gui.data.experiment import METHODS
from rafe_gui.storage import read, sha, write, write_rows


def test_ten_score_intersection_keeps_failed_pair_out_of_every_mean(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    images = []
    for name, env in [
        ("r1", "wechat_real_account1"),
        ("r2", "wechat_real_account2"),
        ("s", "wechat_sim_appsim"),
    ]:
        p = root / (name + ".png")
        Image.new("RGB", (10, 20), "white").save(p)
        images.append(
            dict(
                path=p.name,
                sha256=sha(p),
                image_id=name,
                environment_id=env,
                page_id=1,
                page_type="main",
            )
        )
    tasks = [
        dict(
            task_id="t" + str(i),
            app="wechat",
            page_id=1,
            real=images[i - 1],
            comparison=images[2],
            comparison_environment_id="wechat_sim_appsim",
            aggregation_group_id="g" + str(i),
            relation="direct",
        )
        for i in (1, 2)
    ]
    write_rows(work / "A.jsonl", tasks)
    write_rows(work / "B.jsonl", tasks[:1])
    write_rows(work / "sources.jsonl", images)
    write(
        work / "experiment.json",
        dict(
            dataset_root=str(root),
            source_files={},
            sources_sha256=sha(work / "sources.jsonl"),
            tasks_sha256={s: sha(work / f"{s}.jsonl") for s in ("A", "B")},
        ),
    )
    write_rows(
        work / "variants.jsonl",
        [
            dict(image_id=i["image_id"], status="ok", mask_ratio=0.1, details={})
            for i in images
        ],
    )
    write_rows(
        work / "detections.jsonl",
        [dict(image_id=i["image_id"], status="ok") for i in images],
    )
    for model in ("siglip", "dino"):
        directory = work / "features" / model
        directory.mkdir(parents=True)
        write(
            directory / "run.json",
            dict(
                fingerprint="test",
                configuration={"variants_sha256": sha(work / "variants.jsonl")},
            ),
        )
        write(directory / "parity.json", dict(absolute_difference=0, tolerance=2e-5))
        rows = []
        for i in images:
            for m in METHODS:
                if model == "dino" and i["image_id"] == "r2" and m == "texture":
                    continue
                key = i["image_id"] + "__" + m
                path = directory / (key + ".npy")
                np.save(path, np.array([1.0, 0.0], np.float32))
                rows.append(
                    dict(
                        key=key,
                        status="ok",
                        protocol="test",
                        feature_path=path.relative_to(work).as_posix(),
                        feature_sha256=sha(path),
                    )
                )
        write_rows(directory / "index.jsonl", rows)
    analyze(SimpleNamespace(work=work))
    import json

    summary = json.loads((work / "analysis/summary.json").read_text(encoding="utf-8"))
    assert summary["schemes"]["A"]["expected"] == 2
    assert summary["schemes"]["A"]["complete"] == 1
    assert summary["schemes"]["B"]["complete"] == 1
    assert all(
        v["n"] == 1
        for values in summary["schemes"]["A"]["global_scores"].values()
        for v in values.values()
    )
    output = read(work / "analysis/A/paired_scores.jsonl")
    assert (
        output[1]["scores"]["dino"]["texture"] is None
        and output[1]["complete"] is False
    )
