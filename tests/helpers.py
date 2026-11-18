from PIL import Image

from rafe_gui.data.experiment import METHODS
from rafe_gui.storage import sha, write, write_rows


def fixture(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    work = tmp_path / "prepared"
    work.mkdir()
    write(root / "metadata.json", {})
    images = []
    environments = {
        "eleme": (
            "eleme_real_native",
            "eleme_real_native_account2",
            "eleme_sim_appsim",
        ),
        "wechat": ("wechat_real_account1", "wechat_real_account2", "wechat_sim_appsim"),
    }
    for app, envs in environments.items():
        for env in envs:
            for page in (1, 7) if app == "eleme" else (1,):
                name = f"{env}_{page}"
                path = root / (name + ".png")
                Image.new("RGB", (10, 20), "white").save(path)
                images.append(
                    dict(
                        path=path.name,
                        sha256=sha(path),
                        image_id=name,
                        environment_id=env,
                        page_id=page,
                        page_name="test",
                        page_type="main",
                    )
                )
    tasks = []
    for app, envs in environments.items():
        for image in images:
            if image["environment_id"] not in envs[:2]:
                continue
            sim = next(
                i
                for i in images
                if i["environment_id"] == envs[2] and i["page_id"] == image["page_id"]
            )
            tasks.append(
                dict(
                    task_id=image["image_id"],
                    app=app,
                    page_id=image["page_id"],
                    status="ready",
                    real=image,
                    comparison=sim,
                    comparison_environment_id=sim["environment_id"],
                    aggregation_group_id=image["image_id"],
                    relation="direct",
                )
            )
    write_rows(work / "A.jsonl", tasks)
    write_rows(work / "B.jsonl", [t for t in tasks if t["page_id"] == 1])
    write_rows(work / "sources.jsonl", images)
    write(
        work / "experiment.json",
        dict(
            dataset_root=str(root),
            source_files={"metadata.json": sha(root / "metadata.json")},
            sources_sha256=sha(work / "sources.jsonl"),
            tasks_sha256={s: sha(work / f"{s}.jsonl") for s in ("A", "B")},
        ),
    )
    rows = []
    for image in images:
        path = work / "images" / image["image_id"] / "prepared.png"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (10, 20), "gray").save(path)
        artifact = dict(path=path.relative_to(work).as_posix(), sha256=sha(path))
        rows.append(
            dict(
                image_id=image["image_id"],
                status="ok",
                mask_ratio=0.1,
                details={},
                artifacts={m: artifact for m in METHODS},
            )
        )
    write_rows(work / "variants.jsonl", rows)
    write_rows(
        work / "detections.jsonl",
        [dict(image_id=i["image_id"], status="ok") for i in images],
    )
    write(work / "render.run.json", {})
    write(work / "detect.run.json", {})
    return work, root, images
