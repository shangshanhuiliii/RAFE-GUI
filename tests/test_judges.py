import base64
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from rafe_gui.data.groups import build_groups, validate_response
from rafe_gui.metrics.vlm import VLMJudge, _parse_json, prompt


def client_for(payload, requests):
    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
            ],
            model="mock-vision",
            usage=None,
        )

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def test_strict_request_preserves_prompts_encoding_and_normalization(tmp_path):
    a = tmp_path / "a.png"
    Image.new("RGB", (1260, 2800), "white").save(a)
    requests = []
    client = client_for(
        {
            "element_alignment_score": 7,
            "structural_fidelity_score": 8,
            "reasoning": "visible",
        },
        requests,
    )
    result = VLMJudge(client, "mock-vision").s_ele_lay(a, a)
    assert result["S_ele"] == 6 / 9 and result["S_lay"] == 7 / 9
    request = requests[0]
    assert (
        request["max_tokens"] == 4096
        and request["temperature"] == 0
        and request["timeout"] == 600
    )
    assert request["messages"][0]["content"] == prompt("s_ele_lay_system.md")
    content = request["messages"][1]["content"]
    assert content[0]["text"] == "Reference Image (Ground Truth, Image 1):"
    assert content[2]["text"] == "Candidate Image (Prediction, Image 2):"
    assert content[-1]["text"] == prompt("s_ele_lay_user.md")
    raw = base64.b64decode(content[1]["image_url"]["url"].split(",")[1])
    with Image.open(io.BytesIO(raw)) as image:
        assert image.format == "JPEG" and image.size == (576, 1280)


def test_group_deduplicates_shared_sim_but_keeps_missing_group():
    image = lambda page: {"path": str(page), "page_id": page}
    tasks = [
        {
            "app": "eleme",
            "real": image(p),
            "comparison": image(7),
            "real_environment_id": "r",
            "comparison_environment_id": "s",
            "aggregation_group_id": "shop",
            "page_id": p,
            "page_name": "shop",
            "task_id": str(p),
            "status": "ready",
            "metadata_sha256": "test",
        }
        for p in (7, 8)
    ]
    for task in tasks:
        task["real"]["environment_id"] = "r"
    group = build_groups(tasks)[0]
    assert len(group["real_views"]) == 2 and len(group["sim_views"]) == 1
    tasks.append(
        {
            **tasks[0],
            "task_id": "missing",
            "aggregation_group_id": "other",
            "comparison": None,
            "status": "missing_comparison",
        }
    )
    assert any(g["status"] == "missing_comparison" for g in build_groups(tasks))


@pytest.mark.parametrize("verdict", ("partial", "not_comparable", "uncertain"))
def test_unscorable_group_keeps_null_scores(verdict):
    group = {"real_views": [{}], "sim_views": [{}]}
    payload = dict(
        comparability=verdict,
        reasoning="visible",
        element_alignment_score=None,
        structural_fidelity_score=None,
        regions=[
            {
                "region": "main",
                "observation": "visible",
                "real_views": ["R1"],
                "sim_views": ["S1"],
                "assessment": "different",
            }
        ],
        data_variations=[],
        state_differences=[],
        uncertainties=[],
    )
    assert validate_response(payload, group)["metrics"] == {
        "S_ele_group": None,
        "S_lay_group": None,
    }
    payload["regions"][0]["real_views"] = ["S1"]
    with pytest.raises(ValueError):
        validate_response(payload, group)


def test_json_fences_and_thinking_match_original_parser():
    assert _parse_json('<think>internal</think>```json\n{"a":1}\n```') == {"a": 1}


def test_strict_executor_resumes_without_rebilling_and_rejects_model_change(
    tmp_path, monkeypatch
):
    from helpers import fixture

    from rafe_gui.config import Config
    from rafe_gui.execution.judge import run_judge
    from rafe_gui.storage import read, sha, write_rows

    work, root, images = fixture(tmp_path)
    tasks = read(work / "A.jsonl")
    for task in tasks:
        task.update(schema_version="1.0", metadata_sha256=sha(root / "metadata.json"))
    write_rows(root / "tasks.jsonl", tasks)
    config = Config(
        tmp_path,
        {
            "paths": {"dataset": "dataset"},
            "manifest": "tasks.jsonl",
            "vlm_protocol": "strict",
            "limit": 1,
        },
        tmp_path / "config.json",
    )
    requests = []
    client = client_for(
        {
            "element_alignment_score": 7,
            "structural_fidelity_score": 8,
            "reasoning": "visible",
        },
        requests,
    )
    monkeypatch.setenv("RAFE_JUDGE_MODEL", "mock-vision")
    output = tmp_path / "run"
    run_judge(config, output, client=client)
    run_judge(config, output, resume=True, client=client)
    assert len(requests) == 1 and len(read(output / "scores.jsonl")) == 1
    monkeypatch.setenv("RAFE_JUDGE_MODEL", "other-model")
    with pytest.raises(ValueError, match="protocol changed"):
        run_judge(config, output, resume=True, client=client)
    assert len(requests) == 1
