"""GUI-CC visual judge protocol; optional network client injected explicitly."""

from __future__ import annotations

import base64
import io
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image

DEFAULT_MAX_TOKENS = 4096
REQUEST_TIMEOUT = 600.0
VLM_RETRIES = 3
IMAGE_ENCODING = {"step": {"longest": 1280, "image_format": "JPEG", "jpeg_quality": 88}}
_JSON_RE = re.compile(r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}", re.DOTALL)


def prompt(name):
    return (Path(__file__).resolve().parents[1] / "prompts" / name).read_text(
        encoding="utf-8"
    )


def _parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, json.JSONDecodeError):
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    best: tuple[dict, int] | None = None
    for match in _JSON_RE.finditer(cleaned):
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and (best is None or len(match.group(0)) > best[1]):
            best = (parsed, len(match.group(0)))
    return best[0] if best else None


def _is_rate_limit_error(error: object) -> bool:
    text = str(error).lower()
    return any(token in text for token in ("429", "throttling", "rate limit", "限流"))


def image_data_url(
    path: Path, longest: int, image_format: str, jpeg_quality: int
) -> str:
    with Image.open(path) as image:
        converted = image.convert("RGB")
    width, height = converted.size
    if max(width, height) > longest:
        scale = longest / max(width, height)
        converted = converted.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    output_format = image_format.strip().upper().replace("JPG", "JPEG")
    if output_format == "PNG":
        converted.save(buffer, format="PNG")
        mime = "image/png"
    else:
        converted.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        mime = "image/jpeg"
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def step_image_url(path: Path) -> str:
    return image_data_url(path, **IMAGE_ENCODING["step"])


def call_vlm(
    client: Any,
    model: str,
    messages: list,
    retries: Optional[int] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    extra_body: dict | None = None,
) -> dict:
    """调用已配置的 judge 模型，绝不替换为其他模型。"""
    retries = VLM_RETRIES if retries is None else retries
    request_timeout = REQUEST_TIMEOUT
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=request_timeout,
                extra_body=extra_body or {},
            )
            raw = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            return {
                "raw": raw,
                "parsed": _parse_json(raw) or {},
                "requested_model": model,
                "api_model": getattr(response, "model", None),
                # token 用量进结果文件：小样本试跑时据此估全量的 API 开销。
                "usage": usage.model_dump(exclude_none=True)
                if usage is not None
                else None,
            }
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt < retries:
                time.sleep(
                    min(60, 5 * (attempt + 1))
                    if _is_rate_limit_error(error)
                    else min(30, 2**attempt)
                )
    return {
        "raw": "",
        "parsed": {},
        "error": str(last_error)[:300],
        "requested_model": model,
        "api_model": None,
    }


def parsed_with_meta(result: dict) -> dict:
    parsed = dict(result.get("parsed") or {})
    if result.get("api_model"):
        parsed["_api_model"] = result["api_model"]
    if result.get("requested_model"):
        parsed["_requested_model"] = result["requested_model"]
    return parsed


def vlm_error_payload(
    result: dict, default_error: str = "json_parse_failed"
) -> Optional[dict]:
    if not result.get("error") and result.get("parsed"):
        return None
    return {
        "error": result.get("error") or default_error,
        "raw_response": result.get("raw", ""),
        "_api_model": result.get("api_model"),
        "_requested_model": result.get("requested_model"),
    }


class JudgeError(RuntimeError):
    """API or response error that makes an episode unscorable."""

    def __init__(self, metric: str, message: str):
        super().__init__(f"{metric}: {message}")
        self.metric = metric


def _number(parsed: dict[str, Any], key: str, low: float, high: float) -> float:
    value = parsed.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key}  must be a number")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{key}  must lie in [{low}, {high}]")
    return value


def _required_text(parsed: dict[str, Any], key: str) -> str:
    value = parsed.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key}  must be a non-empty string")
    return value.strip()


class VLMJudge:
    def __init__(
        self,
        client: Any,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        extra_body=None,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}

    def _request(self, metric: str, messages: list[dict]) -> dict:
        result = call_vlm(
            self.client,
            self.model,
            messages,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body,
        )
        error = vlm_error_payload(result)
        if error:
            raise JudgeError(metric, str(error.get("error", "invalid response")))
        # Some gateways do not echo the model name; only a present, mismatched echo is an error.
        if result.get("api_model") not in (None, self.model):
            raise JudgeError(
                metric,
                f"judge response model {result.get('api_model')!r} does not match the requested model {self.model!r}",
            )
        parsed = parsed_with_meta(result)
        parsed["api_model"] = parsed.pop("_api_model", None)
        parsed["requested_model"] = parsed.pop("_requested_model", self.model)
        parsed["raw_response"] = result.get("raw", "")
        parsed["usage"] = result.get("usage")
        if parsed.get("error"):
            raise JudgeError(metric, str(parsed["error"]))
        return parsed

    def s_ele_lay(self, reference: Path, prediction: Path) -> dict:
        messages = [
            {"role": "system", "content": prompt("s_ele_lay_system.md")},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Reference Image (Ground Truth, Image 1):",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": step_image_url(reference)},
                    },
                    {"type": "text", "text": "Candidate Image (Prediction, Image 2):"},
                    {
                        "type": "image_url",
                        "image_url": {"url": step_image_url(prediction)},
                    },
                    {"type": "text", "text": prompt("s_ele_lay_user.md")},
                ],
            },
        ]
        parsed = self._request("S_ele/S_lay", messages)
        element = _number(parsed, "element_alignment_score", 1.0, 10.0)
        layout = _number(parsed, "structural_fidelity_score", 1.0, 10.0)
        _required_text(parsed, "reasoning")
        return {
            "S_ele": (element - 1.0) / 9.0,
            "S_lay": (layout - 1.0) / 9.0,
            "raw_scores": {"S_ele": element, "S_lay": layout},
            "judge": parsed,
        }
