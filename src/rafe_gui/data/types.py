"""Small contracts shared by data adapters, metrics and reports."""

from pathlib import Path
from typing import Literal, Protocol, TypedDict


class ImageRecord(TypedDict):
    image_id: str
    path: str
    sha256: str
    environment_id: str
    page_id: int
    page_name: str
    page_state: str
    page_type: str


class PairTask(TypedDict):
    task_id: str
    app: str
    page_id: int
    status: Literal["ready", "missing_comparison", "review_required"]
    real: ImageRecord
    comparison: ImageRecord | None
    aggregation_group_id: str


class FeatureEncoder(Protocol):
    def feature(self, path: Path, spec: dict): ...


class TextDetector(Protocol):
    def __call__(self, pixels, path: Path) -> list: ...


class LogicalPageGroup(TypedDict):
    group_id: str
    app: str
    real_views: list[ImageRecord]
    sim_views: list[ImageRecord]
    status: Literal["ready", "missing_comparison", "review_required"]


class MetricResult(TypedDict):
    metrics: dict[str, float | None]
    status: str
