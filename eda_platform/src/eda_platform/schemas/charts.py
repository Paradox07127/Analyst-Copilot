from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChartSpec(BaseModel):
    dataset_id: str
    title: str
    description: str = ""
    category: Literal[
        "quality",
        "distribution",
        "relationship",
        "time",
        "comparison",
    ] = "distribution"
    mark: Literal["bar", "line", "point", "area", "rect", "boxplot"]
    encoding: dict[str, Any]
    data: dict[str, Any] = Field(default_factory=dict)

    def to_vegalite(self) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
            "title": self.title,
            "mark": self.mark,
            "encoding": self.encoding,
        }
        if self.description:
            spec["description"] = self.description
        if self.data:
            spec["data"] = self.data
        return spec
