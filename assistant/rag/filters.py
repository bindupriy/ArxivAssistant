"""Per-question metadata filters shared by the CLI and streaming API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SectionType = Literal[
    "abstract", "header", "intro", "background", "related", "method",
    "experiments", "results", "discussion", "analysis", "conclusion",
    "limitations", "appendix", "body",
]


class MetadataFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_year: int | None = Field(default=None, ge=1800, le=2100)
    max_year: int | None = Field(default=None, ge=1800, le=2100)
    venue: str | None = Field(default=None, max_length=128)
    section_types: list[SectionType] = Field(default_factory=list, max_length=14)

    @field_validator("venue")
    @classmethod
    def strip_venue(cls, venue: str | None) -> str | None:
        return (venue.strip() or None) if venue is not None else None

    @model_validator(mode="after")
    def validate_years(self) -> "MetadataFilters":
        if self.min_year is not None and self.max_year is not None and self.min_year > self.max_year:
            raise ValueError("min_year must not exceed max_year")
        return self