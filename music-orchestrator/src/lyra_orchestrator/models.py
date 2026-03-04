from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


class RunState(str, Enum):
    queued = "queued"
    planning = "planning"
    generating = "generating"
    qc = "qc"
    mixing = "mixing"
    done = "done"
    failed = "failed"


class SectionSpec(BaseModel):
    section_id: str
    name: str
    bars: int = Field(gt=0)
    energy: Literal["low", "medium", "high"] = "medium"
    notes: str = ""


class CreativeDirection(BaseModel):
    genre: str
    mood: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    prohibited_elements: list[str] = Field(default_factory=list)
    similarity_constraints: list[str] = Field(default_factory=list)


class GlobalAudio(BaseModel):
    bpm: int = Field(ge=60, le=220)
    meter: str = Field(default="4/4")
    key: str = Field(default="C minor")
    sample_rate: int = Field(default=48000)
    bit_depth: int = Field(default=24)
    channels: int = Field(default=2)
    loudness_target_lufs: float = Field(default=-16.0)


class StemPlan(BaseModel):
    inventory: list[str] = Field(default_factory=lambda: ["drums", "bass", "pads"])
    per_stem_constraints: dict[str, str] = Field(default_factory=dict)
    section_on_off: dict[str, dict[str, bool]] = Field(default_factory=dict)


class Deliverables(BaseModel):
    full_mix: bool = True
    stems: bool = True
    midi: bool = False
    instrumental: bool = True
    alt_edits: bool = False


class MusicBrief(BaseModel):
    creative_direction: CreativeDirection
    global_audio: GlobalAudio
    structure: list[SectionSpec]
    stems: StemPlan
    deliverables: Deliverables
    uncertainty_flags: list[str] = Field(default_factory=list)


class TimelineSection(BaseModel):
    section_id: str
    name: str
    bars: int = Field(gt=0)
    start_bar: int = Field(ge=1)
    end_bar: int = Field(ge=1)


class TimelineContract(BaseModel):
    bpm: int = Field(ge=60, le=220)
    meter: str = "4/4"
    sections: list[TimelineSection]
    total_bars: int = Field(gt=0)
    total_seconds: float = Field(gt=0)


class StemTechnicalSpec(BaseModel):
    sample_rate: int = 48000
    bit_depth: int = 24
    channels: int = 2
    align_start_zero: bool = True
    exact_duration_seconds: float = Field(gt=0)


class RetryPolicy(BaseModel):
    max_retries: int = 2
    tighten_prompt_suffix: str = "Tight timing, clean transients, avoid clutter and keep strict tempo alignment."


class StemContract(BaseModel):
    stem_name: Literal["drums", "bass", "pads"]
    prompt_rules: list[str]
    negative_rules: list[str] = Field(default_factory=list)
    technical_spec: StemTechnicalSpec
    retry_policy: RetryPolicy


class GenerationResult(BaseModel):
    stem_name: Literal["drums", "bass", "pads"]
    suno_ids: list[str] = Field(default_factory=list)
    audio_url: str
    local_raw_path: str
    normalized_wav_path: str
    duration_seconds: float
    status: Literal["complete", "streaming", "submitted", "error"]
    provider_payload: dict[str, Any] = Field(default_factory=dict)


class StemQCMetric(BaseModel):
    stem_name: Literal["drums", "bass", "pads"]
    duration_error_ms: float
    duration_pass: bool
    clipping_dbtp: float
    clipping_pass: bool
    silence_ratio: float
    silence_pass: bool
    tempo_bpm_estimate: float
    tempo_pass: bool
    stem_sanity_pass: bool
    notes: list[str] = Field(default_factory=list)


class QCReport(BaseModel):
    run_id: str
    target_bpm: int
    target_duration_seconds: float
    stem_metrics: list[StemQCMetric]
    pass_ratio: float
    passed: bool
    judge_notes: str | None = None


class ArtifactManifest(BaseModel):
    run_id: str
    stems: dict[str, str]
    mix: str
    reports: dict[str, str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("stems", "reports")
    @classmethod
    def validate_absolute_paths(cls, value: dict[str, str]) -> dict[str, str]:
        for path in value.values():
            if not Path(path).is_absolute():
                raise ValueError(f"Path must be absolute: {path}")
        return value

    @field_validator("mix")
    @classmethod
    def validate_mix_path(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("mix path must be absolute")
        return value


class RunCreateRequest(BaseModel):
    prompt: str = Field(min_length=3)
    references: list[str] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)


class RunCreateResponse(BaseModel):
    run_id: str
    status: RunState
    music_brief: MusicBrief


class RetryRequest(BaseModel):
    target: Literal["drums", "bass", "pads", "all"]
    reason: str = Field(default="manual retry")


class RetryResponse(BaseModel):
    run_id: str
    queued: bool
    target: Literal["drums", "bass", "pads", "all"]
    retry_count: int


class RunStatusResponse(BaseModel):
    run_id: str
    status: RunState
    created_at: str
    updated_at: str
    prompt: str
    music_brief: MusicBrief | None = None
    timeline_contract: TimelineContract | None = None
    stem_contracts: list[StemContract] = Field(default_factory=list)
    qc_report: QCReport | None = None
    artifact_manifest: ArtifactManifest | None = None
    error: str | None = None
    logs: list[dict[str, Any]] = Field(default_factory=list)


def default_structure() -> list[SectionSpec]:
    return [
        SectionSpec(section_id="intro", name="Intro", bars=8, energy="low"),
        SectionSpec(section_id="verse", name="Verse", bars=16, energy="medium"),
        SectionSpec(section_id="chorus", name="Chorus", bars=16, energy="high"),
        SectionSpec(section_id="bridge", name="Bridge", bars=8, energy="medium"),
        SectionSpec(section_id="outro", name="Outro", bars=8, energy="low"),
    ]
