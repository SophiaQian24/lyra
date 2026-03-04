from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .models import (
    CreativeDirection,
    Deliverables,
    GlobalAudio,
    MusicBrief,
    StemContract,
    StemPlan,
    StemTechnicalSpec,
    RetryPolicy,
    default_structure,
)
from .prompts import (
    CONDUCTOR_SYSTEM_PROMPT,
    INTENT_SYSTEM_PROMPT,
    QC_JUDGE_SYSTEM_PROMPT,
    STEM_PROMPT_SYSTEM_PROMPT,
)
from .settings import Settings

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


class StemPayload(BaseModel):
    stem_name: str
    prompt: str
    tags: str
    negative_tags: str
    title: str


class QCJudgeDecision(BaseModel):
    passed: bool
    notes: str


@dataclass
class LLMClient:
    settings: Settings

    def __post_init__(self) -> None:
        self.client = None
        if self.settings.openai_api_key and OpenAI:
            self.client = OpenAI(api_key=self.settings.openai_api_key)

    def _extract_json_text(self, response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return text

        try:
            outputs = getattr(response, "output", [])
            if outputs and outputs[0].content:
                chunks: list[str] = []
                for c in outputs[0].content:
                    raw = getattr(c, "text", None)
                    if raw:
                        chunks.append(raw)
                if chunks:
                    return "".join(chunks)
        except Exception:
            pass

        raise ValueError("Unable to extract JSON from OpenAI response")

    def _structured(self, *, model: str, system_prompt: str, user_prompt: str, schema_model: type[BaseModel]) -> BaseModel:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        schema = schema_model.model_json_schema()
        response = self.client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_model.__name__,
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        data = json.loads(self._extract_json_text(response))
        return schema_model.model_validate(data)

    def _infer_bpm(self, prompt: str) -> int:
        hit = re.search(r"(?P<bpm>\d{2,3})\s*bpm", prompt.lower())
        if hit:
            bpm = int(hit.group("bpm"))
            return max(60, min(220, bpm))
        if any(k in prompt.lower() for k in ["trap", "drill", "house", "techno"]):
            return 140
        if any(k in prompt.lower() for k in ["ballad", "ambient", "chill", "lofi"]):
            return 90
        return 120

    def build_initial_brief(self, prompt: str, references: list[str], overrides: dict[str, Any]) -> MusicBrief:
        fallback = MusicBrief(
            creative_direction=CreativeDirection(
                genre=overrides.get("genre") or "electronic",
                mood=["focused", "cinematic"],
                references=references,
                prohibited_elements=["vocals", "artist imitation"],
                similarity_constraints=["avoid direct melody copying from references"],
            ),
            global_audio=GlobalAudio(
                bpm=overrides.get("bpm") or self._infer_bpm(prompt),
                meter=overrides.get("meter") or "4/4",
                key=overrides.get("key") or "C minor",
                sample_rate=48000,
                bit_depth=24,
                channels=2,
            ),
            structure=default_structure(),
            stems=StemPlan(
                inventory=["drums", "bass", "pads"],
                per_stem_constraints={
                    "drums": "punchy transient-focused groove",
                    "bass": "monophonic low-end anchor",
                    "pads": "sustained harmonic support",
                },
            ),
            deliverables=Deliverables(full_mix=True, stems=True, midi=False, instrumental=True, alt_edits=False),
            uncertainty_flags=["User prompt may not specify precise key/meter."] if "key" not in prompt.lower() else [],
        )

        if not self.client:
            return fallback

        user_prompt = json.dumps(
            {
                "user_prompt": prompt,
                "references": references,
                "overrides": overrides,
                "defaults": fallback.model_dump(),
            },
            ensure_ascii=True,
        )

        try:
            result = self._structured(
                model=self.settings.openai_model_intent,
                system_prompt=INTENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema_model=MusicBrief,
            )
            return MusicBrief.model_validate(result.model_dump())
        except Exception:
            return fallback

    def build_stem_contracts(self, brief: MusicBrief, total_duration_seconds: float) -> list[StemContract]:
        fallback_contracts = [
            StemContract(
                stem_name="drums",
                prompt_rules=[
                    "Design a punchy rhythmic backbone that stays consistent across sections.",
                    "Use section energy contour: intro low, chorus high, outro low.",
                    "Keep groove quantized to global BPM.",
                ],
                negative_rules=["no melodic lead", "no bass-focused sub drones", "avoid clipping"],
                technical_spec=StemTechnicalSpec(exact_duration_seconds=total_duration_seconds),
                retry_policy=RetryPolicy(max_retries=self.settings.max_run_retries),
            ),
            StemContract(
                stem_name="bass",
                prompt_rules=[
                    "Create a locked low-end line in key with simple motif continuity.",
                    "Prioritize root notes and occasional passing tones.",
                    "Strict tempo adherence and consistent note lengths.",
                ],
                negative_rules=["no heavy percussion", "no bright lead melody", "avoid mud around 200-350Hz"],
                technical_spec=StemTechnicalSpec(exact_duration_seconds=total_duration_seconds),
                retry_policy=RetryPolicy(max_retries=self.settings.max_run_retries),
            ),
            StemContract(
                stem_name="pads",
                prompt_rules=[
                    "Generate sustained harmonic texture supporting sections and transitions.",
                    "Keep spatial width moderate and stable.",
                    "Match harmonic center without introducing dominant lead melody.",
                ],
                negative_rules=["no dominant drums", "no excessive sub bass", "avoid abrupt rhythmic stabs"],
                technical_spec=StemTechnicalSpec(exact_duration_seconds=total_duration_seconds),
                retry_policy=RetryPolicy(max_retries=self.settings.max_run_retries),
            ),
        ]

        if not self.client:
            return fallback_contracts

        class ConductorOut(BaseModel):
            stem_contracts: list[StemContract]

        user_prompt = json.dumps({"music_brief": brief.model_dump(), "total_duration_seconds": total_duration_seconds}, ensure_ascii=True)
        try:
            result = self._structured(
                model=self.settings.openai_model_conductor,
                system_prompt=CONDUCTOR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema_model=ConductorOut,
            )
            contracts = ConductorOut.model_validate(result.model_dump()).stem_contracts
            if {c.stem_name for c in contracts} == {"drums", "bass", "pads"}:
                return contracts
        except Exception:
            pass
        return fallback_contracts

    def build_stem_payload(
        self,
        brief: MusicBrief,
        stem_contract: StemContract,
        retry_index: int,
    ) -> StemPayload:
        base_tags = f"{brief.creative_direction.genre}, instrumental, stage-1, {stem_contract.stem_name}"
        base_prompt = (
            f"Create ONLY the {stem_contract.stem_name} layer for a full song at {brief.global_audio.bpm} BPM, "
            f"meter {brief.global_audio.meter}, key {brief.global_audio.key}. "
            f"Keep section arc Intro8 Verse16 Chorus16 Bridge8 Outro8. "
            + " ".join(stem_contract.prompt_rules)
        )

        if retry_index > 0:
            base_prompt = f"{base_prompt} Retry {retry_index}: {stem_contract.retry_policy.tighten_prompt_suffix}"

        fallback = StemPayload(
            stem_name=stem_contract.stem_name,
            prompt=base_prompt,
            tags=base_tags,
            negative_tags=", ".join(stem_contract.negative_rules),
            title=f"{stem_contract.stem_name}_run",
        )

        if not self.client:
            return fallback

        user_payload = {
            "music_brief": brief.model_dump(),
            "stem_contract": stem_contract.model_dump(),
            "retry_index": retry_index,
            "fallback": fallback.model_dump(),
        }
        try:
            result = self._structured(
                model=self.settings.openai_model_stem,
                system_prompt=STEM_PROMPT_SYSTEM_PROMPT,
                user_prompt=json.dumps(user_payload, ensure_ascii=True),
                schema_model=StemPayload,
            )
            parsed = StemPayload.model_validate(result.model_dump())
            if parsed.stem_name != stem_contract.stem_name:
                return fallback
            return parsed
        except Exception:
            return fallback

    def qc_judge(self, qc_payload: dict[str, Any]) -> QCJudgeDecision:
        fallback = QCJudgeDecision(passed=qc_payload.get("numeric_passed", False), notes="Numeric QC decision used.")
        if not self.client:
            return fallback

        try:
            result = self._structured(
                model=self.settings.openai_model_qc_judge,
                system_prompt=QC_JUDGE_SYSTEM_PROMPT,
                user_prompt=json.dumps(qc_payload, ensure_ascii=True),
                schema_model=QCJudgeDecision,
            )
            return QCJudgeDecision.model_validate(result.model_dump())
        except Exception:
            return fallback
