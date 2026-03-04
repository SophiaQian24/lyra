from __future__ import annotations

from lyra_orchestrator.llm import LLMClient
from lyra_orchestrator.models import (
    CreativeDirection,
    Deliverables,
    GlobalAudio,
    MusicBrief,
    RetryPolicy,
    StemContract,
    StemPlan,
    StemTechnicalSpec,
    default_structure,
)
from lyra_orchestrator.settings import Settings


def test_retry_prompt_tightening() -> None:
    settings = Settings(OPENAI_API_KEY=None)
    llm = LLMClient(settings)

    brief = MusicBrief(
        creative_direction=CreativeDirection(genre="synthwave"),
        global_audio=GlobalAudio(bpm=120, meter="4/4", key="C minor"),
        structure=default_structure(),
        stems=StemPlan(),
        deliverables=Deliverables(),
    )
    contract = StemContract(
        stem_name="drums",
        prompt_rules=["Keep groove tight."],
        negative_rules=["no clipping"],
        technical_spec=StemTechnicalSpec(exact_duration_seconds=112.0),
        retry_policy=RetryPolicy(max_retries=2, tighten_prompt_suffix="be tighter"),
    )

    first = llm.build_stem_payload(brief, contract, retry_index=0)
    retry = llm.build_stem_payload(brief, contract, retry_index=1)

    assert "Retry" not in first.prompt
    assert "Retry 1" in retry.prompt
    assert "be tighter" in retry.prompt
