from __future__ import annotations

from lyra_orchestrator.models import (
    CreativeDirection,
    Deliverables,
    GlobalAudio,
    MusicBrief,
    StemPlan,
    default_structure,
)
from lyra_orchestrator.timeline import bars_to_seconds, beats_per_bar, build_timeline_contract


def test_beats_per_bar() -> None:
    assert beats_per_bar("4/4") == 4
    assert beats_per_bar("3/4") == 3


def test_bars_to_seconds() -> None:
    # 56 bars at 120 BPM in 4/4 = 112s
    assert abs(bars_to_seconds(56, 120, "4/4") - 112.0) < 1e-6


def test_build_timeline_contract() -> None:
    brief = MusicBrief(
        creative_direction=CreativeDirection(genre="house"),
        global_audio=GlobalAudio(bpm=120, meter="4/4", key="C minor"),
        structure=default_structure(),
        stems=StemPlan(),
        deliverables=Deliverables(),
    )
    timeline = build_timeline_contract(brief)
    assert timeline.total_bars == 56
    assert abs(timeline.total_seconds - 112.0) < 1e-6
    assert timeline.sections[0].start_bar == 1
    assert timeline.sections[-1].end_bar == 56
