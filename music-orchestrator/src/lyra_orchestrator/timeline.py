from __future__ import annotations

from .models import MusicBrief, TimelineContract, TimelineSection


def beats_per_bar(meter: str) -> int:
    num, _ = meter.split("/", 1)
    return int(num)


def bars_to_seconds(bars: int, bpm: int, meter: str = "4/4") -> float:
    return (bars * beats_per_bar(meter) * 60.0) / float(bpm)


def build_timeline_contract(brief: MusicBrief) -> TimelineContract:
    start_bar = 1
    sections: list[TimelineSection] = []
    total_bars = 0
    for item in brief.structure:
        end_bar = start_bar + item.bars - 1
        sections.append(
            TimelineSection(
                section_id=item.section_id,
                name=item.name,
                bars=item.bars,
                start_bar=start_bar,
                end_bar=end_bar,
            )
        )
        total_bars += item.bars
        start_bar = end_bar + 1

    total_seconds = bars_to_seconds(total_bars, brief.global_audio.bpm, brief.global_audio.meter)
    return TimelineContract(
        bpm=brief.global_audio.bpm,
        meter=brief.global_audio.meter,
        sections=sections,
        total_bars=total_bars,
        total_seconds=total_seconds,
    )
