from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from lyra_orchestrator.models import ArtifactManifest, MusicBrief, default_structure


def test_default_structure_bars() -> None:
    structure = default_structure()
    assert sum(s.bars for s in structure) == 56


def test_artifact_manifest_requires_absolute_paths() -> None:
    with pytest.raises(ValidationError):
        ArtifactManifest(
            run_id="x",
            stems={"drums": "relative/drums.wav", "bass": "C:/abs/bass.wav", "pads": "C:/abs/pads.wav"},
            mix="C:/abs/mix.wav",
            reports={"music_brief": "C:/abs/music_brief.json"},
            created_at=datetime.now(timezone.utc),
        )


def test_music_brief_validation() -> None:
    brief = MusicBrief.model_validate(
        {
            "creative_direction": {"genre": "house", "mood": ["uplifting"], "references": [], "prohibited_elements": [], "similarity_constraints": []},
            "global_audio": {"bpm": 124, "meter": "4/4", "key": "A minor", "sample_rate": 48000, "bit_depth": 24, "channels": 2, "loudness_target_lufs": -16},
            "structure": [s.model_dump() for s in default_structure()],
            "stems": {"inventory": ["drums", "bass", "pads"], "per_stem_constraints": {}, "section_on_off": {}},
            "deliverables": {"full_mix": True, "stems": True, "midi": False, "instrumental": True, "alt_edits": False},
            "uncertainty_flags": [],
        }
    )
    assert brief.global_audio.bpm == 124
