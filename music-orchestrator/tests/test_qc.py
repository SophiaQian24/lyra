from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from lyra_orchestrator.qc import compute_qc_report


def _write_tone(path: Path, *, sr: int, seconds: float, freq: float, amp: float = 0.2) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    audio = amp * np.sin(2 * np.pi * freq * t)
    stereo = np.stack([audio, audio], axis=1)
    sf.write(path, stereo, sr, subtype="PCM_24")


def test_qc_report_duration_and_clipping(tmp_path: Path) -> None:
    sr = 48000
    duration = 2.0
    stems = {}
    for name, freq in [("drums", 120.0), ("bass", 60.0), ("pads", 240.0)]:
        p = tmp_path / f"{name}.wav"
        _write_tone(p, sr=sr, seconds=duration, freq=freq)
        stems[name] = str(p)

    report = compute_qc_report(
        run_id="run1",
        target_bpm=120,
        target_duration_seconds=duration,
        stems=stems,
        duration_tolerance_ms=250,
        min_pass_ratio=0.0,
    )

    assert len(report.stem_metrics) == 3
    for metric in report.stem_metrics:
        assert metric.duration_pass
        assert metric.clipping_pass
