from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal

from .audio import load_wav, wav_duration_seconds
from .models import QCReport, StemQCMetric


@dataclass
class QCThresholds:
    duration_tolerance_ms: float = 250.0
    max_peak_dbtp: float = -1.0
    max_silence_ratio: float = 0.7
    tempo_tolerance_bpm: float = 10.0


def _peak_db(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio)) + 1e-12)
    return 20.0 * math.log10(max(peak, 1e-9))


def _frame_rms(audio_mono: np.ndarray, frame_size: int, hop: int) -> np.ndarray:
    frames = []
    for idx in range(0, max(1, len(audio_mono) - frame_size), hop):
        frame = audio_mono[idx : idx + frame_size]
        rms = np.sqrt(np.mean(np.square(frame)) + 1e-12)
        frames.append(rms)
    if not frames:
        return np.array([0.0])
    return np.array(frames)


def _silence_ratio(audio_mono: np.ndarray, sr: int, threshold_db: float = -45.0) -> float:
    rms = _frame_rms(audio_mono, frame_size=int(sr * 0.05), hop=int(sr * 0.025))
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    return float(np.mean(db < threshold_db))


def _estimate_tempo(audio_mono: np.ndarray, sr: int) -> float:
    envelope = np.abs(signal.hilbert(audio_mono))
    envelope = signal.medfilt(envelope, kernel_size=31)
    env = envelope - np.mean(envelope)
    if np.allclose(env, 0):
        return 0.0

    corr = signal.correlate(env, env, mode="full")
    corr = corr[corr.size // 2 :]

    min_bpm, max_bpm = 60.0, 220.0
    min_lag = int(sr * 60.0 / max_bpm)
    max_lag = int(sr * 60.0 / min_bpm)
    if max_lag <= min_lag or len(corr) <= max_lag:
        return 0.0

    segment = corr[min_lag:max_lag]
    if len(segment) == 0:
        return 0.0
    lag = int(np.argmax(segment)) + min_lag
    return 60.0 * sr / max(lag, 1)


def _transient_density(audio_mono: np.ndarray, sr: int) -> float:
    diff = np.abs(np.diff(audio_mono))
    if diff.size == 0:
        return 0.0
    thr = np.mean(diff) + 2.5 * np.std(diff)
    peaks, _ = signal.find_peaks(diff, height=thr, distance=max(1, int(sr * 0.03)))
    duration = len(audio_mono) / sr
    return float(len(peaks)) / max(duration, 1e-3)


def _low_end_dominance(audio_mono: np.ndarray, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(audio_mono))
    freqs = np.fft.rfftfreq(len(audio_mono), 1.0 / sr)
    low = np.sum(spectrum[(freqs >= 20) & (freqs <= 200)])
    mid = np.sum(spectrum[(freqs > 200) & (freqs <= 2000)]) + 1e-9
    return float(low / mid)


def _sustained_ratio(audio_mono: np.ndarray, sr: int) -> float:
    density = _transient_density(audio_mono, sr)
    return float(1.0 / (1.0 + density))


def stem_sanity_pass(stem_name: str, audio_mono: np.ndarray, sr: int) -> tuple[bool, str]:
    if stem_name == "drums":
        density = _transient_density(audio_mono, sr)
        return density >= 8.0, f"drums_transient_density={density:.2f}"
    if stem_name == "bass":
        ratio = _low_end_dominance(audio_mono, sr)
        return ratio >= 0.8, f"bass_low_end_ratio={ratio:.2f}"
    if stem_name == "pads":
        ratio = _sustained_ratio(audio_mono, sr)
        return ratio >= 0.08, f"pads_sustained_ratio={ratio:.2f}"
    return False, "unknown stem"


def compute_stem_metric(
    stem_name: str,
    wav_path: str,
    target_duration_seconds: float,
    target_bpm: int,
    thresholds: QCThresholds,
) -> StemQCMetric:
    audio, sr = load_wav(wav_path)
    mono = np.mean(audio, axis=1)

    duration_sec = wav_duration_seconds(wav_path)
    duration_error_ms = abs(duration_sec - target_duration_seconds) * 1000.0
    duration_pass = duration_error_ms <= thresholds.duration_tolerance_ms

    clipping_dbtp = _peak_db(audio)
    clipping_pass = clipping_dbtp <= thresholds.max_peak_dbtp

    silence_ratio = _silence_ratio(mono, sr)
    silence_pass = silence_ratio <= thresholds.max_silence_ratio

    estimated_bpm = _estimate_tempo(mono, sr)
    tempo_pass = abs(estimated_bpm - target_bpm) <= thresholds.tempo_tolerance_bpm if estimated_bpm > 0 else False

    sanity_pass, sanity_note = stem_sanity_pass(stem_name, mono, sr)

    notes = [sanity_note]
    return StemQCMetric(
        stem_name=stem_name,  # type: ignore[arg-type]
        duration_error_ms=duration_error_ms,
        duration_pass=duration_pass,
        clipping_dbtp=clipping_dbtp,
        clipping_pass=clipping_pass,
        silence_ratio=silence_ratio,
        silence_pass=silence_pass,
        tempo_bpm_estimate=estimated_bpm,
        tempo_pass=tempo_pass,
        stem_sanity_pass=sanity_pass,
        notes=notes,
    )


def compute_qc_report(
    run_id: str,
    target_bpm: int,
    target_duration_seconds: float,
    stems: dict[str, str],
    duration_tolerance_ms: float,
    min_pass_ratio: float,
) -> QCReport:
    thresholds = QCThresholds(duration_tolerance_ms=duration_tolerance_ms)
    stem_metrics = [
        compute_stem_metric(stem_name, path, target_duration_seconds, target_bpm, thresholds)
        for stem_name, path in stems.items()
    ]

    passed_count = 0
    for metric in stem_metrics:
        all_pass = (
            metric.duration_pass
            and metric.clipping_pass
            and metric.silence_pass
            and metric.tempo_pass
            and metric.stem_sanity_pass
        )
        if all_pass:
            passed_count += 1

    pass_ratio = passed_count / max(len(stem_metrics), 1)
    passed = pass_ratio >= min_pass_ratio

    return QCReport(
        run_id=run_id,
        target_bpm=target_bpm,
        target_duration_seconds=target_duration_seconds,
        stem_metrics=stem_metrics,
        pass_ratio=pass_ratio,
        passed=passed,
    )


def is_borderline(report: QCReport) -> bool:
    if report.passed:
        return False
    for metric in report.stem_metrics:
        # borderline: exactly one metric fails and it is near threshold
        fails = [
            not metric.duration_pass,
            not metric.clipping_pass,
            not metric.silence_pass,
            not metric.tempo_pass,
            not metric.stem_sanity_pass,
        ]
        if sum(fails) == 1:
            if not metric.duration_pass and metric.duration_error_ms <= 350:
                return True
            if not metric.tempo_pass and abs(metric.tempo_bpm_estimate - report.target_bpm) <= 14:
                return True
            if not metric.silence_pass and metric.silence_ratio <= 0.78:
                return True
    return False
