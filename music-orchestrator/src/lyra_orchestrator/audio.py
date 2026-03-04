from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but not found in PATH")


def convert_to_wav_48k_24bit(input_path: str, output_path: str) -> None:
    ensure_ffmpeg()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ac",
        "2",
        "-ar",
        "48000",
        "-c:a",
        "pcm_s24le",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {proc.stderr}")


def _rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    return 20.0 * math.log10(max(rms, 1e-9))


def _peak_db(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio)) + 1e-12)
    return 20.0 * math.log10(max(peak, 1e-9))


def _apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    gain = 10 ** (gain_db / 20.0)
    return audio * gain


def _limit_peak(audio: np.ndarray, target_peak_db: float = -1.0) -> np.ndarray:
    peak_db = _peak_db(audio)
    if peak_db <= target_peak_db:
        return audio
    return _apply_gain_db(audio, target_peak_db - peak_db)


def ensure_target_duration(audio: np.ndarray, sr: int, target_seconds: float) -> np.ndarray:
    target_samples = int(round(target_seconds * sr))
    current = audio.shape[0]
    if current == target_samples:
        return audio
    if current > target_samples:
        return audio[:target_samples]
    pad = np.zeros((target_samples - current, audio.shape[1]), dtype=audio.dtype)
    return np.concatenate([audio, pad], axis=0)


def normalize_audio(audio: np.ndarray, target_rms_db: float = -16.0, peak_limit_db: float = -1.0) -> np.ndarray:
    current_rms_db = _rms_db(audio)
    normalized = _apply_gain_db(audio, target_rms_db - current_rms_db)
    return _limit_peak(normalized, target_peak_db=peak_limit_db)


def process_stem(
    raw_input_path: str,
    normalized_output_path: str,
    *,
    target_duration_seconds: float,
    target_rms_db: float,
) -> float:
    temp_wav = str(Path(normalized_output_path).with_suffix(".tmp.wav"))
    convert_to_wav_48k_24bit(raw_input_path, temp_wav)

    audio, sr = sf.read(temp_wav, always_2d=True)
    audio = ensure_target_duration(audio, sr, target_duration_seconds)
    audio = normalize_audio(audio, target_rms_db=target_rms_db, peak_limit_db=-1.0)

    out = Path(normalized_output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, sr, subtype="PCM_24")
    Path(temp_wav).unlink(missing_ok=True)
    return float(audio.shape[0]) / float(sr)


def load_wav(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True)
    return audio, sr


def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, sr, subtype="PCM_24")


def mix_stems(stem_paths: list[str], output_path: str, target_rms_db: float = -16.0) -> None:
    loaded = [load_wav(path) for path in stem_paths]
    sample_rates = {sr for _, sr in loaded}
    if len(sample_rates) != 1:
        raise RuntimeError("Stem sample rates do not match")
    sr = sample_rates.pop()

    max_len = max(audio.shape[0] for audio, _ in loaded)
    aligned = []
    for audio, _ in loaded:
        if audio.shape[0] < max_len:
            pad = np.zeros((max_len - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
            audio = np.concatenate([audio, pad], axis=0)
        aligned.append(audio)

    mix = np.sum(np.stack(aligned, axis=0), axis=0) / max(1, len(aligned))
    mix = normalize_audio(mix, target_rms_db=target_rms_db, peak_limit_db=-1.0)
    write_wav(output_path, mix, sr)


def wav_duration_seconds(path: str) -> float:
    audio, sr = load_wav(path)
    return float(audio.shape[0]) / float(sr)
