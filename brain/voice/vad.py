"""
Voice Activity Detection.

Backends, in preference order:
  1. Silero VAD  — best accuracy, handles desk taps / keyboard / AC hum
  2. WebRTC VAD  — fast C lib, decent accuracy
  3. Energy VAD  — stdlib-only fallback (always available)

The public API is the same across all backends: feed() raw 16-bit PCM
mono at 16 kHz, get back a VADState (SPEECH / SILENCE with confidence).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class VADState(str, Enum):
    SILENCE = "silence"
    SPEECH = "speech"


@dataclass
class VADResult:
    state: VADState
    confidence: float        # 0.0 - 1.0
    backend: str
    energy: float = 0.0      # RMS energy (useful for debugging)


class _EnergyVAD:
    """Stdlib-only VAD — RMS energy threshold with hysteresis."""
    name = "energy"

    def __init__(
        self,
        speech_threshold: float = 0.02,      # RMS normalized 0-1
        silence_threshold: float = 0.008,    # lower threshold to exit speech
        consecutive_frames: int = 2,         # require N in a row to flip state
    ):
        self.speech_threshold = speech_threshold
        self.silence_threshold = silence_threshold
        self.consecutive_frames = consecutive_frames
        self._state = VADState.SILENCE
        self._streak = 0

    def feed(self, pcm16: bytes) -> VADResult:
        # Compute RMS of int16 samples
        if len(pcm16) < 2:
            return VADResult(self._state, 0.0, self.name, 0.0)
        import struct
        n = len(pcm16) // 2
        samples = struct.unpack(f"<{n}h", pcm16[: n * 2])
        sum_sq = sum(s * s for s in samples)
        rms = math.sqrt(sum_sq / n) / 32768.0  # normalized 0..1

        # Hysteresis: different thresholds for entering vs leaving speech
        if self._state == VADState.SILENCE:
            if rms > self.speech_threshold:
                self._streak += 1
            else:
                self._streak = 0
            if self._streak >= self.consecutive_frames:
                self._state = VADState.SPEECH
                self._streak = 0
        else:  # SPEECH
            if rms < self.silence_threshold:
                self._streak += 1
            else:
                self._streak = 0
            if self._streak >= self.consecutive_frames:
                self._state = VADState.SILENCE
                self._streak = 0

        confidence = min(1.0, rms / max(self.speech_threshold, 1e-6))
        return VADResult(self._state, confidence, self.name, rms)


class _WebRTCVAD:
    """WebRTC VAD — aggressive desk-tap filter when used with energy gate."""
    name = "webrtc"

    def __init__(self, aggressiveness: int = 3, energy_floor: float = 0.005):
        try:
            import webrtcvad  # type: ignore
        except ImportError as e:
            raise RuntimeError("webrtcvad not installed") from e
        self._v = webrtcvad.Vad(aggressiveness)
        self.energy_floor = energy_floor

    def feed(self, pcm16: bytes) -> VADResult:
        # WebRTC VAD requires 10/20/30ms frames at 8/16/32/48 kHz.
        # We assume 16kHz 20ms frames → 320 samples → 640 bytes.
        # If not that shape, split into chunks.
        frame_bytes = 640
        if len(pcm16) < frame_bytes:
            return VADResult(VADState.SILENCE, 0.0, self.name, 0.0)

        import struct
        speech_hits = 0
        total = 0
        sum_sq = 0.0
        sample_count = 0
        for off in range(0, len(pcm16) - frame_bytes + 1, frame_bytes):
            frame = pcm16[off : off + frame_bytes]
            n = frame_bytes // 2
            samples = struct.unpack(f"<{n}h", frame)
            for s in samples:
                sum_sq += s * s
            sample_count += n
            if self._v.is_speech(frame, 16000):
                speech_hits += 1
            total += 1

        rms = math.sqrt(sum_sq / max(sample_count, 1)) / 32768.0
        ratio = speech_hits / max(total, 1)

        # Gate: reject if energy is below floor (filters out desk taps
        # that briefly spike WebRTC VAD but have no sustained energy)
        if rms < self.energy_floor:
            return VADResult(VADState.SILENCE, ratio, self.name, rms)

        state = VADState.SPEECH if ratio > 0.5 else VADState.SILENCE
        return VADResult(state, ratio, self.name, rms)


class _SileroVAD:
    """Silero VAD — best accuracy, model-based. Handles desk taps gracefully."""
    name = "silero"

    def __init__(self, threshold: float = 0.5):
        try:
            import torch  # type: ignore
        except ImportError as e:
            raise RuntimeError("torch not installed") from e
        # Load pre-trained model — downloaded + cached to torch hub on first use
        self._model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            onnx=False,
        )
        self.threshold = threshold
        self._torch = torch

    def feed(self, pcm16: bytes) -> VADResult:
        import struct
        n = len(pcm16) // 2
        samples = struct.unpack(f"<{n}h", pcm16[: n * 2])
        # Silero wants float32 [-1,1]
        floats = [s / 32768.0 for s in samples]
        tensor = self._torch.tensor(floats, dtype=self._torch.float32)
        # Silero wants 512 samples at 16kHz
        if tensor.shape[0] < 512:
            # pad
            padding = self._torch.zeros(512 - tensor.shape[0])
            tensor = self._torch.cat([tensor, padding])
        elif tensor.shape[0] > 512:
            tensor = tensor[:512]

        with self._torch.no_grad():
            conf = float(self._model(tensor, 16000).item())

        state = VADState.SPEECH if conf >= self.threshold else VADState.SILENCE
        sum_sq = sum(f * f for f in floats)
        rms = math.sqrt(sum_sq / max(len(floats), 1))
        return VADResult(state, conf, self.name, rms)


class VAD:
    """
    Auto-picks the best backend available. Use feed() in a loop.
    """

    def __init__(self, prefer: Optional[str] = None):
        self.backend: Any = None
        self.backend_name: str = ""

        order = [prefer] if prefer else []
        order += ["silero", "webrtc", "energy"]

        errors = []
        for name in order:
            if name is None:
                continue
            try:
                if name == "silero":
                    self.backend = _SileroVAD()
                elif name == "webrtc":
                    self.backend = _WebRTCVAD()
                elif name == "energy":
                    self.backend = _EnergyVAD()
                else:
                    continue
                self.backend_name = self.backend.name
                break
            except Exception as e:
                errors.append(f"{name}: {e}")
                continue

        if self.backend is None:
            raise RuntimeError(f"No VAD backend available. Tried: {errors}")

    def feed(self, pcm16: bytes) -> VADResult:
        return self.backend.feed(pcm16)

    @property
    def state(self) -> VADState:
        if hasattr(self.backend, "_state"):
            return self.backend._state
        return VADState.SILENCE
