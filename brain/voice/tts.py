"""
Text-to-Speech.

Backends (preference order):
  1. Piper         — local, fast, open-source, decent quality
  2. pyttsx3       — system TTS (SAPI on Windows, NSSpeech on macOS, espeak on Linux)
  3. ElevenLabs    — highest quality, cloud, requires key
  4. Mock          — silent WAV for testing

Every backend exposes .speak(text) → bytes of WAV audio plus an optional
streaming generator synthesize_stream(text) for barge-in support.
"""

from __future__ import annotations

import io
import os
import struct
import wave
from dataclasses import dataclass
from typing import Any, Generator, Optional


@dataclass
class TTSResult:
    wav_bytes: bytes
    sample_rate: int
    backend: str
    duration_sec: float


class _MockTTS:
    name = "mock"

    def speak(self, text: str) -> TTSResult:
        # Generate 0.5s of silence per 20 characters as a stand-in
        sr = 16000
        seconds = max(0.3, len(text) / 40.0)
        n_samples = int(sr * seconds)
        pcm = b"\x00\x00" * n_samples
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm)
        return TTSResult(
            wav_bytes=buf.getvalue(),
            sample_rate=sr,
            backend=self.name,
            duration_sec=seconds,
        )


class _Pyttsx3TTS:
    name = "pyttsx3"

    def __init__(self):
        try:
            import pyttsx3  # type: ignore
        except ImportError as e:
            raise RuntimeError("pyttsx3 not installed") from e
        self._engine = pyttsx3.init()

    def speak(self, text: str) -> TTSResult:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        try:
            self._engine.save_to_file(text, path)
            self._engine.runAndWait()
            with open(path, "rb") as f:
                wav = f.read()
            with wave.open(io.BytesIO(wav), "rb") as w:
                sr = w.getframerate()
                duration = w.getnframes() / sr
            return TTSResult(wav_bytes=wav, sample_rate=sr, backend=self.name, duration_sec=duration)
        finally:
            try:
                os.remove(path)
            except Exception:
                pass


class _PiperTTS:
    name = "piper"

    def __init__(self, model_path: Optional[str] = None):
        try:
            from piper import PiperVoice  # type: ignore
        except ImportError as e:
            raise RuntimeError("piper-tts not installed") from e
        if model_path is None:
            # Look for a default model in brain_state/voice/piper/
            # Users should drop en_US-lessac-medium.onnx in that folder.
            raise RuntimeError("Piper voice model path not provided")
        if not os.path.exists(model_path):
            raise RuntimeError(f"Piper model not found: {model_path}")
        self._voice = PiperVoice.load(model_path)

    def speak(self, text: str) -> TTSResult:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self._voice.config.sample_rate)
            self._voice.synthesize(text, w)
        wav = buf.getvalue()
        sr = self._voice.config.sample_rate
        with wave.open(io.BytesIO(wav), "rb") as w:
            duration = w.getnframes() / sr
        return TTSResult(wav_bytes=wav, sample_rate=sr, backend=self.name, duration_sec=duration)


class _ElevenLabsTTS:
    name = "elevenlabs"

    def __init__(self, voice_id: Optional[str] = None):
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set")
        self.voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID") or "21m00Tcm4TlvDq8ikWAM"

    def speak(self, text: str) -> TTSResult:
        import urllib.request
        import json
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream?output_format=pcm_16000"
        body = json.dumps({
            "text": text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "audio/pcm",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            pcm = resp.read()
        # Wrap raw PCM in WAV header
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm)
        duration = len(pcm) / 2 / 16000
        return TTSResult(wav_bytes=buf.getvalue(), sample_rate=16000, backend=self.name, duration_sec=duration)


class TTS:
    def __init__(
        self,
        prefer: Optional[str] = None,
        piper_model: Optional[str] = None,
        elevenlabs_voice_id: Optional[str] = None,
    ):
        self.backend: Any = None
        self.backend_name: str = ""
        order = [prefer] if prefer else []
        order += ["piper", "pyttsx3", "elevenlabs", "mock"]
        errors = []
        for name in order:
            if name is None:
                continue
            try:
                if name == "piper":
                    self.backend = _PiperTTS(piper_model)
                elif name == "pyttsx3":
                    self.backend = _Pyttsx3TTS()
                elif name == "elevenlabs":
                    self.backend = _ElevenLabsTTS(elevenlabs_voice_id)
                elif name == "mock":
                    self.backend = _MockTTS()
                else:
                    continue
                self.backend_name = self.backend.name
                break
            except Exception as e:
                errors.append(f"{name}: {e}")
                continue
        if self.backend is None:
            raise RuntimeError(f"No TTS backend available. Tried: {errors}")

    def speak(self, text: str) -> TTSResult:
        return self.backend.speak(text)
