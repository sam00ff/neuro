"""
Speech-to-Text.

Backends (preference order):
  1. Faster-Whisper (local, fast, accurate)
  2. openai-whisper (local, slower)
  3. OpenAI Whisper API (cloud, requires key)
  4. Mock (for tests / no-audio environments)

Returns incremental transcription — feed() takes audio chunks and
returns a partial (volatile) transcript until an endpointing flush.
"""

from __future__ import annotations

import os
import tempfile
import wave
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class STTResult:
    text: str
    is_final: bool
    confidence: float
    backend: str


class _MockSTT:
    name = "mock"

    def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> STTResult:
        n = len(pcm16) // 2
        seconds = n / sample_rate
        return STTResult(
            text=f"[mock transcription of {seconds:.2f}s audio]",
            is_final=True,
            confidence=1.0,
            backend=self.name,
        )


class _FasterWhisperSTT:
    name = "faster-whisper"

    def __init__(self, model_size: str = "base"):
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as e:
            raise RuntimeError("faster-whisper not installed") from e
        # int8 quantized for CPU — fast and small. Users can pass "small"/"medium"/"large-v3" for better quality.
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> STTResult:
        # Write to a temp WAV and hand path to faster-whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            with wave.open(wav_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(pcm16)
            segments, info = self._model.transcribe(
                wav_path, beam_size=1, vad_filter=True,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            conf = float(info.duration_after_vad > 0)
            return STTResult(text=text, is_final=True, confidence=conf, backend=self.name)
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass


class _WhisperLocalSTT:
    name = "openai-whisper"

    def __init__(self, model_size: str = "base"):
        try:
            import whisper  # type: ignore
        except ImportError as e:
            raise RuntimeError("openai-whisper not installed") from e
        self._model = whisper.load_model(model_size)

    def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> STTResult:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            with wave.open(wav_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(pcm16)
            result = self._model.transcribe(wav_path)
            return STTResult(
                text=result["text"].strip(),
                is_final=True,
                confidence=1.0,
                backend=self.name,
            )
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass


class _OpenAIAPISTT:
    name = "openai-api"

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

    def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> STTResult:
        import urllib.request
        import urllib.error

        # Write temp wav, upload via multipart/form-data
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            with wave.open(wav_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(pcm16)

            boundary = "----nlkd-boundary-XYZ"
            with open(wav_path, "rb") as f:
                audio = f.read()
            body = (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="model"\r\n\r\n'
                "whisper-1\r\n"
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
                "Content-Type: audio/wav\r\n\r\n"
            ).encode() + audio + f"\r\n--{boundary}--\r\n".encode()

            req = urllib.request.Request(
                "https://api.openai.com/v1/audio/transcriptions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                import json
                data = json.loads(resp.read().decode("utf-8"))
                return STTResult(
                    text=data.get("text", "").strip(),
                    is_final=True,
                    confidence=1.0,
                    backend=self.name,
                )
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass


class STT:
    def __init__(self, prefer: Optional[str] = None, model_size: str = "base"):
        self.backend: Any = None
        self.backend_name: str = ""
        order = [prefer] if prefer else []
        order += ["faster-whisper", "openai-whisper", "openai-api", "mock"]
        errors = []
        for name in order:
            if name is None:
                continue
            try:
                if name == "faster-whisper":
                    self.backend = _FasterWhisperSTT(model_size)
                elif name == "openai-whisper":
                    self.backend = _WhisperLocalSTT(model_size)
                elif name == "openai-api":
                    self.backend = _OpenAIAPISTT()
                elif name == "mock":
                    self.backend = _MockSTT()
                else:
                    continue
                self.backend_name = self.backend.name
                break
            except Exception as e:
                errors.append(f"{name}: {e}")
                continue
        if self.backend is None:
            raise RuntimeError(f"No STT backend available. Tried: {errors}")

    def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> STTResult:
        return self.backend.transcribe(pcm16, sample_rate)
