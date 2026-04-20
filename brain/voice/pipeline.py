"""
Full-duplex voice pipeline with barge-in.

Orchestrates VAD → STT → (handler) → TTS. Barge-in means: if the user
starts talking while the brain is speaking, we cut TTS playback instantly
and start listening again — like a phone call, not a walkie-talkie.

Usage:
    pipe = VoicePipeline(on_utterance=handle_user_said)
    pipe.start()  # begins listening
    pipe.feed_audio(pcm16_chunk)  # streaming input
    pipe.stop()

`on_utterance` is called with the finalized user transcript once VAD
detects end-of-speech. Whatever it returns is spoken back via TTS.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from brain.voice.vad import VAD, VADState, VADResult
from brain.voice.stt import STT, STTResult
from brain.voice.tts import TTS, TTSResult


class PipelineState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    BARGED_IN = "barged_in"  # user interrupted while we were speaking


@dataclass
class UtteranceEvent:
    text: str
    start_ts: float
    end_ts: float
    duration_sec: float
    vad_state: str
    barge_in: bool = False


@dataclass
class PipelineStats:
    utterances: int = 0
    barge_ins: int = 0
    total_listen_sec: float = 0.0
    total_speak_sec: float = 0.0
    avg_response_ms: float = 0.0
    rejected_frames: int = 0   # frames that hit VAD but got filtered (e.g. desk taps)

    def to_dict(self) -> dict:
        return {
            "utterances": self.utterances,
            "barge_ins": self.barge_ins,
            "total_listen_sec": round(self.total_listen_sec, 2),
            "total_speak_sec": round(self.total_speak_sec, 2),
            "avg_response_ms": round(self.avg_response_ms, 2),
            "rejected_frames": self.rejected_frames,
        }


class VoicePipeline:
    """Full-duplex orchestrator. Thread-safe."""

    def __init__(
        self,
        on_utterance: Optional[Callable[[str], str]] = None,
        vad_prefer: Optional[str] = None,
        stt_prefer: Optional[str] = None,
        tts_prefer: Optional[str] = None,
        sample_rate: int = 16000,
        min_utterance_sec: float = 0.3,      # ignore utterances shorter than this (desk tap protection)
        silence_trailing_sec: float = 0.8,   # after N sec silence, finalize utterance
        tts_piper_model: Optional[str] = None,
        tts_elevenlabs_voice_id: Optional[str] = None,
    ):
        self.vad = VAD(prefer=vad_prefer)
        self.stt = STT(prefer=stt_prefer)
        self.tts = TTS(
            prefer=tts_prefer,
            piper_model=tts_piper_model,
            elevenlabs_voice_id=tts_elevenlabs_voice_id,
        )
        self.sample_rate = sample_rate
        self.min_utterance_sec = min_utterance_sec
        self.silence_trailing_sec = silence_trailing_sec
        self.on_utterance = on_utterance

        self.state = PipelineState.IDLE
        self.stats = PipelineStats()

        # Thread-safe buffer for incoming audio during utterance.
        # We track duration via sample count (audio-time) so that the
        # pipeline works correctly whether fed in real-time from a mic
        # or synchronously from a test harness.
        self._buffer = bytearray()
        self._utterance_samples: int = 0      # samples since utterance started
        self._silence_samples: int = 0        # consecutive silence samples
        # RLock (reentrant) so _finalize_utterance can call self.speak()
        # without deadlocking when the user's handler triggers TTS.
        self._lock = threading.RLock()
        self._on_audio_out: Optional[Callable[[bytes, int], None]] = None
        self._speaking_cancel = threading.Event()
        self._utterances: list = []

    # ---------- Public API ----------

    def start(self):
        """Move pipeline from IDLE -> LISTENING."""
        with self._lock:
            self.state = PipelineState.LISTENING
            self._buffer.clear()
            self._utterance_samples = 0
            self._silence_samples = 0

    def stop(self):
        """Halt everything."""
        with self._lock:
            self._speaking_cancel.set()
            self.state = PipelineState.IDLE

    def feed_audio(self, pcm16: bytes) -> Optional[UtteranceEvent]:
        """
        Feed a chunk of input audio. Returns UtteranceEvent if this chunk
        completes an utterance (i.e. user stopped talking).

        This is the heart of the full-duplex loop. Must be called with
        small chunks (~20ms = 640 bytes at 16kHz) for responsive VAD.

        Uses audio-time (sample count) rather than wall-clock for silence
        measurement, so tests can feed audio synchronously and behavior
        matches real-time mic input exactly.
        """
        if self.state == PipelineState.IDLE:
            return None

        vad_result = self.vad.feed(pcm16)
        chunk_samples = len(pcm16) // 2  # 16-bit mono
        silence_trailing_samples = int(self.silence_trailing_sec * self.sample_rate)
        min_utterance_samples = int(self.min_utterance_sec * self.sample_rate)
        event = None

        with self._lock:
            # Barge-in: if we're speaking and user starts talking, interrupt
            if self.state == PipelineState.SPEAKING and vad_result.state == VADState.SPEECH:
                self._speaking_cancel.set()
                self.state = PipelineState.BARGED_IN
                self.stats.barge_ins += 1

            if vad_result.state == VADState.SPEECH:
                # Reset silence counter; accumulate utterance samples
                if self._utterance_samples == 0:
                    self.state = PipelineState.LISTENING
                self._utterance_samples += chunk_samples
                self._silence_samples = 0
                self._buffer.extend(pcm16)
            else:
                # Silence frame
                if self._utterance_samples > 0:
                    self._buffer.extend(pcm16)
                    self._silence_samples += chunk_samples
                    if self._silence_samples >= silence_trailing_samples:
                        # Finalize utterance
                        speech_samples = self._utterance_samples
                        speech_duration = speech_samples / self.sample_rate
                        if speech_samples < min_utterance_samples:
                            # Too short — probably a desk tap or cough. Reject.
                            self.stats.rejected_frames += 1
                            self._buffer.clear()
                            self._utterance_samples = 0
                            self._silence_samples = 0
                        else:
                            barge = self.state == PipelineState.BARGED_IN
                            audio = bytes(self._buffer)
                            self._buffer.clear()
                            self._utterance_samples = 0
                            self._silence_samples = 0
                            # end_ts is best-effort wall clock; duration is sample-accurate
                            event = self._finalize_utterance(audio, speech_duration, time.time(), barge)

        return event

    def speak(self, text: str) -> TTSResult:
        """Synthesize and return audio. State tracked for barge-in."""
        with self._lock:
            self.state = PipelineState.SPEAKING
            self._speaking_cancel.clear()
        t0 = time.time()
        result = self.tts.speak(text)
        self.stats.total_speak_sec += result.duration_sec
        # Hand off to audio-out callback if registered
        if self._on_audio_out:
            try:
                self._on_audio_out(result.wav_bytes, result.sample_rate)
            except Exception as e:
                print(f"[VOICE] audio_out callback error: {e}")
        with self._lock:
            if self.state == PipelineState.SPEAKING:  # not barged-in
                self.state = PipelineState.LISTENING
        return result

    def set_audio_out(self, callback: Callable[[bytes, int], None]):
        """Register a callback that receives WAV bytes + sample rate to play."""
        self._on_audio_out = callback

    # ---------- Internal ----------

    def _finalize_utterance(
        self, audio: bytes, duration: float, end_ts: float, barge: bool
    ) -> UtteranceEvent:
        # STT
        self.state = PipelineState.THINKING
        t_stt = time.time()
        stt = self.stt.transcribe(audio, self.sample_rate)
        stt_ms = (time.time() - t_stt) * 1000
        self.stats.utterances += 1
        self.stats.total_listen_sec += duration

        # Update rolling avg response time
        prev_avg = self.stats.avg_response_ms
        n = self.stats.utterances
        self.stats.avg_response_ms = ((prev_avg * (n - 1)) + stt_ms) / n

        event = UtteranceEvent(
            text=stt.text,
            start_ts=end_ts - duration,
            end_ts=end_ts,
            duration_sec=duration,
            vad_state=self.vad.backend_name,
            barge_in=barge,
        )
        self._utterances.append(event)

        # Let user handler produce a reply
        if self.on_utterance:
            try:
                reply = self.on_utterance(stt.text)
                if reply:
                    self.speak(reply)
            except Exception as e:
                print(f"[VOICE] on_utterance handler error: {e}")

        self.state = PipelineState.LISTENING
        return event

    def recent_utterances(self, limit: int = 20) -> list:
        return [
            {
                "text": u.text,
                "start_ts": u.start_ts,
                "end_ts": u.end_ts,
                "duration_sec": round(u.duration_sec, 2),
                "vad_state": u.vad_state,
                "barge_in": u.barge_in,
            }
            for u in self._utterances[-limit:]
        ]

    def status(self) -> dict:
        return {
            "state": self.state.value,
            "vad_backend": self.vad.backend_name,
            "stt_backend": self.stt.backend_name,
            "tts_backend": self.tts.backend_name,
            "sample_rate": self.sample_rate,
            "stats": self.stats.to_dict(),
            "recent_utterances": self.recent_utterances(5),
        }
