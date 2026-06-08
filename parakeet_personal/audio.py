"""
Microphone capture + transcription.

Two backends:
  "api"   – sends audio to OpenAI Whisper API  (requires openai key, internet)
  "local" – runs faster-whisper on-device       (requires: pip install faster-whisper)

VAD logic: collect audio while RMS > threshold; fire transcription after
SILENCE_SECS of quiet.  Each transcription runs in its own daemon thread so
the capture loop never blocks.
"""

import io
import queue
import threading

import numpy as np

SAMPLE_RATE = 16_000
CHANNELS = 1
CHUNK_SECS = 0.4          # seconds per read block
SILENCE_THRESHOLD = 0.012  # RMS below this = silence
SILENCE_SECS = 1.6         # consecutive silence before firing
MIN_SPEECH_SECS = 0.4      # ignore clips shorter than this


class _AudioCapture:
    def __init__(self, on_transcription):
        self.on_transcription = on_transcription
        self._recording = False
        self._thread = None
        self._buf: list[np.ndarray] = []
        self._silence_chunks = 0
        self._silence_limit = int(SILENCE_SECS / CHUNK_SECS)
        self._min_chunks = int(MIN_SPEECH_SECS / CHUNK_SECS)

    def start(self):
        self._recording = True
        self._buf = []
        self._silence_chunks = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._recording = False

    def _loop(self):
        import sounddevice as sd
        chunk_samples = int(SAMPLE_RATE * CHUNK_SECS)
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="float32", blocksize=chunk_samples) as stream:
            while self._recording:
                data, _ = stream.read(chunk_samples)
                chunk = data.flatten()
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                if rms > SILENCE_THRESHOLD:
                    self._buf.append(chunk)
                    self._silence_chunks = 0
                elif self._buf:
                    self._silence_chunks += 1
                    self._buf.append(chunk)
                    if self._silence_chunks >= self._silence_limit:
                        if len(self._buf) >= self._min_chunks:
                            audio = np.concatenate(self._buf)
                            threading.Thread(target=self._transcribe,
                                            args=(audio,), daemon=True).start()
                        self._buf = []
                        self._silence_chunks = 0

    def _transcribe(self, audio: np.ndarray):
        raise NotImplementedError


class WhisperAPICapture(_AudioCapture):
    """Transcribes via OpenAI Whisper API."""

    def __init__(self, on_transcription, openai_api_key: str):
        super().__init__(on_transcription)
        import openai
        self._client = openai.OpenAI(api_key=openai_api_key)

    def _transcribe(self, audio: np.ndarray):
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV")
        buf.seek(0)
        buf.name = "audio.wav"
        try:
            result = self._client.audio.transcriptions.create(model="whisper-1", file=buf)
            text = result.text.strip()
            if text:
                self.on_transcription(text)
        except Exception as e:
            self.on_transcription(f"[Transcription error: {e}]")


class LocalWhisperCapture(_AudioCapture):
    """Transcribes using faster-whisper running on-device (no internet needed)."""

    def __init__(self, on_transcription, model_size: str = "base"):
        super().__init__(on_transcription)
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def _transcribe(self, audio: np.ndarray):
        try:
            segments, _ = self._model.transcribe(audio, beam_size=5)
            text = " ".join(s.text for s in segments).strip()
            if text:
                self.on_transcription(text)
        except Exception as e:
            self.on_transcription(f"[Transcription error: {e}]")


def make_audio_capture(config, on_transcription) -> _AudioCapture:
    if config.whisper_mode == "local":
        return LocalWhisperCapture(on_transcription, config.whisper_local_model)
    else:
        if not config.openai_api_key:
            raise ValueError("OpenAI API key needed for Whisper API transcription.")
        return WhisperAPICapture(on_transcription, config.openai_api_key)
