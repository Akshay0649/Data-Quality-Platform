import io
import threading
import numpy as np

SAMPLE_RATE = 16_000
CHANNELS = 1
CHUNK_SECS = 0.4
SILENCE_THRESHOLD = 0.003
SILENCE_SECS = 0.5
MIN_SPEECH_SECS = 0.4
MAX_BUF_SECS = 4


def list_input_devices():
    import sounddevice as sd
    devices = sd.query_devices()
    return [{"index": i, "name": d["name"]}
            for i, d in enumerate(devices) if d["max_input_channels"] > 0]


def find_device_index(name_substring):
    if not name_substring:
        return None
    for d in list_input_devices():
        if name_substring.lower() in d["name"].lower():
            return d["index"]
    return None


class _AudioCapture:
    def __init__(self, on_transcription, device_index=None):
        self.on_transcription = on_transcription
        self.device_index = device_index
        self._recording = False
        self._thread = None
        self._buf = []
        self._silence_chunks = 0
        self._silence_limit = int(SILENCE_SECS / CHUNK_SECS)
        self._min_chunks = int(MIN_SPEECH_SECS / CHUNK_SECS)
        self._max_chunks = int(MAX_BUF_SECS / CHUNK_SECS)

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
                            dtype="float32", blocksize=chunk_samples,
                            device=self.device_index) as stream:
            while self._recording:
                data, _ = stream.read(chunk_samples)
                chunk = data.flatten()
                rms = float(np.sqrt(np.mean(chunk ** 2)))

                if rms > SILENCE_THRESHOLD:
                    self._buf.append(chunk)
                    self._silence_chunks = 0
                    if len(self._buf) >= self._max_chunks:
                        audio = np.concatenate(self._buf)
                        threading.Thread(target=self._transcribe,
                                         args=(audio,), daemon=True).start()
                        self._buf = []
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

    def _transcribe(self, audio):
        raise NotImplementedError


class WhisperAPICapture(_AudioCapture):
    def __init__(self, on_transcription, openai_api_key, device_index=None):
        super().__init__(on_transcription, device_index)
        import openai
        self._client = openai.OpenAI(api_key=openai_api_key)

    def _transcribe(self, audio):
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV")
        buf.seek(0)
        buf.name = "audio.wav"
        try:
            result = self._client.audio.transcriptions.create(
                model="whisper-1", file=buf)
            text = result.text.strip()
            if text:
                self.on_transcription(text)
        except Exception as e:
            self.on_transcription("[Transcription error: " + str(e) + "]")


class LocalWhisperCapture(_AudioCapture):
    def __init__(self, on_transcription, model_size="base", device_index=None):
        super().__init__(on_transcription, device_index)
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def _transcribe(self, audio):
        try:
            segments, _ = self._model.transcribe(audio, beam_size=5)
            text = " ".join(s.text for s in segments).strip()
            if text:
                self.on_transcription(text)
        except Exception as e:
            self.on_transcription("[Transcription error: " + str(e) + "]")


def make_audio_capture(config, on_transcription):
    device_index = find_device_index(config.audio_device)
    if config.whisper_mode == "local":
        return LocalWhisperCapture(on_transcription, config.whisper_local_model,
                                   device_index)
    else:
        if not config.openai_api_key:
            raise ValueError("OpenAI API key needed for Whisper API transcription.")
        return WhisperAPICapture(on_transcription, config.openai_api_key, device_index)
