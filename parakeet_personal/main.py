"""
Parakeet Personal — AI interview assistant.
Run: python parakeet_personal/main.py
"""

import threading
from pathlib import Path

from ai_client import make_client
from audio import make_audio_capture
from config import Config
from overlay import ParakeetOverlay
from resume import build_system_prompt, load_resume
from screen import capture_screen_text


class ParakeetApp:
    def __init__(self):
        self.config = Config.load()
        self.resume_text = load_resume(self.config.resume_path)
        self._client = None
        self._audio = None
        self.overlay: ParakeetOverlay | None = None
        self._refresh_client()

    # ── AI client ─────────────────────────────────────────────────────────────

    def _refresh_client(self):
        try:
            self._client = make_client(self.config)
        except ValueError as e:
            self._client = None
            print(f"[Parakeet] {e}")

    # ── Overlay callbacks ─────────────────────────────────────────────────────

    def on_ask(self, question: str):
        if not self._client:
            self.overlay.set_status(
                f"⚠ No {self.config.ai_provider} client — open Settings and add your API key."
            )
            return
        system = build_system_prompt(self.resume_text, self.config.system_prompt_extra)
        self.overlay.start_answer(question)
        self.overlay.set_status("Thinking…")
        threading.Thread(
            target=self._stream_answer, args=(question, system), daemon=True
        ).start()

    def _stream_answer(self, question: str, system: str):
        try:
            for token in self._client.stream([{"role": "user", "content": question}], system):
                self.overlay.stream_token(token)
        except Exception as e:
            self.overlay.stream_token(f"\n\n[Error: {e}]")
        finally:
            self.overlay.end_answer()

    def on_screen_capture(self):
        self.overlay.set_status("Capturing screen…")
        try:
            text = capture_screen_text()
            if text:
                self.overlay.set_question(text)
                self.overlay.set_status(f"Screen captured ({len(text)} chars) — edit if needed, then Ask")
            else:
                self.overlay.set_status("No text found on screen")
        except Exception as e:
            self.overlay.set_status(f"Screen capture failed: {e}")

    def on_audio_toggle(self, active: bool):
        if active:
            if self._audio is None:
                try:
                    self._audio = make_audio_capture(self.config, self._on_transcription)
                except Exception as e:
                    self.overlay.set_status(f"Audio init failed: {e}")
                    return
            self._audio.start()
            self.overlay.set_status("Listening… (speak now)")
        else:
            if self._audio:
                self._audio.stop()
            self.overlay.set_status("Stopped listening")

    def _on_transcription(self, text: str):
        preview = text[:60] + ("…" if len(text) > 60 else "")
        self.overlay.set_status(f"Heard: {preview}")
        self.overlay.set_question(text)
        self.on_ask(text)

    def on_provider_change(self, provider: str):
        self.config.ai_provider = provider
        self.config.save()
        self._audio = None  # may need a different key
        self._refresh_client()
        label = "Claude" if provider == "claude" else "OpenAI"
        self.overlay.set_status(f"Switched to {label}")

    def on_resume_load(self, path: str):
        self.config.resume_path = path
        self.config.save()
        self.resume_text = load_resume(path)
        name = Path(path).name
        chars = len(self.resume_text)
        self.overlay.set_status(f"Resume loaded: {name} ({chars} chars)")

    def on_settings_save(self):
        self._refresh_client()
        self._audio = None  # keys may have changed
        self.overlay.set_status("Settings saved")

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        self.overlay = ParakeetOverlay(
            config=self.config,
            on_ask=self.on_ask,
            on_screen_capture=self.on_screen_capture,
            on_audio_toggle=self.on_audio_toggle,
            on_provider_change=self.on_provider_change,
            on_resume_load=self.on_resume_load,
            on_settings_save=self.on_settings_save,
        )
        self.overlay.run()


if __name__ == "__main__":
    ParakeetApp().run()
