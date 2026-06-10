import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_FILE = Path.home() / ".parakeet_personal" / "config.json"


@dataclass
class Config:
    ai_provider: str = "claude"
    claude_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4o"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    whisper_mode: str = "api"
    whisper_local_model: str = "base"
    whisper_language: str = ""  # e.g. 'de' for German, 'en' for English, blank = auto
    resume_path: str = ""
    overlay_opacity: float = 0.93
    system_prompt_extra: str = ""
    audio_device: str = ""

    def __post_init__(self):
        if os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
        if os.getenv("OPENAI_API_KEY"):
            self.openai_api_key = os.environ["OPENAI_API_KEY"]

    def save(self):
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        if os.getenv("ANTHROPIC_API_KEY"):
            data["anthropic_api_key"] = ""
        if os.getenv("OPENAI_API_KEY"):
            data["openai_api_key"] = ""
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls) -> "Config":
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                data = json.load(f)
            valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**valid)
        return cls()
