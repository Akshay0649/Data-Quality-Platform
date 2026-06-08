from abc import ABC, abstractmethod
from typing import Iterator


class AIClient(ABC):
    @abstractmethod
    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        ...


class ClaudeClient(AIClient):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=system or "You are a helpful assistant.",
            messages=messages,
        ) as s:
            yield from s.text_stream


class OpenAIClient(AIClient):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        all_msgs = []
        if system:
            all_msgs.append({"role": "system", "content": system})
        all_msgs.extend(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=all_msgs,
            stream=True,
            max_tokens=1024,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def make_client(config) -> AIClient:
    if config.ai_provider == "claude":
        if not config.anthropic_api_key:
            raise ValueError("Anthropic API key not set. Add it in Settings or set ANTHROPIC_API_KEY env var.")
        return ClaudeClient(config.anthropic_api_key, config.claude_model)
    else:
        if not config.openai_api_key:
            raise ValueError("OpenAI API key not set. Add it in Settings or set OPENAI_API_KEY env var.")
        return OpenAIClient(config.openai_api_key, config.openai_model)
