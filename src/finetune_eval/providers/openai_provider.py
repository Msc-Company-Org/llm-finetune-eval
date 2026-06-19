"""OpenAI baseline provider (official `openai` SDK)."""

from __future__ import annotations

import time

from .base import Completion


class OpenAIProvider:
    """Wraps `chat.completions.create` and reports tokens + latency."""

    def __init__(self, spec: str, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError(
                "the openai extra is required: pip install 'llm-finetune-eval[openai]'"
            ) from e
        self.spec = spec
        self.model = model
        # Resolves OPENAI_API_KEY from the environment.
        self._client = OpenAI()

    def complete(self, prompt: str, system: str | None, max_tokens: int) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
        )
        latency = time.perf_counter() - start

        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_s=latency,
        )
