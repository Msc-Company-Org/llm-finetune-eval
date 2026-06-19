"""Anthropic baseline provider (official `anthropic` SDK).

Used as a frontier baseline to benchmark a fine-tune against. The model id comes
from the task/CLI (e.g. ``anthropic:claude-opus-4-8``); we don't hardcode it so
you can baseline against whichever Claude model your product actually calls.
"""

from __future__ import annotations

import time

from .base import Completion


class AnthropicProvider:
    """Wraps `client.messages.create` and reports tokens + latency."""

    def __init__(self, spec: str, model: str) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError(
                "the anthropic extra is required: pip install 'llm-finetune-eval[anthropic]'"
            ) from e
        self.spec = spec
        self.model = model
        # Resolves ANTHROPIC_API_KEY from the environment.
        self._client = Anthropic()

    def complete(self, prompt: str, system: str | None, max_tokens: int) -> Completion:
        start = time.perf_counter()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.perf_counter() - start

        # content is a list of blocks; concatenate the text blocks.
        text = "".join(block.text for block in resp.content if block.type == "text")
        return Completion(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_s=latency,
        )
