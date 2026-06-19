"""Provider interface: anything that turns a prompt into a completion + usage."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Completion:
    """One model response plus the signals the metrics need."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class Provider(Protocol):
    """Every backend (OpenAI, Anthropic, local) implements this one method."""

    #: provider string this instance was built from, e.g. "openai:gpt-4"
    spec: str
    #: the model id, e.g. "gpt-4" or "claude-opus-4-8" or "my-qlora-model"
    model: str

    def complete(self, prompt: str, system: str | None, max_tokens: int) -> Completion:
        """Run one completion. Implementations should populate token counts."""
        ...


def timed(fn):
    """Helper: wrap a call, measure wall-clock latency, attach it to the Completion."""

    def wrapper(*args, **kwargs) -> Completion:
        start = time.perf_counter()
        result: Completion = fn(*args, **kwargs)
        result.latency_s = time.perf_counter() - start
        return result

    return wrapper


def build_provider(spec: str) -> Provider:
    """Parse a provider string and return the matching Provider.

    Forms:
      openai:<model>
      anthropic:<model>
      local:<base_url>::<model>
    """
    if ":" not in spec:
        raise ValueError(
            f"bad provider spec {spec!r} — expected 'provider:model' "
            "(or 'local:base_url::model')"
        )
    kind, rest = spec.split(":", 1)

    if kind == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(spec=spec, model=rest)
    if kind == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(spec=spec, model=rest)
    if kind == "local":
        from .local_provider import LocalProvider

        if "::" not in rest:
            raise ValueError(
                f"bad local spec {spec!r} — expected 'local:<base_url>::<model>'"
            )
        base_url, model = rest.split("::", 1)
        return LocalProvider(spec=spec, base_url=base_url, model=model)

    raise ValueError(f"unknown provider kind {kind!r} in {spec!r}")
