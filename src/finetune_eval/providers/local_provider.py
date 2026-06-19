"""Local / self-hosted provider for any OpenAI-compatible server.

Works with vLLM, TGI (OpenAI-compatible mode), Ollama, llama.cpp server, etc.
This is the path your fine-tuned candidate model usually takes — you serve the
QLoRA-merged weights behind an OpenAI-compatible `/v1/chat/completions` endpoint
and point the eval at it.
"""

from __future__ import annotations

import os
import time

import httpx

from .base import Completion


class LocalProvider:
    """Calls an OpenAI-compatible chat endpoint over HTTP (no SDK dependency)."""

    def __init__(self, spec: str, base_url: str, model: str, timeout: float = 120.0) -> None:
        self.spec = spec
        self.model = model
        self.base_url = base_url.rstrip("/")
        # Most local servers ignore the key; send a dummy unless one is set.
        self._api_key = os.environ.get("LOCAL_API_KEY", "not-needed")
        self._client = httpx.Client(timeout=timeout)

    def complete(self, prompt: str, system: str | None, max_tokens: int) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        start = time.perf_counter()
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
        )
        resp.raise_for_status()
        latency = time.perf_counter() - start
        data = resp.json()

        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        return Completion(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=latency,
        )
