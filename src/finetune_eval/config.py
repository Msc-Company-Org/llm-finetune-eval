"""Task configuration: load and validate the YAML that describes an eval."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_METRICS = ["exact_match", "format_adherence", "latency", "cost"]


@dataclass
class TaskConfig:
    """A single eval task: the prompt, the test set, and which metrics to score."""

    name: str
    prompt_template: str
    test_set: str
    system: str | None = None
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    max_tokens: int = 512
    # provider-string -> {"input": $/1M, "output": $/1M}
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)

    def render(self, example_input: Any) -> str:
        """Fill the prompt template with one test-set input."""
        if isinstance(example_input, (dict, list)):
            example_input = json.dumps(example_input, ensure_ascii=False)
        return self.prompt_template.format(input=example_input)

    def load_test_set(self, base_dir: Path) -> list[dict[str, Any]]:
        """Read the .jsonl test set. Each line: {"input": ..., "expected": ...}."""
        path = Path(self.test_set)
        if not path.is_absolute():
            path = base_dir / path
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
                if "input" not in row or "expected" not in row:
                    raise ValueError(
                        f"{path}:{lineno}: each row needs 'input' and 'expected' keys"
                    )
                rows.append(row)
        if not rows:
            raise ValueError(f"{path}: test set is empty")
        return rows


def load_task(path: str | Path) -> TaskConfig:
    """Parse a task YAML file into a validated TaskConfig."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    required = ("name", "prompt_template", "test_set")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"{path}: task is missing required keys: {', '.join(missing)}")

    return TaskConfig(
        name=raw["name"],
        prompt_template=raw["prompt_template"],
        test_set=raw["test_set"],
        system=raw.get("system"),
        metrics=raw.get("metrics", list(DEFAULT_METRICS)),
        max_tokens=int(raw.get("max_tokens", 512)),
        pricing=raw.get("pricing", {}),
    )
