"""Orchestrates the eval: run both models over the test set, then score."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import metrics as M
from .config import TaskConfig
from .providers import Provider


def _run_model(task: TaskConfig, provider: Provider, rows: list[dict[str, Any]]) -> list[dict]:
    """Run one model over every test row, collecting completions."""
    records = []
    for i, row in enumerate(rows):
        prompt = task.render(row["input"])
        completion = provider.complete(prompt, task.system, task.max_tokens)
        records.append({"expected": row["expected"], "completion": completion})
        print(f"  [{provider.spec}] {i + 1}/{len(rows)}", end="\r", flush=True)
    print()
    return records


def _score(task: TaskConfig, provider: Provider, records: list[dict]) -> dict[str, Any]:
    """Apply the task's configured metrics to one model's records."""
    out: dict[str, Any] = {}
    for name in task.metrics:
        if name in M.SCALAR_METRICS:
            out[name] = round(M.SCALAR_METRICS[name](records), 4)
        elif name == "latency":
            out["latency"] = M.latency(records)
        elif name == "cost":
            out["cost_per_1k_usd"] = M.cost_per_1k(records, task.pricing.get(provider.spec))
        else:
            out[name] = f"<unknown metric '{name}'>"
    return out


def run_eval(
    task: TaskConfig,
    baseline: Provider,
    candidate: Provider,
    task_dir: Path,
) -> dict[str, Any]:
    """Run baseline + candidate over the test set and return a results dict."""
    rows = task.load_test_set(task_dir)
    print(f"Task: {task.name}  ({len(rows)} examples)")

    print(f"Running baseline:  {baseline.spec}")
    base_records = _run_model(task, baseline, rows)

    print(f"Running candidate: {candidate.spec}")
    cand_records = _run_model(task, candidate, rows)

    return {
        "task": task.name,
        "n_examples": len(rows),
        "baseline": {"model": baseline.spec, "metrics": _score(task, baseline, base_records)},
        "candidate": {"model": candidate.spec, "metrics": _score(task, candidate, cand_records)},
    }
