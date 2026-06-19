"""Scoring metrics. Each works on a list of per-example records.

A record is {"expected": ..., "completion": Completion}. Metric functions return
a float (higher = better, except cost/latency which report raw values the report
labels explicitly).
"""

from __future__ import annotations

import json
import re
from statistics import median
from typing import Any

# ---- text normalization -----------------------------------------------------

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.strip()).lower()


def _extract_json(text: str) -> Any | None:
    """Best-effort: pull the first JSON object/array out of a model response."""
    text = text.strip()
    # Strip ```json fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the first {...} or [...] span.
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                return None
    return None


_REFUSAL_MARKERS = (
    "i can't", "i cannot", "i'm sorry", "i am sorry", "i'm unable",
    "as an ai", "i won't", "i will not", "cannot assist", "can't help",
)


# ---- metrics ----------------------------------------------------------------


def exact_match(records: list[dict[str, Any]]) -> float:
    hits = sum(
        _norm(str(r["completion"].text)) == _norm(str(r["expected"]))
        for r in records
    )
    return hits / len(records)


def json_match(records: list[dict[str, Any]]) -> float:
    hits = 0
    for r in records:
        pred = _extract_json(r["completion"].text)
        if pred is not None and pred == r["expected"]:
            hits += 1
    return hits / len(records)


def f1_token(records: list[dict[str, Any]]) -> float:
    total = 0.0
    for r in records:
        pred = set(_norm(str(r["completion"].text)).split())
        gold = set(_norm(str(r["expected"])).split())
        if not pred and not gold:
            total += 1.0
            continue
        if not pred or not gold:
            continue
        overlap = len(pred & gold)
        if overlap == 0:
            continue
        precision = overlap / len(pred)
        recall = overlap / len(gold)
        total += 2 * precision * recall / (precision + recall)
    return total / len(records)


def format_adherence(records: list[dict[str, Any]]) -> float:
    """Fraction of outputs that parse as JSON (the most common required format)."""
    ok = sum(_extract_json(r["completion"].text) is not None for r in records)
    return ok / len(records)


def refusal_rate(records: list[dict[str, Any]]) -> float:
    refused = sum(
        any(m in r["completion"].text.lower() for m in _REFUSAL_MARKERS)
        for r in records
    )
    return refused / len(records)


def latency(records: list[dict[str, Any]]) -> dict[str, float]:
    lat = sorted(r["completion"].latency_s for r in records)
    p95_idx = max(0, int(round(0.95 * (len(lat) - 1))))
    return {"p50_s": round(median(lat), 3), "p95_s": round(lat[p95_idx], 3)}


def cost_per_1k(records: list[dict[str, Any]], price: dict[str, float] | None) -> float:
    """Cost (USD) per 1,000 calls, from measured tokens and per-1M-token pricing."""
    if not price:
        return float("nan")
    n = len(records)
    in_tok = sum(r["completion"].input_tokens for r in records)
    out_tok = sum(r["completion"].output_tokens for r in records)
    usd = (in_tok * price.get("input", 0.0) + out_tok * price.get("output", 0.0)) / 1e6
    return round(usd / n * 1000, 4)


SCALAR_METRICS = {
    "exact_match": exact_match,
    "json_match": json_match,
    "f1_token": f1_token,
    "format_adherence": format_adherence,
    "refusal_rate": refusal_rate,
}
