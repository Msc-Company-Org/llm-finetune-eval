"""Unit tests for the scoring metrics — no network, no API keys."""

from finetune_eval import metrics as M
from finetune_eval.providers.base import Completion


def _rec(text, expected):
    return {"expected": expected, "completion": Completion(text, 10, 5, 0.1)}


def test_exact_match_normalizes_whitespace_and_case():
    records = [
        _rec("Hello World", "hello   world"),
        _rec("nope", "yes"),
    ]
    assert M.exact_match(records) == 0.5


def test_json_match_handles_fences_and_order():
    records = [
        _rec('```json\n{"b": 2, "a": 1}\n```', {"a": 1, "b": 2}),
        _rec('{"a": 1}', {"a": 2}),
    ]
    assert M.json_match(records) == 0.5


def test_format_adherence_counts_parseable_json():
    records = [
        _rec('{"ok": true}', {}),
        _rec("not json at all", {}),
    ]
    assert M.format_adherence(records) == 0.5


def test_refusal_rate_detects_markers():
    records = [
        _rec("I can't help with that.", {}),
        _rec("Sure, here it is.", {}),
    ]
    assert M.refusal_rate(records) == 0.5


def test_latency_reports_percentiles():
    records = [
        {"expected": "", "completion": Completion("x", 1, 1, s)}
        for s in (0.1, 0.2, 0.3, 0.4)
    ]
    lat = M.latency(records)
    assert "p50_s" in lat and "p95_s" in lat
    assert lat["p95_s"] >= lat["p50_s"]


def test_cost_per_1k_uses_pricing():
    records = [_rec("x", "y")]  # 10 input + 5 output tokens
    price = {"input": 30.0, "output": 60.0}  # per 1M
    # (10*30 + 5*60)/1e6 = 0.0006 per call -> *1000 = 0.6
    assert M.cost_per_1k(records, price) == 0.6
