# llm-finetune-eval

**Benchmark a fine-tuned model against your frontier-API baseline — the part most
fine-tuning skips.**

A fine-tune without a before/after eval is a vibe, not an upgrade. This toolkit
makes the comparison reproducible: same test set, same metrics, frontier API
(GPT-4 / Claude) vs. your tuned open model.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Why

For a narrow, high-volume task, a small fine-tuned open model often matches or
beats a frontier API at **10–50× lower inference cost** — and can run on-prem.
But you only *know* that if you measure it. This is the measuring tool.

```
                 ┌──────────────┐
   test set ───▶ │  baseline    │ ──▶ predictions ─┐
                 │ (gpt-4/claude)│                  │
                 └──────────────┘                   ├──▶ metrics ──▶ report.md
                 ┌──────────────┐                   │    (acc / cost /
   test set ───▶ │  candidate   │ ──▶ predictions ─┘     latency / format)
                 │ (your tune)  │
                 └──────────────┘
```

## Features

- 🎯 **Side-by-side eval** — frontier baseline vs. fine-tuned candidate on your test set
- 📊 **Metrics that decide production** — accuracy / F1, format adherence, refusal
  rate, **cost per 1k calls**, p50/p95 latency
- 🔌 **Pluggable providers** — OpenAI, Anthropic, and any OpenAI-compatible local
  server (vLLM, TGI, Ollama)
- 🧪 **Deterministic test splits** — same inputs to both sides, every run
- 📄 **Shareable before/after report** — Markdown + JSON, ready to drop in a deck

## Install

```bash
pip install -e .
```

Set whichever provider keys you'll use:

```bash
export OPENAI_API_KEY=...        # baseline via OpenAI
export ANTHROPIC_API_KEY=...     # baseline via Anthropic
# local/candidate models need no key (or a dummy one)
```

## Quickstart

1. Describe your task in a YAML file (see [`tasks/extraction.example.yaml`](tasks/extraction.example.yaml)).
2. Point it at your test set (`.jsonl`, one `{ "input": ..., "expected": ... }` per line).
3. Run the eval:

```bash
finetune-eval run \
  --task tasks/extraction.example.yaml \
  --baseline openai:gpt-4 \
  --candidate local:http://localhost:8000/v1::my-qlora-model \
  --out report.md
```

You get `report.md` (human-readable) and `report.json` (machine-readable) with the
before/after table.

### Provider strings

`finetune-eval` takes a model as `provider:model` (or `provider:base_url::model`
for local servers):

| String | Meaning |
|---|---|
| `openai:gpt-4` | OpenAI GPT-4 |
| `anthropic:claude-opus-4-8` | Anthropic Claude (baseline) |
| `local:http://localhost:8000/v1::my-model` | OpenAI-compatible server (vLLM/TGI/Ollama) |

## Defining a task

```yaml
name: invoice-field-extraction
system: |
  Extract the fields as strict JSON. Output ONLY the JSON object.
prompt_template: |
  Extract vendor, total, and due_date from this invoice text:
  ---
  {input}
test_set: data/invoices.test.jsonl     # {"input": "...", "expected": {...}}
metrics: [json_match, format_adherence, latency, cost]
max_tokens: 512
# per-1M-token prices used for the cost metric (USD)
pricing:
  "openai:gpt-4":               { input: 30.0, output: 60.0 }
  "anthropic:claude-opus-4-8":  { input: 5.0,  output: 25.0 }
  "local:my-qlora-model":       { input: 0.1,  output: 0.1 }   # your GPU amortized
```

## Metrics

| Metric | What it measures |
|---|---|
| `exact_match` | Prediction equals expected (normalized string) |
| `json_match` | Parsed JSON equals expected dict (order-insensitive) |
| `f1_token` | Token-level F1 (good for extraction / spans) |
| `format_adherence` | Fraction of outputs that parse as the required format |
| `refusal_rate` | Fraction of outputs that look like a refusal |
| `latency` | p50 / p95 wall-clock per call |
| `cost` | Cost per 1k calls, from `pricing` + measured tokens |

## Who builds this

Maintained by **MSC Labs**. We build custom models end to end — data, training,
evaluation, deployment — and hand you the weights.

If you'd rather not run the whole pipeline yourself, we do it as a service. Free
30-min model audit: we'll tell you if fine-tuning is worth it for your task.
→ https://labs.msccompany.com.br/assessment

## License

Apache-2.0 — see [LICENSE](LICENSE).
