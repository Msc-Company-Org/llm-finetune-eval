"""Command-line entrypoint: `finetune-eval run --task ... --baseline ... --candidate ...`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_task
from .providers import build_provider
from .report import render_markdown
from .runner import run_eval


def _cmd_run(args: argparse.Namespace) -> int:
    task_path = Path(args.task)
    task = load_task(task_path)

    baseline = build_provider(args.baseline)
    candidate = build_provider(args.candidate)

    results = run_eval(task, baseline, candidate, task_dir=task_path.parent)

    md = render_markdown(results)
    out_md = Path(args.out)
    out_md.write_text(md, encoding="utf-8")
    out_json = out_md.with_suffix(".json")
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print()
    print(md)
    print(f"Wrote {out_md} and {out_json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="finetune-eval",
        description="Benchmark a fine-tuned model against a frontier-API baseline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an eval (baseline vs. candidate)")
    run.add_argument("--task", required=True, help="path to the task YAML")
    run.add_argument(
        "--baseline",
        required=True,
        help="provider:model for the frontier baseline (e.g. openai:gpt-4)",
    )
    run.add_argument(
        "--candidate",
        required=True,
        help="provider:model for the fine-tune "
        "(e.g. local:http://localhost:8000/v1::my-model)",
    )
    run.add_argument("--out", default="report.md", help="output report path (.md)")
    run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
