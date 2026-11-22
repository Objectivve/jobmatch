"""Run the golden set: `python -m jobmatch.evals.run --backend fake`.

`fake` needs no API key or network; `live` uses the settings in `.env`.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ..config import Settings, get_settings
from ..graph.workflow import build_graph, create_sqlite_checkpointer
from .fake_model import DeterministicModel, OFFLINE_TOOLS
from .harness import evaluate_cases, load_golden_set, summarize, write_reports

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLDEN = ROOT / "data" / "evals" / "golden_set.sample.jsonl"


def offline_graph(settings: Settings):
    """Same graph, deterministic model and fake tools: zero external calls."""
    model = DeterministicModel()
    return build_graph(
        settings=settings,
        checkpointer=create_sqlite_checkpointer(":memory:"),
        model_factory=lambda _name: model,
        tools=OFFLINE_TOOLS,
    )


def live_graph(settings: Settings):
    return build_graph(settings=settings, checkpointer=create_sqlite_checkpointer(":memory:"))


def _settings(backend: str) -> Settings:
    base = get_settings()
    rubric_path = Path(base.rubric_path)
    if not rubric_path.is_absolute():
        rubric_path = ROOT / rubric_path
    settings = replace(base, rubric_path=str(rubric_path), checkpoint_db_path=":memory:")
    if backend == "fake":
        settings = replace(settings, gemini_api_key=settings.gemini_api_key or "offline")
    return settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the JobMatch golden set")
    parser.add_argument("--golden-set", default=str(DEFAULT_GOLDEN))
    parser.add_argument("--out-dir", default=str(ROOT / "reports"))
    parser.add_argument("--backend", choices=("fake", "live"), default="fake")
    parser.add_argument("--limit", type=int, default=0, help="0 runs every case")
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 if any case fails"
    )
    args = parser.parse_args(argv)

    settings = _settings(args.backend)
    cases = load_golden_set(args.golden_set)
    if args.limit > 0:
        cases = cases[: args.limit]

    factory = offline_graph if args.backend == "fake" else live_graph
    results = evaluate_cases(
        cases, graph_factory=factory, settings=settings, base_dir=ROOT
    )
    summary = summarize(results)

    out_dir = Path(args.out_dir)
    write_reports(summary, out_dir / "eval_summary.json", out_dir / "eval_summary.md")

    print(f"backend: {args.backend} | cases: {summary['cases']}")
    print(
        "task success: {task_success_rate} | faithfulness: {faithfulness_rate} | "
        "p95: {latency_p95_s}s | cost: ${cost_total_usd}".format(**summary)
    )
    for result in results:
        print(
            f"  {result.case_id}: status={result.status} "
            f"task={result.task_success} grounded={result.grounding_passed}"
            + (f" error={result.error}" if result.error else "")
        )
    print(f"reports: {out_dir / 'eval_summary.json'}, {out_dir / 'eval_summary.md'}")

    failed = any(
        not (result.task_success and result.grounding_passed) for result in results
    )
    return 1 if failed and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
