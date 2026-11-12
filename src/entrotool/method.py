"""Compatibility facade and CLI for Structural-Entropy ToolScope.

Algorithm implementations are intentionally separated:
- toolscope_merger.py: offline graph/entropy/components/LLM consolidation
- toolscope_retriever.py: downstream ToolScopeRetriever/Agent/evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .toolscope_merger import (
    EntroTool,
    KGEdge,
    KnowledgeGraphBuilder,
    OpenAICompatibleLLM,
    PromptStore,
    StructuralEntropyMerger,
    ToolRecord,
    load_config,
    load_tools,
    normalize_tool,
    run_merger_only,
)
from .toolscope_retriever import (
    QueryDecomposer,
    ToolScopeRetriever,
    ToolSelectionAgent,
    evaluate_dataset,
    load_dataset,
    query_text,
    solution_names,
)

__all__ = [
    "EntroTool",
    "KGEdge",
    "KnowledgeGraphBuilder",
    "OpenAICompatibleLLM",
    "PromptStore",
    "StructuralEntropyMerger",
    "ToolRecord",
    "QueryDecomposer",
    "ToolScopeRetriever",
    "ToolSelectionAgent",
    "evaluate_dataset",
    "load_config",
    "load_dataset",
    "load_tools",
    "normalize_tool",
    "query_text",
    "run_merger_only",
    "solution_names",
    "main",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EntroTool: structural-entropy ToolScopeMerger replacement"
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", "BFCL", "BFCL_PM", "SealTool"],
    )
    parser.add_argument(
        "--dataset-root", default=str(Path(__file__).resolve().parents[3] / "datasets")
    )
    parser.add_argument(
        "--config", default=str(Path(__file__).resolve().parents[2] / "config.json")
    )
    parser.add_argument(
        "--prompt-dir", default=str(Path(__file__).resolve().parents[2] / "prompts")
    )
    parser.add_argument(
        "--output-root", default=str(Path(__file__).resolve().parents[2] / "outputs")
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--merger-only",
        action="store_true",
        help="stop after entropy pruning graph and connected components; load no queries/gold",
    )
    parser.add_argument(
        "--no-merge", action="store_true", help="identity control; do not merge tools"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="enable ToolScope doc merger, query decomposer and selection agent",
    )
    parser.add_argument(
        "--auto-correct",
        action="store_true",
        help="also enable ToolScope's optional connected-component auto-correction",
    )
    parser.add_argument(
        "--tool-selection-llm-id",
        default=None,
        help="config llm id used by ToolScope's final tool-selection agent",
    )
    parser.add_argument(
        "--matched-toolscope-size",
        action="store_true",
        help="optional control: stop at ToolScope's reported merged tool count",
    )
    parser.add_argument(
        "--strict-models",
        action="store_true",
        help="fail instead of using TF-IDF/reranker fallback",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="do not print per-query ToolScopeRetriever/Agent progress",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    datasets = (
        [args.dataset]
        if args.dataset != "all"
        else ["BFCL", "BFCL_pm", "SealTool"]
    )
    for dataset in datasets:
        output = Path(args.output_root) / dataset
        if args.merger_only:
            summary = run_merger_only(
                args.dataset_root,
                dataset,
                config,
                output,
                matched_toolscope_size=args.matched_toolscope_size,
            )
            print(json.dumps({**summary, "output": str(output.resolve())}, ensure_ascii=False, indent=2))
            continue
        summary = evaluate_dataset(
            args.dataset_root,
            dataset,
            config,
            output,
            use_llm=args.use_llm,
            auto_correct=args.auto_correct,
            no_merge=args.no_merge,
            matched_toolscope_size=args.matched_toolscope_size,
            tool_selection_llm_id=args.tool_selection_llm_id,
            max_queries=args.max_queries,
            strict_models=args.strict_models,
            prompt_dir=args.prompt_dir,
            show_progress=not args.quiet_progress,
        )
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "input_tools": summary["input_tool_count"],
                    "merged_tools": summary["merged_tool_count"],
                    "metrics": summary["metrics"],
                    "output": str(output.resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

