"""Structural-Entropy ToolScope with a lazily loaded retriever stage."""

from __future__ import annotations

from typing import Any

from .toolscope_merger import (
    EntroTool,
    KnowledgeGraphBuilder,
    StructuralEntropyMerger,
    load_config,
    run_merger_only,
)


_RETRIEVER_EXPORTS = {"ToolScopeRetriever", "evaluate_dataset", "load_dataset"}


def __getattr__(name: str) -> Any:
    if name in _RETRIEVER_EXPORTS:
        from . import toolscope_retriever

        return getattr(toolscope_retriever, name)
    raise AttributeError(name)


__all__ = [
    "EntroTool",
    "KnowledgeGraphBuilder",
    "StructuralEntropyMerger",
    "ToolScopeRetriever",
    "evaluate_dataset",
    "load_config",
    "load_dataset",
    "run_merger_only",
]
