"""Shared tool-selection metrics used by evaluation pipelines."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def selected_tool_names(value: Any) -> set[str]:
    """Normalize a scalar or collection of selected tool names."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, Iterable):
        return {str(item) for item in value if item is not None and str(item)}
    return {str(value)}


def candidate_recall(candidates: Any, gold_variants: Sequence[Iterable[str]]) -> float:
    candidate_set = selected_tool_names(candidates)
    variants = [{str(name) for name in gold} for gold in gold_variants] or [set()]
    return max(
        len(candidate_set & gold) / len(gold) if gold else float(not candidate_set)
        for gold in variants
    )


def tool_selection_metrics(
    selected: Any, gold_variants: Sequence[Iterable[str]]
) -> dict[str, Any]:
    """Compute exact-set CSR and ToolCoder-style Path recall.

    Multiple gold variants are supported for benchmarks whose name mapping is
    ambiguous. CSR accepts any exact variant and Path uses the best variant.
    """
    selected_set = selected_tool_names(selected)
    variants = [{str(name) for name in gold} for gold in gold_variants] or [set()]
    csr = any(selected_set == gold for gold in variants)
    path = max(
        (len(selected_set & gold) / len(gold) if gold else float(not selected_set))
        for gold in variants
    )
    return {
        "csr": float(csr),
        "path": float(path),
        "selected_tools": sorted(selected_set),
    }
