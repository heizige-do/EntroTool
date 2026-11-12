from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from entrotool.toolscope_merger import (
    EntroTool,
    KnowledgeGraphBuilder,
    StructuralEntropyMerger,
    ToolRecord,
    load_config,
    load_tools,
    run_merger_only,
)
from entrotool.evaluation_metrics import (
    candidate_recall,
    tool_selection_metrics,
)
from entrotool.toolscope_retriever import ToolScopeRetriever, evaluate_dataset


def test_tool_selection_metrics_report_csr_and_path() -> None:
    exact = tool_selection_metrics(["search", "credits"], [{"search", "credits"}])
    assert exact["csr"] == 1.0
    assert exact["path"] == 1.0

    redundant = tool_selection_metrics(
        ["search", "credits", "reviews"], [{"search", "credits"}]
    )
    assert redundant["csr"] == 0.0
    assert redundant["path"] == 1.0

    incomplete = tool_selection_metrics(["search"], [{"search", "credits"}])
    assert incomplete["csr"] == 0.0
    assert incomplete["path"] == 0.5

    empty = tool_selection_metrics([], [set()])
    assert empty["csr"] == 1.0
    assert empty["path"] == 1.0


def test_candidate_recall_counts_partial_gold_coverage() -> None:
    assert (
        candidate_recall(["search", "search", "unrelated"], [{"search", "credits"}])
        == 0.5
    )
    assert candidate_recall(["search", "credits"], [{"search", "credits"}]) == 1.0
    assert candidate_recall(["search"], [{"search", "credits"}]) == 0.5
    assert tool_selection_metrics([], [{"search", "credits"}])["path"] == 0.0


def _tool(name: str, args: set[str], results: set[str], description: str) -> ToolRecord:
    return ToolRecord(
        name=name,
        description=description,
        field="",
        arguments={
            "type": "object",
            "properties": {key: {"type": "string"} for key in args},
        },
        required_arguments=set(args),
        result_fields=set(results),
    )


class FixedEncoder:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def encode(self, documents, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        return self.values[: len(documents)]


def require_datasets(*datasets: str) -> Path:
    root = Path(__file__).resolve().parents[2] / "datasets"
    missing = [name for name in datasets if not (root / name).is_dir()]
    if missing:
        pytest.skip(f"External datasets are not available: {', '.join(missing)}")
    return root


def test_entropy_merger_uses_cosine_edges_and_preserves_singletons() -> None:
    tools = [
        _tool("triangle_area", {"base", "height"}, {"area"}, "calculate triangle area"),
        _tool(
            "triangle_area_alt", {"base", "height"}, {"area"}, "calculate triangle area"
        ),
        _tool("send_email", {"recipient", "body"}, {"message_id"}, "send an email"),
    ]
    encoder = FixedEncoder([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    graph, edges = KnowledgeGraphBuilder(
        {
            "semantic_graph": {
                "embedding_model": "thenlper/gte-large",
                "minimum_cosine": 0.95,
                "use_faiss": False,
            }
        },
        encoder=encoder,
    ).build(tools)
    assert len(edges) == 1
    assert "semantic_cosine" in edges[0].relations
    assert graph["embedding_model"] == "thenlper/gte-large"
    assert graph["candidate_generation"] == "all unordered tool pairs"
    assert graph["evaluated_pair_count"] == 3
    assert edges[0].weight > 0.99

    merger = StructuralEntropyMerger(
        tools,
        edges,
        {"target_clusters": 2, "min_structural_gain": -1.0, "min_cluster_size": 2},
    ).fit()
    assert len(merger.merge_history) == 1
    assert sorted(len(group) for group in merger.report_partition) == [1, 2]
    pruning_graph, components = merger.pruning_graph()
    assert sorted(len(group) for group in components) == [1, 2]
    assert pruning_graph["non_singleton_component_count"] == 1


def test_cosine_threshold_is_strict_and_configurable() -> None:
    tools = [
        _tool("same_a", {"x"}, set(), "same function"),
        _tool("same_b", {"x"}, set(), "same function"),
    ]
    encoder = FixedEncoder([[1.0, 0.0], [1.0, 0.0]])
    common = {
        "embedding_model": "thenlper/gte-large",
        "use_faiss": False,
    }
    _, excluded = KnowledgeGraphBuilder(
        {"semantic_graph": {**common, "minimum_cosine": 1.0}},
        encoder=encoder,
    ).build(tools)
    _, included = KnowledgeGraphBuilder(
        {"semantic_graph": {**common, "minimum_cosine": 0.95}},
        encoder=encoder,
    ).build(tools)
    assert excluded == []  # cosine == threshold is not enough
    assert len(included) == 1
    assert included[0].weight == 1.0


def test_contract_does_not_filter_shared_arguments_by_type() -> None:
    tools = [
        ToolRecord(
            name="alpha",
            description="shared domain operation",
            field="",
            arguments={"type": "object", "properties": {"x": {"type": "string"}}},
            required_arguments={"x"},
        ),
        ToolRecord(
            name="beta",
            description="shared domain operation",
            field="",
            arguments={"type": "object", "properties": {"x": {"type": "object"}}},
            required_arguments={"x"},
        ),
    ]
    encoder = FixedEncoder([[1.0, 0.0], [1.0, 0.0]])
    common = {
        "embedding_model": "thenlper/gte-large",
        "minimum_cosine": 0.95,
        "use_faiss": False,
        "contract_filter": {
            "minimum_required_jaccard": 0.25,
            "semantic_contract_fallback": {"enabled": False},
        },
    }
    graph, edges = KnowledgeGraphBuilder(
        {"semantic_graph": common}, encoder=encoder
    ).build(tools)
    assert len(edges) == 1
    assert "required_argument_type_compatibility" not in edges[0].relations
    assert "required_argument_types_must_be_compatible" not in graph["contract_filter"]


def test_semantic_contract_fallback_uses_only_cosine_or_same_name() -> None:
    tools = [
        _tool("alpha", {"left"}, set(), "different first operation"),
        _tool("beta", {"right"}, set(), "unrelated second action"),
    ]
    encoder = FixedEncoder([[1.0, 0.0], [0.87, np.sqrt(1 - 0.87**2)]])
    semantic = {
        "embedding_model": "thenlper/gte-large",
        "minimum_cosine": 0.82,
        "use_faiss": False,
        "contract_filter": {
            "enabled": True,
            "semantic_contract_fallback": {"enabled": True, "minimum_cosine": 0.86},
        },
    }
    graph, edges = KnowledgeGraphBuilder(
        {"semantic_graph": semantic}, encoder=encoder
    ).build(tools)
    assert len(edges) == 1
    assert edges[0].relations["semantic_contract_fallback"]
    assert set(graph["contract_filter"]["semantic_contract_fallback"]) == {
        "enabled",
        "minimum_cosine",
    }

    semantic["contract_filter"]["semantic_contract_fallback"]["minimum_cosine"] = 1.0
    _, boundary = KnowledgeGraphBuilder(
        {"semantic_graph": semantic}, encoder=FixedEncoder([[1.0, 0.0], [1.0, 0.0]])
    ).build(tools)
    assert boundary == []

    semantic["contract_filter"]["semantic_contract_fallback"]["minimum_cosine"] = 0.88
    _, excluded = KnowledgeGraphBuilder(
        {"semantic_graph": semantic}, encoder=encoder
    ).build(tools)
    assert excluded == []

    same_name = [
        _tool("shared", {"left"}, set(), "one"),
        _tool("shared", {"right"}, set(), "two"),
    ]
    _, retained = KnowledgeGraphBuilder(
        {"semantic_graph": semantic}, encoder=encoder
    ).build(same_name)
    assert len(retained) == 1


def test_real_osa_counts_and_duplicate_bfcl_names() -> None:
    root = require_datasets("BFCL", "BFCL_PM", "SealTool")
    bfcl = load_tools(root / "BFCL" / "BFCL_osa.json")
    bfcl_pm = load_tools(root / "BFCL_PM" / "BFCL_pm_osa.json")
    seal = load_tools(root / "SealTool" / "SealTool_osa.json")
    assert len(bfcl) == 400
    assert len({tool.name for tool in bfcl}) < len(bfcl)
    assert len({tool.uid for tool in bfcl}) == len(bfcl)
    assert len(bfcl_pm) == 482
    assert len(seal) == 1034


def test_evaluation_writes_posthoc_gold_and_metrics(tmp_path, monkeypatch) -> None:
    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            return np.asarray(
                [
                    [1.0, 0.0] if "triangle" in str(text).lower() else [0.0, 1.0]
                    for text in texts
                ],
                dtype=np.float32,
            )

    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs):
            return np.asarray(
                [1.0 if "triangle" in str(pair[1]).lower() else 0.0 for pair in pairs]
            )

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(
            SentenceTransformer=FakeSentenceTransformer,
            CrossEncoder=FakeCrossEncoder,
        ),
    )
    root = require_datasets("BFCL")
    config = load_config(Path(__file__).resolve().parents[1] / "config.json")
    result = evaluate_dataset(
        root, "BFCL", config, tmp_path, max_queries=1, no_merge=True
    )
    assert result["gold_usage"].startswith("Gold solution names are parsed only after")
    saved = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert saved["query_count"] == 1
    expected_k = {str(value) for value in config["evaluation"]["k_values"]["BFCL"]}
    assert set(saved["metrics"]) == expected_k
    assert saved["metrics"]["5"]["success"] is None
    assert saved["metrics"]["5"]["path"] == 0.0
    assert saved["metrics"]["5"]["csr"] == 0.0
    assert not (tmp_path / "entropy_pruning_graph.json").exists()


def test_merger_only_stops_before_queries_retriever_and_llm(tmp_path) -> None:
    root = require_datasets("BFCL")
    config = load_config(Path(__file__).resolve().parents[1] / "config.json")
    # One embedding per real BFCL tool. Pair filtering is exercised without
    # loading the production encoder.
    count = 400
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    values = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
    values[1] = values[0]
    local_config = json.loads(json.dumps(config))
    local_config["merger"]["semantic_graph"]["use_faiss"] = False
    result = run_merger_only(
        root, "BFCL", local_config, tmp_path, encoder=FixedEncoder(values.tolist())
    )
    assert result["stage"] == "connected_components_complete"
    assert result["queries_or_gold_loaded"] is False
    assert result["retriever_initialized"] is False
    assert result["llm_called"] is False
    assert (
        result["semantic_graph"]["embedding_model"]
        == config["merger"]["semantic_graph"]["embedding_model"]
    )
    assert (
        result["semantic_graph"]["candidate_generation"] == "all unordered tool pairs"
    )
    assert result["semantic_graph"]["edge_condition"] == "cosine > minimum_cosine"
    assert "cosine similarity" in result["semantic_graph"]["edge_weight"]
    assert (tmp_path / "connected_components.json").exists()
    assert (tmp_path / "connected_components_review.md").exists()
    assert not (tmp_path / "evaluation.json").exists()


def test_algorithm_implementations_live_in_separate_modules() -> None:
    assert KnowledgeGraphBuilder.__module__.endswith("toolscope_merger")
    assert StructuralEntropyMerger.__module__.endswith("toolscope_merger")
    assert EntroTool.__module__.endswith("toolscope_merger")
    assert ToolScopeRetriever.__module__.endswith("toolscope_retriever")
