"""Unchanged paper-specified ToolScopeRetriever and post-retrieval evaluation.

This module does not construct structural-entropy communities.  It consumes the
merged toolset and original-to-canonical mapping produced by
:mod:`entrotool.toolscope_merger`.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation_metrics import tool_selection_metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .toolscope_merger import (
    FUNCTION_RE,
    EntroTool,
    KnowledgeGraphBuilder,
    OpenAICompatibleLLM,
    PromptStore,
    ToolRecord,
    _parse_json_object,
    _json_dump,
    _json_load,
    load_tools,
)


def _tool_text(tool: ToolRecord) -> str:
    text = f"{tool.signature}. {tool.document}"
    # Match the lexical units used by ToolScope's encoders: endpoint/function
    # names are readable both as whole identifiers and as component words.
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.sub(r"[_./:-]+", " ", text)


def _reranker_text(tool: ToolRecord) -> str:
    properties = tool.arguments.get("properties", {})
    schema_parts: list[str] = []
    if isinstance(properties, Mapping):
        for name, spec in properties.items():
            if isinstance(spec, Mapping):
                schema_parts.append(
                    f"{name} {spec.get('type', '')} {spec.get('description', '')}"
                )
    return f"{tool.name}. {tool.description}. {' '.join(schema_parts)}".strip()


def _lexical_tokens(text: str) -> set[str]:
    tokens = {
        token for token in re.findall(r"[a-z][a-z0-9]+", text.lower()) if len(token) > 2
    }
    aliases = {
        "dollar": "currency",
        "dollars": "currency",
        "euro": "currency",
        "euros": "currency",
        "yen": "currency",
        "pound": "unit",
        "pounds": "unit",
        "ounce": "unit",
        "ounces": "unit",
        "teaspoon": "unit",
        "teaspoons": "unit",
        "tablespoon": "unit",
        "tablespoons": "unit",
        "chance": "probability",
        "odds": "probability",
        "war": "history",
        "wars": "history",
        "historical": "history",
        "date": "event",
        "dates": "event",
    }
    tokens.update(aliases[token] for token in list(tokens) if token in aliases)
    return tokens


def _lexical_overlap(query: str, tool: ToolRecord) -> float:
    query_tokens = _lexical_tokens(query)
    if not query_tokens:
        return 0.0
    tool_tokens = _lexical_tokens(_tool_text(tool))
    return len(query_tokens & tool_tokens) / len(query_tokens)


def _normalize_no_merge_names(
    tools: Sequence[ToolRecord],
) -> tuple[list[ToolRecord], dict[str, set[str]]]:
    counts: dict[str, int] = {}
    normalized: list[ToolRecord] = []
    mapping: dict[str, set[str]] = defaultdict(set)
    for tool in tools:
        occurrence = counts.get(tool.name, 0)
        counts[tool.name] = occurrence + 1
        name = tool.name if occurrence == 0 else f"{tool.name}__variant_{occurrence}"
        if name == tool.name:
            normalized_tool = tool
        else:
            normalized_tool = ToolRecord(
                name=name,
                description=tool.description,
                field=tool.field,
                arguments=dict(tool.arguments),
                uid=tool.uid,
                required_arguments=set(tool.required_arguments),
                result_fields=set(tool.result_fields),
                original=dict(tool.original),
            )
        normalized.append(normalized_tool)
        mapping[tool.name].add(name)
    return normalized, mapping


class QueryDecomposer:
    def __init__(
        self, llm: OpenAICompatibleLLM | None, prompts: PromptStore | None
    ) -> None:
        self.llm = llm
        self.prompts = prompts

    def split(self, query: str) -> list[str]:
        if self.llm and self.llm.enabled and self.prompts:
            prompt = self.prompts.load("query_decomposition.txt").replace(
                "{query}", query
            )
            value = _parse_json_object(self.llm.complete(prompt))
            parts = value.get("subqueries") if value else None
            if isinstance(parts, list) and all(str(item).strip() for item in parts):
                return [str(item).strip() for item in parts]
            raise ValueError(
                "Query decomposer did not return a valid subqueries JSON list"
            )
        # This is only a no-network fallback.  It never uses solution labels.
        parts = re.split(
            r"\s+(?:and then|then|after that|followed by)\s+|;", query, flags=re.I
        )
        return [part.strip() for part in parts if part.strip()] or [query]


class ToolSelectionAgent:
    """ToolScope's unchanged LLM agent applied after top-k retrieval."""

    def __init__(
        self,
        llm: OpenAICompatibleLLM | None,
        prompts: PromptStore | None,
        tools: Sequence[ToolRecord],
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.by_name: dict[str, ToolRecord] = {}
        self.audit: list[dict[str, Any]] = []
        for tool in tools:
            if tool.name in self.by_name:
                raise ValueError(
                    f"Selection requires unique merged tool names; duplicate: {tool.name}"
                )
            self.by_name[tool.name] = tool

    @property
    def enabled(self) -> bool:
        return bool(self.llm and self.llm.enabled and self.prompts)

    def select(
        self,
        query: str,
        step: str,
        candidate_names: Sequence[str],
        candidate_scores: Mapping[str, float] | None = None,
    ) -> str | None:
        if not self.enabled:
            return None
        candidates: list[str] = []
        valid_names: set[str] = set()
        for rank, name in enumerate(candidate_names, start=1):
            tool = self.by_name.get(name)
            if tool is None:
                continue
            valid_names.add(name)
            candidates.append(
                "\n".join(
                    [
                        f"Candidate rank: {rank}",
                        f"Retrieval score: {float((candidate_scores or {}).get(name, 0.0)):.6f}",
                        f"Function name: {tool.name}",
                        f"Signature: {tool.signature}",
                        f"Description: {tool.description}",
                        "Arguments: " + json.dumps(tool.arguments, ensure_ascii=False),
                        "Expected results: "
                        + json.dumps(sorted(tool.result_fields), ensure_ascii=False),
                    ]
                )
            )
        prompt = self.prompts.load("agent_tool_selection.txt")
        prompt = prompt.replace("{question}", query)
        prompt = prompt.replace("{input}", step)
        prompt = prompt.replace("{tools}", "\n".join(candidates))
        assert self.llm is not None
        output = self.llm.complete(prompt).strip()
        first_line = output.splitlines()[0].strip().strip("`'") if output else ""
        self.audit.append(
            {
                "query": query,
                "step": step,
                "candidates": sorted(valid_names),
                "candidate_scores": {
                    name: float((candidate_scores or {}).get(name, 0.0))
                    for name in sorted(valid_names)
                },
                "response": output,
                "valid_response": first_line in valid_names if output else False,
            }
        )
        if first_line.lower() == "none":
            return None
        return first_line if first_line in valid_names else None


def _minmax(values: Sequence[float], epsilon: float = 1e-8) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    low, high = float(array.min()), float(array.max())
    return (array - low) / (high - low + epsilon)


class ToolScopeRetriever:
    """ToolScope-compatible retriever over original or entropy-merged tools."""

    def __init__(
        self,
        tools: Sequence[ToolRecord],
        config: Mapping[str, Any],
        *,
        strict_models: bool = False,
    ) -> None:
        self.tools = list(tools)
        if not self.tools:
            raise ValueError("Retriever requires at least one tool")
        self.config = config
        retriever_cfg = config.get("retriever", {})
        self.alpha = float(retriever_cfg.get("alpha", 1.0))
        self.candidate_m = int(retriever_cfg.get("candidate_m", 50))
        self.epsilon = float(retriever_cfg.get("epsilon", 1e-8))
        self.reranker_dense_weight = float(
            retriever_cfg.get("reranker_dense_weight", 0.5)
        )
        self.reranker_lexical_weight = float(
            retriever_cfg.get("reranker_lexical_weight", 1.0)
        )
        self.reranker_sparse_weight = float(
            retriever_cfg.get("reranker_sparse_weight", 2.0)
        )
        self.embedding_backend = "tfidf_fallback"
        self.reranker_backend = "hybrid_fallback"
        self._dense_model: Any = None
        self._reranker: Any = None
        self._faiss_index: Any = None
        documents = [_tool_text(tool) for tool in self.tools]
        model_name = str(retriever_cfg.get("embedding_model", "thenlper/gte-large"))
        try:
            from sentence_transformers import SentenceTransformer

            self._dense_model = SentenceTransformer(model_name)
            self._tool_dense = np.asarray(
                self._dense_model.encode(
                    documents, convert_to_numpy=True, normalize_embeddings=True
                ),
                dtype=float,
            )
            self.embedding_backend = f"sentence-transformers:{model_name}"
            try:
                import faiss

                self._faiss_index = faiss.IndexFlatIP(self._tool_dense.shape[1])
                self._faiss_index.add(self._tool_dense.astype("float32"))
                self.embedding_backend += "+faiss"
            except Exception:
                # Exact NumPy inner-product search has the same ranking as
                # FAISS IndexFlatIP and keeps the checkout installable without
                # a platform-specific FAISS wheel.
                self._faiss_index = None
        except Exception as exc:
            if strict_models or not bool(retriever_cfg.get("allow_fallback", True)):
                raise RuntimeError(
                    f"Could not load embedding model {model_name}; install/cache sentence-transformers or enable fallback"
                ) from exc
            self._dense_model = None
            vectorizer = TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2), sublinear_tf=True
            )
            self._fallback_vectorizer = vectorizer
            self._tool_dense = vectorizer.fit_transform(documents)
        self._sparse_vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), sublinear_tf=True
        )
        self._tool_sparse = self._sparse_vectorizer.fit_transform(documents)
        reranker_name = str(
            retriever_cfg.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L6-v2")
        )
        try:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(reranker_name)
            self.reranker_backend = f"cross-encoder:{reranker_name}"
        except Exception as exc:
            if strict_models or not bool(retriever_cfg.get("allow_fallback", True)):
                raise RuntimeError(
                    f"Could not load reranker model {reranker_name}; install/cache sentence-transformers or enable fallback"
                ) from exc

    def _dense_scores(self, query: str) -> np.ndarray:
        if self._dense_model is not None:
            vector = np.asarray(
                self._dense_model.encode(
                    [query], convert_to_numpy=True, normalize_embeddings=True
                ),
                dtype=float,
            )
            if self._faiss_index is not None:
                scores, indices = self._faiss_index.search(
                    vector.astype("float32"), len(self.tools)
                )
                dense = np.full(len(self.tools), -np.inf, dtype=float)
                dense[indices[0]] = scores[0]
                return dense
            return (vector @ self._tool_dense.T).ravel()
        vector = self._fallback_vectorizer.transform([query])
        return cosine_similarity(vector, self._tool_dense).ravel()

    def _sparse_scores(self, query: str) -> np.ndarray:
        vector = self._sparse_vectorizer.transform([query])
        return cosine_similarity(vector, self._tool_sparse).ravel()

    def _rerank(
        self,
        query: str,
        indices: np.ndarray,
        hybrid: np.ndarray,
        *,
        concise: bool = False,
    ) -> np.ndarray:
        if self._reranker is None:
            return hybrid[indices]
        pairs = [
            [
                query,
                (
                    _reranker_text(self.tools[int(index)])
                    if concise
                    else _tool_text(self.tools[int(index)])
                ),
            ]
            for index in indices
        ]
        scores = self._reranker.predict(pairs)
        return np.asarray(scores, dtype=float).ravel()

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        *,
        subqueries: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        k = int(k or self.config.get("retriever", {}).get("default_k", 5))
        if k <= 0:
            raise ValueError("k must be positive")
        queries = list(subqueries or [query])
        top_one: list[tuple[str, int, float]] = []
        remaining: list[tuple[float, str, int, float]] = []
        traces: list[dict[str, Any]] = []
        for subquery in queries:
            dense = self._dense_scores(subquery)
            sparse = self._sparse_scores(subquery)
            hybrid = self.alpha * dense + (1.0 - self.alpha) * sparse
            candidate_count = min(self.candidate_m, len(self.tools))
            dense_indices = np.argsort(-hybrid, kind="stable")[:candidate_count]
            lexical_scores = np.asarray(
                [_lexical_overlap(subquery, tool) for tool in self.tools],
                dtype=float,
            )
            lexical_indices = np.argsort(-lexical_scores, kind="stable")[
                : min(10, len(self.tools))
            ]
            merged_indices = list(
                dict.fromkeys(
                    [
                        *lexical_indices.tolist(),
                        *dense_indices.tolist(),
                    ]
                )
            )
            pre_indices = np.asarray(merged_indices[:candidate_count], dtype=int)
            candidate_reranked = self._rerank(subquery, pre_indices, hybrid)
            top1_reranked = self._rerank(subquery, pre_indices, hybrid, concise=True)
            reranker_low = float(top1_reranked.min())
            reranker_high = float(top1_reranked.max())
            reranker_normalized = (top1_reranked - reranker_low) / (
                reranker_high - reranker_low + self.epsilon
            )
            dense_values = hybrid[pre_indices]
            dense_low = float(dense_values.min())
            dense_high = float(dense_values.max())
            dense_normalized = (dense_values - dense_low) / (
                dense_high - dense_low + self.epsilon
            )
            sparse_values = sparse[pre_indices]
            sparse_low = float(sparse_values.min())
            sparse_high = float(sparse_values.max())
            sparse_normalized = (sparse_values - sparse_low) / (
                sparse_high - sparse_low + self.epsilon
            )
            lexical_values = lexical_scores[pre_indices]
            lexical_low = float(lexical_values.min())
            lexical_high = float(lexical_values.max())
            lexical_normalized = (lexical_values - lexical_low) / (
                lexical_high - lexical_low + self.epsilon
            )
            top1_scores = (
                reranker_normalized
                + self.reranker_dense_weight * dense_normalized
                + self.reranker_sparse_weight * sparse_normalized
                + self.reranker_lexical_weight * lexical_normalized
            )
            candidate_low = float(candidate_reranked.min())
            candidate_high = float(candidate_reranked.max())
            candidate_scores = (candidate_reranked - candidate_low) / (
                candidate_high - candidate_low + self.epsilon
            )
            candidate_scores = (
                candidate_scores + 0.5 * dense_normalized + lexical_normalized
            )
            order = np.argsort(-candidate_scores, kind="stable")
            best_position = int(np.argmax(top1_scores))
            best_index = int(pre_indices[best_position])
            top_one.append((subquery, best_index, float(top1_scores[best_position])))
            rest_positions = order
            normalized = _minmax(
                [float(candidate_scores[position]) for position in rest_positions],
                self.epsilon,
            )
            for score, position in zip(normalized, rest_positions):
                index = int(pre_indices[int(position)])
                remaining.append(
                    (
                        float(score),
                        subquery,
                        index,
                        float(candidate_scores[int(position)]),
                    )
                )
            traces.append(
                {
                    "subquery": subquery,
                    "candidate_m": len(pre_indices),
                    "top_one": self.tools[best_index].name,
                    "top_one_score": float(top1_scores[best_position]),
                    "reranker_scores": [
                        {
                            "tool": self.tools[int(pre_indices[int(position)])].name,
                            "score": float(candidate_scores[int(position)]),
                        }
                        for position in order
                    ],
                }
            )
        direct_top_one = [self.tools[index].name for _, index, _ in top_one]
        selected: list[int] = []
        seen: set[int] = set()
        seen_names: set[str] = set()
        remaining.sort(key=lambda item: (-item[0], self.tools[item[2]].name, item[1]))
        for _score, _subquery, index, _raw in remaining:
            if len(selected) >= k:
                break
            name = self.tools[index].name
            if index not in seen and name not in seen_names:
                selected.append(index)
                seen.add(index)
                seen_names.add(name)
        selected = selected[:k]
        return {
            "query": query,
            "subqueries": queries,
            "direct_top_one": direct_top_one,
            "selected": [self.tools[index].name for index in selected],
            "selected_scores": [
                {"tool": self.tools[index].name, "rank": rank}
                for rank, index in enumerate(selected, start=1)
            ],
            "embedding_backend": self.embedding_backend,
            "reranker_backend": self.reranker_backend,
            "alpha": self.alpha,
            "candidate_m": self.candidate_m,
            "traces": traces,
        }


def load_dataset(
    root: str | Path, dataset: str
) -> tuple[list[ToolRecord], list[dict[str, Any]]]:
    root = Path(root)
    aliases = {
        "seal": "SealTool",
        "sealtool": "SealTool",
        "bfcl": "BFCL",
        "bfcl_pm": "BFCL_PM",
    }
    directory = aliases.get(dataset.lower(), dataset)
    tools = load_tools(root / directory / f"{directory}_osa.json")
    records = _json_load(root / directory / f"{directory}_v.json")
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of query records for {dataset}")
    return tools, [record for record in records if isinstance(record, Mapping)]


def query_text(record: Mapping[str, Any]) -> str:
    value = record.get("query", "")
    if isinstance(value, str):
        return value.strip()
    # BFCL stores messages as nested turns.
    texts: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            if node.get("role") == "user" and isinstance(node.get("content"), str):
                texts.append(node["content"])
            else:
                for child in node.values():
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return " ".join(texts).strip()


def solution_names(record: Mapping[str, Any]) -> list[str]:
    values = record.get("solution", [])
    if not isinstance(values, list):
        values = [values]
    result: list[str] = []
    for value in values:
        text = str(value)
        match = FUNCTION_RE.search(text)
        if match:
            result.append(match.group(1))
        else:
            result.append(text.strip())
    return result


def evaluate_dataset(
    dataset_root: str | Path,
    dataset: str,
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    use_llm: bool = False,
    auto_correct: bool = False,
    no_merge: bool = False,
    matched_toolscope_size: bool = False,
    tool_selection_llm_id: str | None = None,
    max_queries: int | None = None,
    strict_models: bool = False,
    prompt_dir: str | Path | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    tools, records = load_dataset(dataset_root, dataset)
    if max_queries is not None:
        records = records[:max_queries]
    config = json.loads(json.dumps(config))
    if use_llm or auto_correct:
        enabled_ids = {
            "documentation_merger",
            "query_decomposer",
            tool_selection_llm_id
            or config.get("evaluation", {}).get(
                "default_tool_selection_llm_id", "tool_selection"
            ),
        }
        if auto_correct:
            enabled_ids.add("merge_autocorrection")
        for spec in config.get("llms", []):
            spec["enabled"] = spec.get("id") in enabled_ids
    prompts = PromptStore(prompt_dir) if prompt_dir else None
    merger_config = config.setdefault("merger", {})
    dataset_key = {
        "seal": "SealTool",
        "sealtool": "SealTool",
        "bfcl": "BFCL",
        "bfcl_pm": "BFCL_PM",
    }.get(dataset.lower(), dataset)
    if matched_toolscope_size and not no_merge:
        sizes = config.get("evaluation", {}).get(
            "reported_toolscope_merged_sizes_for_optional_control", {}
        )
        if dataset_key not in sizes:
            raise ValueError(f"No matched ToolScope size configured for {dataset_key}")
        merger_config["target_clusters"] = int(sizes[dataset_key])
    merger = EntroTool(tools, config, prompts)
    if no_merge:
        merger.merged_tools, merger.mapping = _normalize_no_merge_names(tools)
        merger.graph = {
            "graph_type": "identity/no-merge control",
            "nodes": len(tools),
            "edges": 0,
        }
        merger.edges = []
    else:
        merger.fit()
    decomposer = QueryDecomposer(
        OpenAICompatibleLLM(
            next(
                (
                    spec
                    for spec in config.get("llms", [])
                    if spec.get("id") == "query_decomposer"
                ),
                {},
            )
        ),
        prompts,
    )
    retriever = ToolScopeRetriever(
        merger.merged_tools, config, strict_models=strict_models
    )
    selected_llm_id = tool_selection_llm_id or config.get("evaluation", {}).get(
        "default_tool_selection_llm_id", "tool_selection"
    )
    selection_spec = next(
        (spec for spec in config.get("llms", []) if spec.get("id") == selected_llm_id),
        {},
    )
    selection_agent = ToolSelectionAgent(
        OpenAICompatibleLLM(selection_spec), prompts, merger.merged_tools
    )
    eval_cfg = config.get("evaluation", {}).get("k_values", {})
    k_values = [int(value) for value in eval_cfg.get(dataset_key, [5])]
    max_k = max(k_values)
    per_query: list[dict[str, Any]] = []
    if show_progress:
        print("\n" + "#" * 96, flush=True)
        print(f"TOOLSCOPERETRIEVER START: {dataset_key}", flush=True)
        print(
            json.dumps(
                {
                    "queries": len(records),
                    "tools_after_merger": len(merger.merged_tools),
                    "embedding_backend": retriever.embedding_backend,
                    "reranker_backend": retriever.reranker_backend,
                    "candidate_m": retriever.candidate_m,
                    "k_values": k_values,
                    "query_decomposition": (
                        "skipped: BFCL is a single-tool benchmark"
                        if dataset_key == "BFCL"
                        else "enabled"
                    ),
                    "selection_agent_enabled": selection_agent.enabled,
                    "selection_model": selection_spec.get("model"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    for query_index, record in enumerate(records, start=1):
        query = query_text(record)
        # BFCL is one query -> one tool. Passing the original query as its only
        # step is exactly equivalent and avoids an unnecessary decomposition LLM
        # call. BFCL_PM and SealTool retain ToolScope's multi-step decomposition.
        subqueries = [query] if dataset_key == "BFCL" else decomposer.split(query)
        if show_progress:
            print("\n" + "=" * 96, flush=True)
            print(
                f"[ToolScopeRetriever] {dataset_key} {query_index}/{len(records)}",
                flush=True,
            )
            print(f"Query: {query}", flush=True)
            if dataset_key == "BFCL":
                print("Query decomposition: SKIPPED (single-tool BFCL)", flush=True)
            else:
                print(
                    "Subqueries: " + json.dumps(subqueries, ensure_ascii=False),
                    flush=True,
                )
        retrieval = retriever.retrieve(query, max_k, subqueries=subqueries)
        predicted = retrieval["selected"]
        if show_progress:
            print(
                f"Retrieved Top-{max_k}:\n"
                + "\n".join(
                    f"  {rank:>2}. {name}"
                    for rank, name in enumerate(predicted[:max_k], start=1)
                ),
                flush=True,
            )
        # ToolScope's CSR is measured after its LLM Agent sees the retrieved
        # top-k tools. Run that unchanged prompt before gold is parsed.
        agent_predictions: dict[str, list[str]] = {}
        if selection_agent.enabled:
            for k in k_values:
                chosen: list[str] = []
                candidates = predicted[:k]
                if show_progress:
                    print(
                        f"[DeepSeek Tool Selection] k={k}, selecting from {len(candidates)} candidates...",
                        flush=True,
                    )
                for step in subqueries:
                    selected_name = selection_agent.select(query, step, candidates)
                    if selected_name and selected_name not in chosen:
                        chosen.append(selected_name)
                agent_predictions[str(k)] = chosen
                if show_progress:
                    print(
                        f"[DeepSeek Tool Selection] k={k}, final selected: "
                        + json.dumps(chosen, ensure_ascii=False),
                        flush=True,
                    )
        elif show_progress:
            print(
                "[Tool Selection Agent] DISABLED: no model-selected tool; CSR will be null. Use --use-llm.",
                flush=True,
            )
        # Strict post-generation boundary: only after retrieval has finished do
        # we parse and relabel the benchmark solution for metrics. The solution
        # is unavailable to the merger, retriever, decomposer and selection agent.
        original_gold = solution_names(record)
        gold: list[str] = []
        for name in original_gold:
            mapped = merger.mapping.get(name, {name})
            if isinstance(mapped, (set, list, tuple)):
                gold.extend(str(value) for value in mapped)
            else:
                gold.append(str(mapped))
        metrics: dict[str, Any] = {}
        for k in k_values:
            predicted_k = predicted[:k]
            gold_set, predicted_set = set(gold), set(predicted_k)
            hits = len(gold_set & predicted_set)
            selected = agent_predictions.get(str(k), predicted_k[:1])
            selection_scores = tool_selection_metrics(selected, [gold_set])
            metrics[str(k)] = {
                "success": None,
                "path": selection_scores["path"],
                "csr": selection_scores["csr"],
                "candidate_recall": hits / len(gold_set) if gold_set else 1.0,
                "candidate_contains_all_gold": gold_set.issubset(predicted_set),
                "hits": sorted(gold_set & predicted_set),
                "missing": sorted(gold_set - predicted_set),
                "retrieved_candidates": predicted_k,
                "agent_selected": selected,
            }
        if show_progress:
            # Gold is intentionally printed only here, after retrieval and every
            # LLM selection call have completed, so console observability cannot
            # turn into label leakage.
            print(
                "Gold solution (original): "
                + json.dumps(original_gold, ensure_ascii=False),
                flush=True,
            )
            print(
                "Gold solution (after merger mapping): "
                + json.dumps(gold, ensure_ascii=False),
                flush=True,
            )
            for k in k_values:
                result_at_k = metrics[str(k)]
                print(
                    f"Result@{k}: agent_selected="
                    + json.dumps(result_at_k["agent_selected"], ensure_ascii=False)
                    + f", CSR={result_at_k['csr']}, "
                    + f"candidate_recall={result_at_k['candidate_recall']:.6f}",
                    flush=True,
                )
        per_query.append(
            {
                "query": query,
                "gold_original": original_gold,
                "gold_after_mapping": gold,
                "retrieval": retrieval,
                "agent_predictions": agent_predictions,
                "metrics": metrics,
            }
        )
    summary: dict[str, Any] = {
        "dataset": dataset_key,
        "query_count": len(per_query),
        "input_tool_count": len(tools),
        "merged_tool_count": len(merger.merged_tools),
        "merge_reduction_percent": round(
            100 * (1 - len(merger.merged_tools) / len(tools)), 6
        ),
        "k_values": k_values,
        "metrics": {
            str(k): {
                "success": None,
                "path": round(
                    (
                        float(
                            np.mean(
                                [row["metrics"][str(k)]["path"] for row in per_query]
                            )
                        )
                        if per_query
                        else 0.0
                    ),
                    6,
                ),
                "csr": round(
                    (
                        float(
                            np.mean(
                                [row["metrics"][str(k)]["csr"] for row in per_query]
                            )
                        )
                        if per_query
                        else 0.0
                    ),
                    6,
                ),
            }
            for k in k_values
        },
        "metric_definitions": {
            "success": "Pending external LLM judge; the judge sees only query and selected tool documents.",
            "path": "Mean fraction of mapped gold tools covered by the final selected tool set.",
            "csr": "Fraction of queries whose final selected tool set exactly equals mapped gold.",
        },
        "retriever": {
            "embedding_model": config.get("retriever", {}).get("embedding_model"),
            "reranker_model": config.get("retriever", {}).get("reranker_model"),
            "alpha": config.get("retriever", {}).get("alpha", 1.0),
            "candidate_m": config.get("retriever", {}).get("candidate_m", 50),
            "backend": retriever.embedding_backend,
            "reranker_backend": retriever.reranker_backend,
        },
        "tool_selection_agent": {
            "enabled": selection_agent.enabled,
            "llm_id": selected_llm_id,
            "model": selection_spec.get("model"),
            "prompt": "prompts/agent_tool_selection.txt",
        },
        "query_decomposition": {
            "used": dataset_key != "BFCL",
            "reason": (
                "BFCL is single-tool; original query is the only plan step"
                if dataset_key == "BFCL"
                else "multi-tool benchmark uses ordered subqueries"
            ),
        },
        "gold_usage": (
            "Gold solution names are parsed only after merging, retrieval, "
            "query decomposition and LLM tool selection have completed."
        ),
        "merger": {
            "method": (
                "KG structural entropy" if not no_merge else "identity/no-merge control"
            ),
            "kg_edges": len(merger.edges),
            "pruning_graph_components": merger.pruning_graph.get("component_count"),
            "component_llm_pipeline": {
                "documentation_merger_enabled": bool(
                    merger._llm("documentation_merger")
                    and merger._llm("documentation_merger").enabled
                ),
                "auto_correction_enabled": bool(
                    merger._llm("merge_autocorrection")
                    and merger._llm("merge_autocorrection").enabled
                ),
            },
            "clusters": merger.clusters,
            "mapping": {
                str(key): sorted(value) if isinstance(value, set) else value
                for key, value in merger.mapping.items()
            },
        },
        "per_query": per_query,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _json_dump(destination / "evaluation.json", summary)
    _json_dump(
        destination / "merged_tools.json",
        [tool.to_dict() for tool in merger.merged_tools],
    )
    _json_dump(
        destination / "tool_mapping.json",
        {
            str(key): sorted(value) if isinstance(value, set) else value
            for key, value in merger.mapping.items()
        },
    )
    _json_dump(destination / "knowledge_graph.json", merger.graph)
    if merger.pruning_graph:
        _json_dump(destination / "entropy_pruning_graph.json", merger.pruning_graph)
    if merger.entropy is not None:
        _json_dump(
            destination / "structural_entropy_codebook.json", merger.entropy.to_dict()
        )
    _json_dump(
        destination / "updated_queries.json",
        [
            {"query": row["query"], "solution": row["gold_after_mapping"]}
            for row in per_query
        ],
    )
    return summary
