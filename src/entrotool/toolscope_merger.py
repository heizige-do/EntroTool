"""ToolScopeMerger with a structural-entropy graph-construction replacement.

The implementation deliberately keeps ToolScopeRetriever's post-merger contract:
query decomposition, dense/hybrid retrieval, top-M reranking, min-max score
normalization, and top-k assembly.  Only the merger's source of merge evidence is
changed. ToolScope uses cosine-neighbor candidates plus an LLM classifier that
creates a binary graph. This module uses the same dense encoder, keeps every
all-pairs cosine score strictly above the threshold as a continuous semantic
edge weight, and performs agglomerative two-dimensional structural-entropy
clustering.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

FUNCTION_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
DOMAIN_STOPWORDS = {
    "about",
    "based",
    "calculate",
    "calculates",
    "data",
    "details",
    "detailed",
    "find",
    "given",
    "get",
    "information",
    "locate",
    "location",
    "nearby",
    "provide",
    "retrieve",
    "retrieves",
    "search",
    "specific",
    "specified",
    "using",
    "used",
    "value",
    "values",
    "with",
}
FUNCTION_STOPWORDS = DOMAIN_STOPWORDS | {
    "api",
    "call",
    "function",
    "tool",
    "returns",
    "return",
    "input",
    "output",
    "request",
    "response",
    "result",
    "object",
    "method",
}
ACTION_GROUPS = {
    "read": {
        "read",
        "get",
        "fetch",
        "find",
        "lookup",
        "list",
        "search",
        "retrieve",
        "show",
    },
    "calculate": {"calculate", "calculates", "compute", "convert", "estimate", "solve"},
    "create": {"add", "book", "create", "insert", "make", "register"},
    "update": {"edit", "modify", "patch", "set", "update"},
    "delete": {"cancel", "delete", "remove"},
}
ACTION_BY_WORD = {
    word: group for group, words in ACTION_GROUPS.items() for word in words
}
RESULT_STOPWORDS = FUNCTION_STOPWORDS | {
    "available",
    "availability",
    "information",
    "item",
    "items",
    "status",
}
ENGLISH_STOPWORDS = {
    "a",
    "an",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "those",
    "to",
    "using",
    "was",
    "were",
    "which",
    "will",
    "and",
    "but",
    "can",
    "if",
    "its",
    "than",
    "then",
    "these",
    "with",
}


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(value).strip()).strip("_")


def normalize_field(value: str) -> str:
    value = normalize_name(value).lower()
    value = re.sub(r"_+", "_", value)
    return value


def _json_load(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        # Some imported benchmark snapshots contain literal control characters
        # inside descriptions. Keep the benchmark text and parse permissively.
        return json.load(handle, strict=False)


def _json_dump(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _nested_property_names(value: Any) -> set[str]:
    """Collect leaf and intermediate property names from OpenAPI-like JSON."""

    names: set[str] = set()
    if isinstance(value, Mapping):
        properties = value.get("properties")
        if isinstance(properties, Mapping):
            for raw_name, child in properties.items():
                name = normalize_field(str(raw_name))
                if name:
                    names.add(name)
                names.update(_nested_property_names(child))
        for key, child in value.items():
            if key not in {"properties", "items"}:
                if isinstance(child, (Mapping, list)):
                    names.update(_nested_property_names(child))
        if isinstance(value.get("items"), (Mapping, list)):
            names.update(_nested_property_names(value["items"]))
    elif isinstance(value, list):
        for child in value[:1]:
            names.update(_nested_property_names(child))
    return names


def _argument_names(raw: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    arguments = raw.get("arguments") or {}
    properties = arguments.get("properties") if isinstance(arguments, Mapping) else {}
    if not isinstance(properties, Mapping):
        properties = {}
    names = {normalize_field(str(name)) for name in properties if str(name).strip()}
    required = raw.get("required")
    if not isinstance(required, list) and isinstance(arguments, Mapping):
        required = arguments.get("required")
    required_names = {
        normalize_field(str(name)) for name in (required or []) if str(name).strip()
    }
    return names, required_names


def _parameter_entities(name: str) -> set[str]:
    return {token.lower() for token in re.split(r"[^A-Za-z0-9]+", str(name)) if token}


@dataclass
class ToolRecord:
    """Dataset-independent normalized tool representation."""

    name: str
    description: str
    field: str
    arguments: dict[str, Any]
    uid: str = ""
    required_arguments: set[str] = field(default_factory=set)
    result_fields: set[str] = field(default_factory=set)
    original: dict[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> str:
        arguments = self.arguments.get("properties", {})
        if not isinstance(arguments, Mapping):
            arguments = {}
        parts: list[str] = []
        for name, spec in arguments.items():
            if isinstance(spec, Mapping):
                typ = spec.get("type", "any")
            else:
                typ = "any"
            suffix = (
                ""
                if normalize_field(str(name)) in self.required_arguments
                else " = None"
            )
            parts.append(f"{name}: {typ}{suffix}")
        return f"{self.name}({', '.join(parts)})"

    @property
    def document(self) -> str:
        pieces = [self.name, self.description, self.field]
        pieces.extend(sorted(self.required_arguments))
        properties = self.arguments.get("properties", {})
        if isinstance(properties, Mapping):
            pieces.extend(str(key) for key in sorted(properties))
        pieces.extend(sorted(self.result_fields))
        return " ".join(piece for piece in pieces if piece).strip()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["required_arguments"] = sorted(self.required_arguments)
        result["result_fields"] = sorted(self.result_fields)
        result["signature"] = self.signature
        result["document"] = self.document
        return result


def normalize_tool(raw: Mapping[str, Any]) -> ToolRecord:
    name = str(raw.get("name") or raw.get("path") or "").strip()
    if not name:
        raise ValueError("A tool must contain a non-empty name")
    arguments = raw.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = {"type": "object", "properties": {}}
    _args, required = _argument_names(raw)
    result_obj = raw.get("results") or raw.get("response") or raw.get("result") or {}
    result_fields = _nested_property_names(result_obj)
    description_parts = [
        str(raw.get("description", "")),
        str(raw.get("usage", "")),
        str(raw.get("functionality", "")),
    ]
    description = " ".join(part.strip() for part in description_parts if part.strip())
    return ToolRecord(
        name=name,
        description=description,
        field=str(raw.get("field", "") or "").strip(),
        arguments={str(k): v for k, v in arguments.items()},
        required_arguments=required,
        result_fields=result_fields,
        original=dict(raw),
    )


def load_tools(path: str | Path) -> list[ToolRecord]:
    raw = _json_load(path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of tools in {path}")
    tools = [normalize_tool(item) for item in raw if isinstance(item, Mapping)]
    if not tools:
        raise ValueError(f"No tools found in {path}")
    # BFCL intentionally contains overloaded/repeated function names. Keep the
    # benchmark-facing name, but assign a stable UID for graph/node identity.
    for index, tool in enumerate(tools):
        tool.uid = f"{tool.name}@@{index:05d}"
    return tools


@dataclass(frozen=True)
class KGEdge:
    source: str
    target: str
    weight: float
    relations: dict[str, Any]
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": round(self.weight, 8),
            "relations": {
                key: round(value, 8) for key, value in self.relations.items()
            },
            "evidence": list(self.evidence),
        }


class KnowledgeGraphBuilder:
    """Build the weighted semantic tool graph used by structural entropy.

        ToolScope's own ``thenlper/gte-large`` encoder is retained.  The difference
        is that cosine scores are preserved as continuous graph-edge weights and
        passed to structural entropy, rather than thresholded and classified into a
    binary graph by a pairwise LLM. This replacement evaluates all tool pairs and
    retains exactly those whose cosine is strictly above the configured threshold.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        encoder: Any | None = None,
    ) -> None:
        config = config or {}
        self.config = config
        semantic = config.get("semantic_graph", config)
        self.embedding_model = str(
            semantic.get("embedding_model", "thenlper/gte-large")
        )
        threshold = semantic.get("minimum_cosine")
        if threshold is None:
            raise ValueError("semantic_graph.minimum_cosine must be configured")
        self.minimum_cosine = float(threshold)
        self.batch_size = int(semantic.get("batch_size", 64))
        self.search_chunk_size = int(semantic.get("search_chunk_size", 256))
        self.device = semantic.get("device")
        self.show_progress = bool(semantic.get("show_progress", True))
        self.use_faiss = bool(semantic.get("use_faiss", True))
        contract = semantic.get("contract_filter", {})
        self.contract_filter_enabled = bool(contract.get("enabled", True))
        self.minimum_required_jaccard = float(
            contract.get("minimum_required_jaccard", 0.25)
        )
        semantic_fallback = contract.get("semantic_contract_fallback", {})
        self.semantic_fallback_enabled = bool(semantic_fallback.get("enabled", True))
        self.semantic_fallback_cosine = float(
            semantic_fallback.get("minimum_cosine", 0.88)
        )
        self._encoder = encoder
        if not -1.0 <= self.minimum_cosine <= 1.0:
            raise ValueError("semantic_graph.minimum_cosine must be in [-1, 1]")
        if self.search_chunk_size <= 0 or self.batch_size <= 0:
            raise ValueError("semantic graph batch/chunk sizes must be positive")
        if not 0.0 <= self.minimum_required_jaccard <= 1.0:
            raise ValueError(
                "contract_filter.minimum_required_jaccard must be in [0, 1]"
            )

    @staticmethod
    def _contract_features(
        left: ToolRecord, right: ToolRecord
    ) -> tuple[float, float, str]:
        left_properties = left.arguments.get("properties", {})
        right_properties = right.arguments.get("properties", {})
        left_properties = (
            left_properties if isinstance(left_properties, Mapping) else {}
        )
        right_properties = (
            right_properties if isinstance(right_properties, Mapping) else {}
        )
        left_required = left.required_arguments or {
            str(name) for name in left_properties if str(name).strip()
        }
        right_required = right.required_arguments or {
            str(name) for name in right_properties if str(name).strip()
        }
        left_entities = (
            set().union(*(_parameter_entities(name) for name in left_required))
            if left_required
            else set()
        )
        right_entities = (
            set().union(*(_parameter_entities(name) for name in right_required))
            if right_required
            else set()
        )
        contract_source = (
            "required_argument_entities"
            if left.required_arguments and right.required_arguments
            else "properties_fallback"
        )
        union = left_entities | right_entities
        required_jaccard = (
            len(left_entities & right_entities) / len(union) if union else 1.0
        )
        left_words = {
            word.lower()
            for word in re.findall(
                r"[A-Za-z][A-Za-z0-9_]+", f"{left.name} {left.description}"
            )
            if len(word) >= 4 and word.lower() not in DOMAIN_STOPWORDS
        }
        right_words = {
            word.lower()
            for word in re.findall(
                r"[A-Za-z][A-Za-z0-9_]+", f"{right.name} {right.description}"
            )
            if len(word) >= 4 and word.lower() not in DOMAIN_STOPWORDS
        }
        domain_overlap = (
            len(left_words & right_words) / len(left_words | right_words)
            if left_words | right_words
            else 1.0
        )
        return (
            required_jaccard,
            domain_overlap,
            contract_source,
        )

    @staticmethod
    def tool_document(tool: ToolRecord) -> str:
        """Match ToolScope's signature plus natural-language description input."""

        description = tool.description.strip()
        return f"{tool.signature} — {description}" if description else tool.signature

    def _encode(self, documents: Sequence[str]) -> np.ndarray:
        encoder = self._encoder
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "ToolScopeMerger requires sentence-transformers and "
                    f"{self.embedding_model!r}; install requirements before running"
                ) from exc
            kwargs: dict[str, Any] = {}
            if self.device:
                kwargs["device"] = self.device
            encoder = SentenceTransformer(self.embedding_model, **kwargs)
        if hasattr(encoder, "encode"):
            values = encoder.encode(
                list(documents),
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        elif callable(encoder):
            values = encoder(list(documents))
        else:
            raise TypeError("encoder must be callable or expose encode()")
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(documents):
            raise ValueError("embedding encoder returned an invalid matrix shape")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("embedding encoder returned a zero vector")
        return matrix / norms

    def _threshold_pairs(self, embeddings: np.ndarray) -> list[tuple[int, int, float]]:
        """Return every unordered pair with cosine strictly above threshold."""

        count = embeddings.shape[0]
        if count <= 1:
            return []

        if self.use_faiss:
            try:
                import faiss

                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
                limits, scores, indices = index.range_search(
                    np.ascontiguousarray(embeddings, dtype=np.float32),
                    self.minimum_cosine,
                )
                pairs: list[tuple[int, int, float]] = []
                for source in range(count):
                    for position in range(int(limits[source]), int(limits[source + 1])):
                        score = float(scores[position])
                        target = int(indices[position])
                        if source >= target or score <= self.minimum_cosine:
                            continue
                        pairs.append((source, target, score))
                pairs.sort(key=lambda item: (item[0], item[1]))
                return pairs
            except (ImportError, AttributeError):
                pass

        # Exact chunked all-pairs cosine. Each block is discarded immediately,
        # so even SealTool never materializes the full n x n matrix.
        pairs: list[tuple[int, int, float]] = []
        for start in range(0, count, self.search_chunk_size):
            stop = min(start + self.search_chunk_size, count)
            scores = embeddings[start:stop] @ embeddings.T
            for local, source in enumerate(range(start, stop)):
                targets = np.flatnonzero(scores[local] > self.minimum_cosine)
                for target in targets:
                    target = int(target)
                    if source >= target:
                        continue
                    pairs.append((source, target, float(scores[local, target])))
        return pairs

    def build(self, tools: Sequence[ToolRecord]) -> tuple[dict[str, Any], list[KGEdge]]:
        tools = list(tools)
        for index, tool in enumerate(tools):
            if not tool.uid:
                tool.uid = f"{tool.name}@@{index:05d}"
        documents = [self.tool_document(tool) for tool in tools]
        embeddings = self._encode(documents)
        qualified_pairs = self._threshold_pairs(embeddings)
        edges: list[KGEdge] = []
        rejected_contract_pairs = 0
        contract_pairs: list[tuple[int, int, float, float, float, str, bool]] = []
        for source, target, score in qualified_pairs:
            (
                required_jaccard,
                domain_overlap,
                contract_source,
            ) = self._contract_features(tools[source], tools[target])
            same_name = tools[source].name.casefold() == tools[target].name.casefold()
            semantic_contract_ok = (
                self.semantic_fallback_enabled and score > self.semantic_fallback_cosine
            )
            contract_ok = not self.contract_filter_enabled or (
                same_name
                or semantic_contract_ok
                or required_jaccard >= self.minimum_required_jaccard
            )
            if contract_ok:
                contract_pairs.append(
                    (
                        source,
                        target,
                        score,
                        required_jaccard,
                        domain_overlap,
                        contract_source,
                        semantic_contract_ok,
                    )
                )
            else:
                rejected_contract_pairs += 1
        qualified_degrees = [0] * len(tools)
        for (
            source,
            target,
            score,
            required_jaccard,
            domain_overlap,
            contract_source,
            semantic_contract_ok,
        ) in contract_pairs:
            qualified_degrees[source] += 1
            qualified_degrees[target] += 1
            edges.append(
                KGEdge(
                    source=tools[source].uid,
                    target=tools[target].uid,
                    weight=float(score),
                    relations={
                        "semantic_cosine": float(score),
                        "required_argument_jaccard": required_jaccard,
                        "domain_word_overlap": domain_overlap,
                        "semantic_contract_fallback": semantic_contract_ok,
                    },
                    evidence=(
                        f"encoder={self.embedding_model}",
                        f"cosine>{self.minimum_cosine}",
                        f"required_argument_jaccard>={self.minimum_required_jaccard}",
                        f"contract_source={contract_source}",
                        "candidate_generation=all_unordered_tool_pairs",
                    ),
                )
            )
        graph = {
            "graph_type": "weighted semantic tool graph",
            "nodes": len(tools),
            "edges": len(edges),
            "embedding_model": self.embedding_model,
            "document_format": "function signature — natural-language description",
            "candidate_generation": "all unordered tool pairs",
            "evaluated_pair_count": len(tools) * max(0, len(tools) - 1) // 2,
            "minimum_cosine": self.minimum_cosine,
            "edge_condition": "cosine > minimum_cosine",
            "edge_weight": "cosine similarity of L2-normalized tool embeddings",
            "contract_filter": {
                "enabled": self.contract_filter_enabled,
                "minimum_required_jaccard": self.minimum_required_jaccard,
                "semantic_contract_fallback": {
                    "enabled": self.semantic_fallback_enabled,
                    "minimum_cosine": self.semantic_fallback_cosine,
                },
                "rejected_candidate_pairs": rejected_contract_pairs,
            },
            "nodes_detail": [
                {
                    "id": f"tool:{index:05d}",
                    "kind": "tool",
                    "name": tool.name,
                    "uid": tool.uid,
                    "document": documents[index],
                    "embedding_norm": round(
                        float(np.linalg.norm(embeddings[index])), 8
                    ),
                    "qualified_edge_degree": qualified_degrees[index],
                }
                for index, tool in enumerate(tools)
            ],
            "edges_detail": [edge.to_dict() for edge in edges],
        }
        return graph, edges


class StructuralEntropyMerger:
    """Sparse agglomerative two-dimensional structural-entropy merger."""

    def __init__(
        self,
        tools: Sequence[ToolRecord],
        edges: Sequence[KGEdge],
        config: Mapping[str, Any] | None = None,
    ) -> None:
        config = config or {}
        self.tools = list(tools)
        self.edges = list(edges)
        self.min_gain = float(config.get("min_structural_gain", 1e-9))
        self.priority_mode = str(
            config.get("merge_priority", "edge_weight_then_structural_gain")
        )
        if self.priority_mode not in {
            "edge_weight_then_structural_gain",
            "structural_gain",
        }:
            raise ValueError(
                "merge_priority must be 'edge_weight_then_structural_gain' "
                "or 'structural_gain'"
            )
        target = config.get("target_clusters")
        if target is None and config.get("target_reduction_ratio") is not None:
            target = round(
                len(self.tools) * (1.0 - float(config["target_reduction_ratio"]))
            )
        self.target_clusters = int(target) if target is not None else None
        max_size = config.get("max_cluster_size")
        self.max_cluster_size = int(max_size) if max_size is not None else None
        self.min_cluster_size = int(config.get("min_cluster_size", 2))
        admissibility = config.get("merge_admissibility", {})
        self.admissibility_enabled = bool(admissibility.get("enabled", False))
        refinement = config.get("automatic_refinement", {})
        self.refinement_enabled = bool(refinement.get("enabled", False))
        self.minimum_pair_cosine = float(refinement.get("minimum_pair_cosine", 0.88))
        self.refinement_action_compatibility = bool(
            refinement.get("enforce_action_compatibility", True)
        )
        self._uid_to_index = {
            tool.uid or tool.name: i for i, tool in enumerate(self.tools)
        }
        self.tree: dict[str, dict[str, Any]] = {}
        self.report_partition: list[set[int]] = []
        self.entropy_1d = 0.0
        self.final_entropy = 0.0
        self.merge_history: list[dict[str, Any]] = []
        self._labels: dict[int, str] = {}
        self._semantic_fallback_edge_keys: set[tuple[int, int]] = set()
        self.root_id = ""
        self.refinement_stats: dict[str, Any] = {}

    @staticmethod
    def _edge_key(left: int, right: int) -> tuple[int, int]:
        return (left, right) if left <= right else (right, left)

    @staticmethod
    def _function_anchors(
        tool: ToolRecord,
    ) -> tuple[set[str], set[str], set[str]]:
        text = f"{tool.name} {tool.description}"
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower()
        tokens = set(re.findall(r"[a-z][a-z0-9]+", text.replace("_", " ")))
        actions = {ACTION_BY_WORD[token] for token in tokens if token in ACTION_BY_WORD}
        resource = {
            token
            for token in tokens
            if token not in FUNCTION_STOPWORDS
            and token not in ENGLISH_STOPWORDS
            and token not in ACTION_BY_WORD
            and len(token) > 2
        }
        result = {
            normalize_field(str(token))
            for token in tool.result_fields
            if normalize_field(str(token)) not in RESULT_STOPWORDS
        }
        return resource, actions, result

    def _anchor_compatible(
        self,
        left: int,
        right: int,
        *,
        enforce_action_compatibility: bool,
    ) -> bool:
        if (
            self.tools[left].description.strip().lower()
            == self.tools[left].name.strip().lower()
            and self.tools[right].description.strip().lower()
            == self.tools[right].name.strip().lower()
        ):
            return True
        left_anchors, left_actions, left_results = self._function_anchors(
            self.tools[left]
        )
        right_anchors, right_actions, right_results = self._function_anchors(
            self.tools[right]
        )
        if (
            enforce_action_compatibility
            and left_actions
            and right_actions
            and left_actions.isdisjoint(right_actions)
        ):
            return False
        return True

    @staticmethod
    def _contribution(
        volume: float, internal_twice: float, dlogd: float, total: float
    ) -> float:
        if volume <= 0 or total <= 0:
            return 0.0
        boundary = max(0.0, volume - internal_twice)
        value = 0.0
        if boundary > 0:
            value -= (boundary / total) * math.log2(volume / total)
        value -= (dlogd - volume * math.log2(volume)) / total
        return value

    def fit(self) -> "StructuralEntropyMerger":
        n = len(self.tools)
        adjacency: dict[int, dict[int, float]] = {i: {} for i in range(n)}
        for edge in self.edges:
            left = self._uid_to_index[edge.source]
            right = self._uid_to_index[edge.target]
            adjacency[left][right] = adjacency[left].get(right, 0.0) + edge.weight
            adjacency[right][left] = adjacency[right].get(left, 0.0) + edge.weight
            if edge.relations.get("semantic_contract_fallback"):
                self._semantic_fallback_edge_keys.add(self._edge_key(left, right))
        degrees = np.array([sum(adjacency[i].values()) for i in range(n)], dtype=float)
        admissible_neighbors: dict[int, set[int]] = {index: set() for index in range(n)}
        if self.admissibility_enabled:
            admissible_neighbors = {
                index: set(neighbors) for index, neighbors in adjacency.items()
            }
        total = float(degrees.sum())
        if total <= 0:
            self.report_partition = [{i} for i in range(n)]
            self.final_entropy = self.entropy_1d = 0.0
            return self
        dlogd_values = np.zeros_like(degrees)
        positive_degrees = degrees > 0
        dlogd_values[positive_degrees] = degrees[positive_degrees] * np.log2(
            degrees[positive_degrees]
        )
        initial_dlogd = {i: float(dlogd_values[i]) for i in range(n)}
        self.entropy_1d = float(
            -np.sum(
                (degrees[degrees > 0] / total) * np.log2(degrees[degrees > 0] / total)
            )
        )
        active: set[int] = set(range(n))
        members: dict[int, set[int]] = {i: {i} for i in range(n)}
        volume: dict[int, float] = {i: float(degrees[i]) for i in range(n)}
        internal: dict[int, float] = {i: 0.0 for i in range(n)}
        dlogd: dict[int, float] = dict(initial_dlogd)
        current = self.entropy_1d
        heap: list[tuple[float, float, str, int, int]] = []

        def contribution(cluster: int) -> float:
            return self._contribution(
                volume[cluster], internal[cluster], dlogd[cluster], total
            )

        def push_candidate(left: int, right: int) -> None:
            if left not in active or right not in active:
                return
            if right not in adjacency[left]:
                return
            semantic_equivalence = all(
                self._edge_key(node_left, node_right)
                in self._semantic_fallback_edge_keys
                for node_left in members[left]
                for node_right in members[right]
            )
            if (
                self.admissibility_enabled
                and not semantic_equivalence
                and not all(
                    members[right] <= admissible_neighbors[node]
                    for node in members[left]
                )
            ):
                return
            if (
                self.max_cluster_size is not None
                and len(members[left]) + len(members[right]) > self.max_cluster_size
            ):
                return
            cross = adjacency[left][right]
            merged_volume = volume[left] + volume[right]
            merged_internal = internal[left] + internal[right] + 2.0 * cross
            merged_dlogd = dlogd[left] + dlogd[right]
            after = (
                current
                - contribution(left)
                - contribution(right)
                + self._contribution(
                    merged_volume, merged_internal, merged_dlogd, total
                )
            )
            gain = current - after
            a, b = sorted((left, right))
            average_cross_weight = cross / (len(members[left]) * len(members[right]))
            if self.priority_mode == "edge_weight_then_structural_gain":
                priority = (-average_cross_weight, -gain)
            else:
                priority = (-gain, -average_cross_weight)
            heappush(
                heap,
                (*priority, f"{a:08d}:{b:08d}", a, b),
            )

        for left in range(n):
            for right in adjacency[left]:
                if left < right:
                    push_candidate(left, right)

        merge_index = 0
        while len(active) > 1:
            if self.target_clusters is not None and len(active) <= self.target_clusters:
                break
            if not heap:
                break
            _primary, _secondary, _tie, left, right = heappop(heap)
            if (
                left not in active
                or right not in active
                or right not in adjacency[left]
            ):
                continue
            if (
                self.max_cluster_size is not None
                and len(members[left]) + len(members[right]) > self.max_cluster_size
            ):
                continue
            semantic_equivalence = all(
                self._edge_key(node_left, node_right)
                in self._semantic_fallback_edge_keys
                for node_left in members[left]
                for node_right in members[right]
            )
            if (
                self.admissibility_enabled
                and not semantic_equivalence
                and not all(
                    members[right] <= admissible_neighbors[node]
                    for node in members[left]
                )
            ):
                continue
            cross = adjacency[left][right]
            merged_volume = volume[left] + volume[right]
            merged_internal = internal[left] + internal[right] + 2.0 * cross
            merged_dlogd = dlogd[left] + dlogd[right]
            after = (
                current
                - contribution(left)
                - contribution(right)
                + self._contribution(
                    merged_volume, merged_internal, merged_dlogd, total
                )
            )
            gain = current - after
            if gain < self.min_gain:
                if self.priority_mode == "structural_gain":
                    # In gain-first mode, the first valid sub-threshold pair
                    # certifies that no admissible candidate has a larger gain.
                    break
                # Weight-first ordering does not imply gain ordering. Skip this
                # pair and continue looking for an entropy-reducing candidate.
                continue

            new_id = n + merge_index
            merge_index += 1
            left_name = self.tree.get(f"cluster:{left - n:05d}", {}).get(
                "id", f"tool:{left:05d}"
            )
            right_name = self.tree.get(f"cluster:{right - n:05d}", {}).get(
                "id", f"tool:{right:05d}"
            )
            # The lookup above is only for readability; merged node names are
            # assigned below from the active-node labels.
            labels = getattr(self, "_labels", {})
            left_name = labels.get(left, f"tool:{left:05d}")
            right_name = labels.get(right, f"tool:{right:05d}")
            new_name = f"cluster:{merge_index - 1:05d}"
            labels[new_id] = new_name
            self._labels = labels
            merged_members = members[left] | members[right]
            self.tree[new_name] = {
                "id": new_name,
                "kind": "cluster",
                "children": [left_name, right_name],
                "leaves": [self.tools[index].name for index in sorted(merged_members)],
                "size": len(merged_members),
                "cross_weight": round(cross, 8),
                "entropy_before_merge": round(current, 8),
                "entropy_after_merge": round(after, 8),
                "structural_gain": round(gain, 8),
            }
            neighbors = set(adjacency[left]) | set(adjacency[right])
            neighbors.discard(left)
            neighbors.discard(right)
            adjacency[new_id] = {}
            for neighbor in neighbors:
                weight = adjacency[left].get(neighbor, 0.0) + adjacency[right].get(
                    neighbor, 0.0
                )
                if weight <= 0 or neighbor not in active:
                    continue
                adjacency[new_id][neighbor] = weight
                adjacency[neighbor].pop(left, None)
                adjacency[neighbor].pop(right, None)
                adjacency[neighbor][new_id] = weight
            adjacency[left].clear()
            adjacency[right].clear()
            active.remove(left)
            active.remove(right)
            active.add(new_id)
            members[new_id] = merged_members
            volume[new_id] = merged_volume
            internal[new_id] = merged_internal
            dlogd[new_id] = merged_dlogd
            del (
                members[left],
                members[right],
                volume[left],
                volume[right],
                internal[left],
                internal[right],
                dlogd[left],
                dlogd[right],
            )
            current = after
            self.merge_history.append(dict(self.tree[new_name]))
            for neighbor in neighbors:
                if neighbor in active:
                    push_candidate(new_id, neighbor)

        self.final_entropy = current
        self.report_partition = [set(members[index]) for index in sorted(active)]
        self.root_id = next(
            iter(self._labels.get(index, f"tool:{index:05d}") for index in active), ""
        )
        return self

    def cluster_sets(self) -> list[set[int]]:
        return [
            cluster
            for cluster in self.report_partition
            if len(cluster) >= self.min_cluster_size
        ]

    def pruning_graph(self) -> tuple[dict[str, Any], list[set[int]]]:
        """Turn the entropy partition into ToolScope's pruning-graph contract."""

        partition_of: dict[int, int] = {}
        for partition_id, cluster in enumerate(self.report_partition):
            for index in cluster:
                partition_of[index] = partition_id
        adjacency: dict[int, set[int]] = {
            index: set() for index in range(len(self.tools))
        }
        edge_lookup: dict[tuple[int, int], KGEdge] = {}
        for edge in self.edges:
            left = self._uid_to_index[edge.source]
            right = self._uid_to_index[edge.target]
            edge_lookup[self._edge_key(left, right)] = edge

        refined_partition: list[set[int]] = []
        for cluster in self.report_partition:
            if not self.refinement_enabled or len(cluster) <= 1:
                refined_partition.append(set(cluster))
                continue
            ordered = sorted(
                cluster,
                key=lambda node: (
                    -sum(
                        edge_lookup[pair].weight
                        for other in cluster
                        if other != node
                        and (pair := self._edge_key(node, other)) in edge_lookup
                    ),
                    node,
                ),
            )
            groups: list[set[int]] = []
            for node in ordered:
                compatible_groups = [
                    group
                    for group in groups
                    if all(
                        (edge := edge_lookup.get(self._edge_key(node, other)))
                        is not None
                        and (
                            edge.weight >= self.minimum_pair_cosine
                            or self._edge_key(node, other)
                            in self._semantic_fallback_edge_keys
                        )
                        and self._anchor_compatible(
                            node,
                            other,
                            enforce_action_compatibility=(
                                self.refinement_action_compatibility
                            ),
                        )
                        for other in group
                    )
                ]
                if compatible_groups:
                    best = max(
                        compatible_groups,
                        key=lambda group: (
                            sum(
                                edge_lookup[self._edge_key(node, other)].weight
                                for other in group
                            )
                            / len(group),
                            len(group),
                            -min(group),
                        ),
                    )
                    best.add(node)
                else:
                    groups.append({node})
            refined_partition.extend(groups)

        partition_of = {}
        for partition_id, cluster in enumerate(refined_partition):
            for index in cluster:
                partition_of[index] = partition_id
        kept_edges: list[dict[str, Any]] = []
        for edge in self.edges:
            left = self._uid_to_index[edge.source]
            right = self._uid_to_index[edge.target]
            if partition_of.get(left) != partition_of.get(right):
                continue
            adjacency[left].add(right)
            adjacency[right].add(left)
            kept_edges.append(edge.to_dict())

        # This is deliberately the same abstraction used by ToolScope after its
        # cosine+LLM graph construction: downstream code sees only connected
        # components, not the mechanism that proposed graph edges.
        components: list[set[int]] = []
        unseen = set(range(len(self.tools)))
        while unseen:
            start = min(unseen)
            stack = [start]
            component: set[int] = set()
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                unseen.discard(node)
                stack.extend(sorted(adjacency[node] - component, reverse=True))
            components.append(component)
        graph = {
            "method": "structural-entropy pruning graph",
            "automatic_refinement": {
                "enabled": self.refinement_enabled,
                "method": "greedy complete-link",
                "minimum_pair_cosine": self.minimum_pair_cosine,
                "enforce_action_compatibility": (self.refinement_action_compatibility),
                "entropy_partition_count": len(self.report_partition),
                "refined_partition_count": len(refined_partition),
            },
            "nodes": [{"uid": tool.uid, "name": tool.name} for tool in self.tools],
            "edges": kept_edges,
            "connected_components": [
                [self.tools[index].uid for index in sorted(component)]
                for component in components
            ],
            "component_count": len(components),
            "non_singleton_component_count": sum(
                len(component) > 1 for component in components
            ),
        }
        self.refinement_stats = dict(graph["automatic_refinement"])
        return graph, components

    def to_dict(self) -> dict[str, Any]:
        partition = [
            [self.tools[index].name for index in sorted(cluster)]
            for cluster in self.report_partition
        ]
        return {
            "method": "KG-induced agglomerative two-dimensional structural entropy",
            "merge_priority": self.priority_mode,
            "root": self.root_id,
            "tool_count": len(self.tools),
            "entropy_1d": round(self.entropy_1d, 8),
            "entropy_at_stop": round(self.final_entropy, 8),
            "report_partition_count": len(partition),
            "report_partition": partition,
            "merge_count": len(self.merge_history),
            "merge_history": self.merge_history,
            "nodes": self.tree,
        }


class OpenAICompatibleLLM:
    """Small optional client used only for ToolScope-compatible LLM stages."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        self.spec = dict(spec)
        self.enabled = bool(self.spec.get("enabled", False))
        self.model = str(self.spec.get("model", ""))
        self.provider = str(self.spec.get("provider", "openai_compatible"))
        self._client: Any = None

    def complete(self, prompt: str) -> str:
        if not self.enabled:
            raise RuntimeError("LLM stage is disabled")
        if self.provider == "local_transformers":
            from .local_generation import create_local_llm

            return create_local_llm(self.spec).complete(prompt, self.spec)
        api_key = str(self.spec.get("api_key", "")).strip()
        api_key_env = str(self.spec.get("api_key_env", "")).strip()
        if api_key_env:
            import os

            api_key = os.getenv(api_key_env, "").strip()
        if not api_key or api_key.startswith("YOUR_"):
            raise ValueError(
                f"Missing api key or environment variable {api_key_env or 'api_key'} "
                f"for LLM {self.spec.get('id')}"
            )
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install openai to enable LLM stages") from exc
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.spec.get("base_url"),
                timeout=float(self.spec.get("timeout_seconds", 120)),
                max_retries=int(self.spec.get("max_retries", 0)),
            )
        if self.provider == "dashscope":
            request = {
                "model": self.model,
                "input": prompt,
                "extra_body": {
                    "enable_thinking": bool(self.spec.get("thinking", True))
                },
            }
            response = self._client.responses.create(**request)
            output_text = getattr(response, "output_text", None)
            if output_text:
                return str(output_text).strip()
            output = getattr(response, "output", None) or []
            texts: list[str] = []
            for item in output:
                contents = getattr(item, "content", None) or []
                for content in contents:
                    text = getattr(content, "text", None)
                    if text:
                        texts.append(str(text))
            return "\n".join(texts).strip()

        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": int(self.spec.get("max_tokens", 4000)),
        }
        if self.spec.get("temperature") is not None:
            request["temperature"] = float(self.spec["temperature"])
        if self.spec.get("reasoning_effort"):
            request["reasoning_effort"] = str(self.spec["reasoning_effort"])
        if self.provider not in {"dashscope", "quanzil"} and "thinking" in self.spec:
            request["extra_body"] = {
                "thinking": {
                    "type": "enabled" if self.spec.get("thinking") else "disabled"
                }
            }
        response = self._client.chat.completions.create(**request)
        return str(response.choices[0].message.content or "").strip()


class PromptStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def load(self, name: str) -> str:
        return (self.directory / name).read_text(encoding="utf-8")


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


class EntroTool:
    """Offline merger pipeline replacing only ToolScopeMerger's merge evidence."""

    def __init__(
        self,
        tools: Sequence[ToolRecord],
        config: Mapping[str, Any],
        prompt_store: PromptStore | None = None,
        graph_encoder: Any | None = None,
    ) -> None:
        self.tools = list(tools)
        self.config = config
        self.prompt_store = prompt_store
        self.kg_builder = KnowledgeGraphBuilder(
            config.get("merger", {}), encoder=graph_encoder
        )
        self.graph: dict[str, Any] = {}
        self.edges: list[KGEdge] = []
        self.entropy: StructuralEntropyMerger | None = None
        self.pruning_graph: dict[str, Any] = {}
        self.connected_components: list[set[int]] = []
        self.mapping: dict[str, str] = {tool.name: tool.name for tool in self.tools}
        self.merged_tools: list[ToolRecord] = list(self.tools)
        self.clusters: list[dict[str, Any]] = []
        self.autocorrection_audit: list[dict[str, Any]] = []
        self.documentation_audit: list[dict[str, Any]] = []

    def _llm(self, identifier: str) -> OpenAICompatibleLLM | None:
        for spec in self.config.get("llms", []):
            if spec.get("id") == identifier:
                return OpenAICompatibleLLM(spec)
        return None

    @staticmethod
    def _single_tool(
        tool: ToolRecord, name: str | None = None, description: str | None = None
    ) -> ToolRecord:
        return ToolRecord(
            name=name or tool.name,
            description=description or tool.description,
            field=tool.field,
            arguments=dict(tool.arguments),
            uid=tool.uid,
            required_arguments=set(tool.required_arguments),
            result_fields=set(tool.result_fields),
            original=dict(tool.original),
        )

    def _autocorrect(self, members: list[ToolRecord]) -> list[list[ToolRecord]]:
        if len(members) <= 1:
            return [members]
        llm = self._llm("merge_autocorrection")
        if not llm or not llm.enabled or not self.prompt_store:
            return [members]
        group = "\n".join(
            f"{index} : {tool.signature} - {tool.description}"
            for index, tool in enumerate(members)
        )
        prompt = self.prompt_store.load("merge_autocorrection.txt").replace(
            "{GROUP_BLOCK}", group
        )
        raw_response = llm.complete(prompt)
        parsed = _parse_json_object(raw_response)
        self.autocorrection_audit.append(
            {
                "member_uids": [tool.uid for tool in members],
                "response": raw_response,
                "parsed": parsed,
            }
        )
        if parsed and parsed.get("merge") == "MERGE_OK":
            return [members]
        if (
            not parsed
            or parsed.get("merge") != "MERGE_BAD"
            or not isinstance(parsed.get("clusters"), list)
        ):
            return [[member] for member in members]
        by_id = {str(index): tool for index, tool in enumerate(members)}
        result: list[list[ToolRecord]] = []
        used: set[str] = set()
        for group_ids in parsed["clusters"]:
            if not isinstance(group_ids, list):
                continue
            group_tools = [by_id[str(item)] for item in group_ids if str(item) in by_id]
            if group_tools:
                result.append(group_tools)
                used.update(str(item) for item in group_ids)
        for index, member in enumerate(members):
            if str(index) not in used:
                result.append([member])
        return result or [[member] for member in members]

    def _synthesize(self, members: list[ToolRecord]) -> ToolRecord:
        representative = min(members, key=lambda tool: (len(tool.name), tool.name))
        description = representative.description
        llm = self._llm("documentation_merger")
        if llm and llm.enabled and self.prompt_store and len(members) > 1:
            keep_block = (
                f"KEEP: {representative.signature} - {representative.description}"
            )
            prune = "\n".join(
                f"MERGE: {tool.signature} - {tool.description}"
                for tool in members
                if tool is not representative
            )
            schema_block = "\n\n".join(
                "\n".join(
                    [
                        f"SOURCE UID: {tool.uid}",
                        f"SOURCE NAME: {tool.name}",
                        f"SIGNATURE: {tool.signature}",
                        "ARGUMENT SCHEMA: "
                        + json.dumps(tool.arguments, ensure_ascii=False, indent=2),
                        "REQUIRED ARGUMENTS: "
                        + json.dumps(
                            sorted(tool.required_arguments), ensure_ascii=False
                        ),
                        "RESULT FIELDS: "
                        + json.dumps(sorted(tool.result_fields), ensure_ascii=False),
                    ]
                )
                for tool in members
            )
            prompt = self.prompt_store.load("documentation_merger.txt")
            prompt = prompt.replace("{keep_name}", representative.name)
            prompt = prompt.replace("{keep_block}", keep_block).replace(
                "{prune_block}", prune
            )
            prompt = prompt.replace("{schema_block}", schema_block)
            generated = llm.complete(prompt)
            self.documentation_audit.append(
                {
                    "member_uids": [tool.uid for tool in members],
                    "response": generated,
                }
            )
            if generated:
                description = generated
        operation_specs: dict[str, Any] = {}
        operation_names: list[str] = []
        for tool in members:
            operation = normalize_name(tool.name).lower() or "operation"
            if operation in operation_specs:
                operation = f"{operation}_{len(operation_specs)}"
            operation_names.append(operation)
            operation_specs[operation] = {
                "source_uid": tool.uid,
                "source_name": tool.name,
                "arguments": dict(tool.arguments),
                "required_arguments": sorted(tool.required_arguments),
                "result_fields": sorted(tool.result_fields),
            }
        arguments: dict[str, Any] = {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": operation_names,
                    "description": "Select the operation corresponding to the requested function.",
                },
                "parameters": {
                    "type": "object",
                    "description": "Arguments required by the selected operation.",
                },
            },
            "required": ["operation", "parameters"],
        }
        return ToolRecord(
            name=representative.name,
            description=description,
            field=representative.field,
            arguments=arguments,
            required_arguments={"operation", "parameters"},
            result_fields=set().union(*(tool.result_fields for tool in members)),
            original={
                "merged_from": [tool.name for tool in members],
                "representative": representative.name,
                "operation_specs": operation_specs,
            },
        )

    def fit(self) -> "EntroTool":
        self.graph, self.edges = self.kg_builder.build(self.tools)
        self.entropy = StructuralEntropyMerger(
            self.tools, self.edges, self.config.get("merger", {})
        ).fit()
        self.pruning_graph, self.connected_components = self.entropy.pruning_graph()
        merged: list[ToolRecord] = []
        self.mapping = {}
        clustered_indices: set[int] = set()
        non_singleton_components = [
            component for component in self.connected_components if len(component) > 1
        ]
        for index, member_indices in enumerate(non_singleton_components):
            members = [self.tools[item] for item in sorted(member_indices)]
            clustered_indices.update(member_indices)
            for corrected in self._autocorrect(members):
                canonical = self._synthesize(corrected)
                merged.append(canonical)
                for member in corrected:
                    self.mapping[member.name] = canonical.name
                    self.mapping[member.uid] = canonical.name
                self.clusters.append(
                    {
                        "cluster_id": f"merge:{index:05d}",
                        "members": [member.name for member in corrected],
                        "canonical": canonical.name,
                        "size": len(corrected),
                    }
                )
        for tool_index, tool in enumerate(self.tools):
            if tool_index not in clustered_indices:
                merged.append(tool)
                self.mapping[tool.name] = tool.name
                self.mapping[tool.uid] = tool.name
        # Keep deterministic order, as ToolScope's retriever indexes a fixed set.
        self.merged_tools = sorted(merged, key=lambda tool: tool.name)
        return self

    def to_dict(self) -> dict[str, Any]:
        assert self.entropy is not None
        return {
            "input_tool_count": len(self.tools),
            "output_tool_count": len(self.merged_tools),
            "knowledge_graph": self.graph,
            "structural_entropy_codebook": self.entropy.to_dict(),
            "pruning_graph": self.pruning_graph,
            "clusters": self.clusters,
            "mapping": dict(sorted(self.mapping.items())),
            "merged_tools": [tool.to_dict() for tool in self.merged_tools],
        }


def run_merger_only(
    dataset_root: str | Path,
    dataset: str,
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    matched_toolscope_size: bool = False,
    encoder: Any | None = None,
    tools_file: str | None = None,
) -> dict[str, Any]:
    """Stop immediately after structural-entropy connected components.

    This path reads only ``*_osa.json``. It never opens the query/solution file,
    initializes retrieval models, or calls an LLM.
    """

    aliases = {
        "seal": "SealTool",
        "sealtool": "SealTool",
        "bfcl": "BFCL",
        "bfcl_pm": "BFCL_PM",
    }
    dataset_key = aliases.get(dataset.lower(), dataset)
    dataset_root = Path(dataset_root)
    tools_path = dataset_root / dataset_key / (tools_file or f"{dataset_key}_osa.json")
    tools = load_tools(tools_path)
    effective = json.loads(json.dumps(config))
    merger_config = effective.setdefault("merger", {})
    if matched_toolscope_size:
        sizes = effective.get("evaluation", {}).get(
            "reported_toolscope_merged_sizes_for_optional_control", {}
        )
        if dataset_key not in sizes:
            raise ValueError(f"No matched ToolScope size for {dataset_key}")
        merger_config["target_clusters"] = int(sizes[dataset_key])
    builder = KnowledgeGraphBuilder(merger_config, encoder=encoder)
    knowledge_graph, edges = builder.build(tools)
    entropy = StructuralEntropyMerger(tools, edges, merger_config).fit()
    pruning_graph, components = entropy.pruning_graph()
    uid_to_component: dict[str, int] = {}
    for component_index, component in enumerate(components):
        for tool_index in component:
            uid_to_component[tools[tool_index].uid] = component_index
    component_edges: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in pruning_graph["edges"]:
        component_index = uid_to_component.get(str(edge["source"]))
        if component_index is not None:
            component_edges[component_index].append(edge)
    component_records: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        component_records.append(
            {
                "component_id": f"component:{index:05d}",
                "size": len(component),
                "needs_merge_review": len(component) > 1,
                "review_decision": None,
                "review_reason": "",
                "members": [
                    {
                        "uid": tools[item].uid,
                        "name": tools[item].name,
                        "signature": tools[item].signature,
                        "description": tools[item].description,
                        "required_arguments": sorted(tools[item].required_arguments),
                        "result_fields": sorted(tools[item].result_fields),
                    }
                    for item in sorted(component)
                ],
                "internal_edge_count": len(component_edges.get(index, [])),
                "internal_edges": component_edges.get(index, []),
            }
        )
    summary = {
        "dataset": dataset_key,
        "stage": "connected_components_complete",
        "input_tools": len(tools),
        "knowledge_graph_edges": len(edges),
        "semantic_graph": {
            "embedding_model": knowledge_graph.get("embedding_model"),
            "candidate_generation": knowledge_graph.get("candidate_generation"),
            "evaluated_pair_count": knowledge_graph.get("evaluated_pair_count"),
            "minimum_cosine": knowledge_graph.get("minimum_cosine"),
            "edge_condition": knowledge_graph.get("edge_condition"),
            "edge_weight": knowledge_graph.get("edge_weight"),
            "contract_filter": knowledge_graph.get("contract_filter"),
        },
        "structural_merges": len(entropy.merge_history),
        "entropy_partition_count": len(entropy.report_partition),
        "connected_components": len(components),
        "non_singleton_components": sum(len(item) > 1 for item in components),
        "largest_component_size": max((len(item) for item in components), default=0),
        "proposed_tool_count_if_each_component_is_merged": len(components),
        "matched_toolscope_size_control": matched_toolscope_size,
        "queries_or_gold_loaded": False,
        "retriever_initialized": False,
        "llm_called": False,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _json_dump(destination / "merger_only_summary.json", summary)
    _json_dump(destination / "knowledge_graph.json", knowledge_graph)
    _json_dump(destination / "structural_entropy_codebook.json", entropy.to_dict())
    _json_dump(destination / "entropy_pruning_graph.json", pruning_graph)
    _json_dump(destination / "connected_components.json", component_records)
    review_lines = [
        f"# {dataset_key} Structural-Entropy Connected Components Review",
        "",
        "Mark each non-singleton component as reasonable or unreasonable before running the LLM merger.",
        "",
    ]
    for record in sorted(
        (item for item in component_records if item["size"] > 1),
        key=lambda item: (-int(item["size"]), str(item["component_id"])),
    ):
        review_lines.extend(
            [
                f"## {record['component_id']} (size={record['size']}, edges={record['internal_edge_count']})",
                "",
                "- [ ] Reasonable: all tools can be consolidated without losing intent",
                "- [ ] Unreasonable: this component must be split",
                "- Reason: ",
                "",
                "Contract review: inspect required-argument overlap and semantic evidence.",
                "",
                "| UID | Tool | Signature | Description |",
                "|---|---|---|---|",
            ]
        )
        for member in record["members"]:
            values = [
                str(member["uid"]),
                str(member["name"]),
                str(member["signature"]),
                str(member["description"]),
            ]
            values = [value.replace("|", "\\|").replace("\n", " ") for value in values]
            review_lines.append("| " + " | ".join(values) + " |")
        review_lines.extend(["", "### Structural edge evidence", ""])
        for edge in record["internal_edges"]:
            review_lines.append(
                f"- `{edge['source']}` ↔ `{edge['target']}`; "
                f"weight={edge['weight']}; evidence={edge['evidence']}"
            )
        review_lines.append("")
    (destination / "connected_components_review.md").write_text(
        "\n".join(review_lines), encoding="utf-8"
    )
    return summary


def load_config(path: str | Path) -> dict[str, Any]:
    value = _json_load(path)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a JSON object")
    return value
