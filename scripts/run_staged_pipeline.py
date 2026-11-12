"""Run staged tool merging and query evaluation with configured LLMs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from entrotool.toolscope_merger import (  # noqa: E402
    EntroTool,
    OpenAICompatibleLLM,
    PromptStore,
    ToolRecord,
    load_config,
    load_tools,
)
from entrotool.evaluation_metrics import (  # noqa: E402
    candidate_recall,
    tool_selection_metrics,
)
from entrotool.toolscope_retriever import (  # noqa: E402
    QueryDecomposer,
    ToolScopeRetriever,
    ToolSelectionAgent,
    query_text,
    solution_names,
)

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(values: Iterable[Any], **_: Any) -> Iterable[Any]:
        return values


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _audit_entries(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _audit_succeeded(value: Any) -> bool:
    entries = _audit_entries(value)
    return bool(entries) and all(
        not entry.get("error") and str(entry.get("response", "")).strip()
        for entry in entries
    )


def _selection_audit_succeeded(entry: dict[str, Any]) -> bool:
    return bool(
        not entry.get("error")
        and str(entry.get("response", "")).strip()
        and entry.get("valid_response") is not False
    )


def _retry_failed_selection_audits(
    row: dict[str, Any],
    k_values: list[int],
    valid_gold: set[str],
    select_one: Any,
) -> int:
    retrieval = row["retrieval"]
    query = str(row["query"])
    audit_by_k = row.setdefault("deepseek_selection_audit", {})
    retried = 0

    for k in k_values:
        if k <= 1:
            continue
        key = str(k)
        audits = audit_by_k.get(key)
        if isinstance(audits, dict):
            audits = [audits]
        if not isinstance(audits, list):
            continue

        candidates = retrieval["selected"][:k]
        traces = retrieval.get("traces", [])
        for index, audit in enumerate(audits):
            if _selection_audit_succeeded(audit):
                continue
            step = str(audit.get("step") or query)
            trace = traces[index] if index < len(traces) else {}
            score_by_name = {
                item["tool"]: float(item["score"])
                for item in trace.get("reranker_scores", [])
            }
            _selected, updated_audit = select_one(
                query,
                step,
                candidates,
                {name: score_by_name.get(name, 0.0) for name in candidates},
            )
            updated_audit["step"] = step
            audits[index] = updated_audit
            retried += 1

        audit_by_k[key] = audits
        selected_tools: list[str] = []
        for audit in audits:
            if not _selection_audit_succeeded(audit):
                continue
            response = str(audit.get("response", "")).strip()
            selected = response.splitlines()[0].strip().strip("`'")
            if selected in candidates and selected not in selected_tools:
                selected_tools.append(selected)

        row.setdefault("deepseek_selections", {})[key] = selected_tools
        metric = row["metrics"][key]
        scores = tool_selection_metrics(selected_tools, [valid_gold])
        metric["selected"] = selected_tools
        metric["path"] = scores["path"]
        metric["csr"] = scores["csr"]

    return retried


def _read_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row[key])] = row
    return rows


def _compatible_cluster_cache(
    path: Path, autocorrection_enabled: bool
) -> dict[str, dict[str, Any]]:
    return {
        component_id: row
        for component_id, row in _read_jsonl(path, "component_id").items()
        if row.get("autocorrection_enabled") is autocorrection_enabled
    }


def _enable_llms(config: dict[str, Any]) -> None:
    enabled = {
        "merge_autocorrection",
        "documentation_merger",
        "query_decomposer",
        "tool_selection",
    }
    for spec in config.get("llms", []):
        spec["enabled"] = spec.get("id") in enabled
        if spec.get("provider") == "openai_compatible":
            spec["thinking"] = False
            spec.pop("reasoning_effort", None)


def _autocorrection_enabled(config: dict[str, Any], cli_disabled: bool) -> bool:
    configured = config.get("merger", {}).get("autocorrection", {}).get("enabled", True)
    return bool(configured) and not cli_disabled


def _record_to_tool(raw: dict[str, Any]) -> ToolRecord:
    return ToolRecord(
        name=str(raw["name"]),
        description=str(raw.get("description", "")),
        field=str(raw.get("field", "")),
        arguments=dict(raw.get("arguments", {})),
        uid=str(raw.get("uid", "")),
        required_arguments=set(raw.get("required_arguments", [])),
        result_fields=set(raw.get("result_fields", [])),
        original=dict(raw.get("original", {})),
    )


def _load_saved_merge(output: Path) -> tuple[list[ToolRecord], dict[str, set[str]]]:
    tools_path = output / "merged_tools.json"
    mapping_path = output / "tool_mapping.json"
    if not tools_path.exists() or not mapping_path.exists():
        raise FileNotFoundError(
            "Query-only mode requires merged_tools.json and tool_mapping.json. "
            "Run --merge-only first."
        )
    merged_tools = [
        _record_to_tool(item)
        for item in json.loads(tools_path.read_text(encoding="utf-8"))
    ]
    raw_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping = {
        str(name): {str(value) for value in values}
        for name, values in raw_mapping.items()
    }
    return merged_tools, mapping


def _cluster_stage(
    tools: list[ToolRecord],
    config: dict[str, Any],
    prompts: PromptStore,
    components: list[dict[str, Any]],
    output: Path,
    resume: bool,
    merge_concurrency: int,
    autocorrection_enabled: bool,
) -> tuple[list[ToolRecord], dict[str, set[str]], list[dict[str, Any]]]:
    audit_path = output / "clusters.jsonl"
    completed = (
        _compatible_cluster_cache(audit_path, autocorrection_enabled) if resume else {}
    )
    if not resume and audit_path.exists():
        audit_path.unlink()
    by_uid = {tool.uid: tool for tool in tools}
    member_uids = {
        str(member["uid"])
        for component in components
        if int(component["size"]) > 1
        for member in component["members"]
    }
    merged_tools = [tool for tool in tools if tool.uid not in member_uids]
    name_mapping: dict[str, set[str]] = defaultdict(set)
    for tool in merged_tools:
        name_mapping[tool.name].add(tool.name)
    all_rows: list[dict[str, Any]] = []
    merge_components = [item for item in components if int(item["size"]) > 1]

    def process_component(
        component: dict[str, Any],
    ) -> tuple[dict[str, Any], list[ToolRecord]]:
        component_id = str(component["component_id"])
        members = [by_uid[str(item["uid"])] for item in component["members"]]
        if component_id in completed:
            return completed[component_id], members
        pipeline = EntroTool(tools, config, prompt_store=prompts)
        corrected = (
            pipeline._autocorrect(members) if autocorrection_enabled else [members]
        )
        canonical_tools: list[ToolRecord] = []
        for group in corrected:
            canonical_tools.append(
                pipeline._synthesize(group) if len(group) > 1 else group[0]
            )
        return {
            "component_id": component_id,
            "members": [tool.uid for tool in members],
            "autocorrection": pipeline.autocorrection_audit,
            "autocorrection_enabled": autocorrection_enabled,
            "corrected_groups": [[tool.uid for tool in group] for group in corrected],
            "documentation_merger": pipeline.documentation_audit,
            "canonical_tools": [tool.to_dict() for tool in canonical_tools],
        }, members

    pending_components = [
        component
        for component in merge_components
        if str(component["component_id"]) not in completed
    ]
    rows_by_id: dict[str, tuple[dict[str, Any], list[ToolRecord]]] = {
        component_id: (row, [by_uid[uid] for uid in row["members"]])
        for component_id, row in completed.items()
    }
    with ThreadPoolExecutor(max_workers=max(1, int(merge_concurrency))) as executor:
        futures = {
            executor.submit(process_component, component): component
            for component in pending_components
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="LLM cluster merges",
            unit="cluster",
        ):
            component = futures[future]
            component_id = str(component["component_id"])
            row, members = future.result()
            rows_by_id[component_id] = (row, members)
            _append_jsonl(audit_path, row)
            autocorrection_rows = row.get("autocorrection", [])
            decision_text = (
                autocorrection_rows[-1].get("response", "")
                if autocorrection_rows
                else "SKIPPED (--no-autocorrection)"
            )
            print(
                f"[Cluster complete] {component_id}: " + str(decision_text),
                flush=True,
            )

    for component in merge_components:
        component_id = str(component["component_id"])
        row, members = rows_by_id[component_id]
        print("\n" + "=" * 100, flush=True)
        print(f"[Cluster] {component_id} ({len(members)} tools)", flush=True)
        print(
            "Members: "
            + json.dumps([tool.uid for tool in members], ensure_ascii=False),
            flush=True,
        )
        if component_id in completed:
            print("[Resume] Loaded saved LLM cluster result", flush=True)
        autocorrection_rows = row.get("autocorrection", [])
        decision = (
            autocorrection_rows[-1]
            if autocorrection_rows
            else {"response": "SKIPPED (--no-autocorrection)"}
        )
        print("LLM Auto-Correction:", flush=True)
        print(str(decision.get("response", "RESUMED RESULT")), flush=True)
        print(
            "Corrected groups: "
            + json.dumps(row["corrected_groups"], ensure_ascii=False),
            flush=True,
        )
        for doc in row.get("documentation_merger", []):
            print("LLM Documentation Merger:", flush=True)
            print(str(doc.get("response", "")), flush=True)
        canonical_tools = [_record_to_tool(item) for item in row["canonical_tools"]]
        merged_tools.extend(canonical_tools)
        canonical_by_source: dict[str, str] = {}
        for canonical in canonical_tools:
            operation_specs = canonical.original.get("operation_specs", {})
            if isinstance(operation_specs, dict):
                for spec in operation_specs.values():
                    if isinstance(spec, dict):
                        canonical_by_source[str(spec.get("source_uid", ""))] = (
                            canonical.name
                        )
            if canonical.uid:
                canonical_by_source[canonical.uid] = canonical.name
        for member in members:
            mapped = canonical_by_source.get(member.uid, member.name)
            name_mapping[member.name].add(mapped)
        all_rows.append(row)
    _write_jsonl(audit_path, all_rows)
    return (
        sorted(merged_tools, key=lambda tool: (tool.name, tool.uid)),
        name_mapping,
        all_rows,
    )


def _normalize_name_collisions(
    tools: list[ToolRecord], mapping: dict[str, set[str]]
) -> tuple[list[ToolRecord], list[dict[str, Any]]]:
    """Give distinct saved tools deterministic names when a merge leaves collisions."""
    by_name: dict[str, list[ToolRecord]] = defaultdict(list)
    for tool in tools:
        by_name[tool.name].append(tool)
    normalized: list[ToolRecord] = []
    collisions: list[dict[str, Any]] = []
    for name, variants in by_name.items():
        if len(variants) == 1:
            normalized.extend(variants)
            continue
        collision = {"name": name, "count": len(variants), "renamed": []}
        mapped_names: set[str] = set()
        for index, tool in enumerate(sorted(variants, key=lambda item: item.uid)):
            new_name = name if index == 0 else f"{name}__variant_{index}"
            collision["renamed"].append({"uid": tool.uid, "name": new_name})
            mapped_names.add(new_name)
            if new_name != name:
                tool = _record_to_tool({**tool.to_dict(), "name": new_name})
            normalized.append(tool)
        mapping.setdefault(name, set()).update(mapped_names)
        collisions.append(collision)
    return sorted(normalized, key=lambda tool: (tool.name, tool.uid)), collisions


def _query_stage(
    records: list[dict[str, Any]],
    merged_tools: list[ToolRecord],
    name_mapping: dict[str, set[str]],
    config: dict[str, Any],
    prompts: PromptStore,
    output: Path,
    resume: bool,
    retry_failed_selections: bool,
    query_concurrency: int,
    selection_concurrency: int,
    selection_retries: int,
    selection_timeout_seconds: float,
    dataset: str = "BFCL",
) -> list[dict[str, Any]]:
    audit_path = output / "queries.jsonl"
    completed = _read_jsonl(audit_path, "query_id") if resume else {}
    if not resume and audit_path.exists():
        audit_path.unlink()
    retriever = ToolScopeRetriever(merged_tools, config, strict_models=True)
    retriever_lock = Lock()
    decomposer = QueryDecomposer(
        OpenAICompatibleLLM(
            next(
                spec for spec in config["llms"] if spec.get("id") == "query_decomposer"
            )
        ),
        prompts,
    )
    llm_spec = next(
        dict(spec) for spec in config["llms"] if spec.get("id") == "tool_selection"
    )
    llm_spec["timeout_seconds"] = selection_timeout_seconds
    llm_spec["max_retries"] = 0
    k_values = [
        int(value)
        for value in config["evaluation"]
        .get("datasets", {})
        .get(dataset, {})
        .get("k", config["evaluation"].get("k_values", {}).get(dataset, [5]))
    ]
    max_k = max(k_values)
    rows: list[dict[str, Any]] = []
    run_signature = {"k_values": k_values}
    multi_tool = bool(
        config.get("evaluation", {})
        .get("datasets", {})
        .get(dataset, {})
        .get("multi_tool", False)
    )
    gold_uid_by_record_index = bool(
        config.get("evaluation", {})
        .get("datasets", {})
        .get(dataset, {})
        .get("gold_uid_by_record_index", False)
    )
    if retry_failed_selections:
        if not resume:
            raise ValueError("--retry-failed-selections requires --resume")
        requested_ids = [
            str(record.get("id", index))
            for index, record in enumerate(records, start=1)
        ]
        incomplete = [
            query_id
            for query_id in requested_ids
            if query_id not in completed
            or completed[query_id].get("k_values") != k_values
        ]
        if incomplete:
            raise ValueError(
                "--retry-failed-selections requires completed query audits; "
                f"missing or incompatible query IDs: {incomplete[:10]}"
            )
        malformed = [
            query_id
            for query_id in requested_ids
            for row in [completed[query_id]]
            if any(
                not isinstance(
                    row.get("deepseek_selection_audit", {}).get(str(k)), list
                )
                for k in k_values
                if k > 1
            )
        ]
        if malformed:
            raise ValueError(
                "--retry-failed-selections requires per-step selection audits; "
                f"invalid query IDs: {malformed[:10]}"
            )
    canonical_by_uid: dict[str, str] = {}
    for tool in merged_tools:
        operation_specs = tool.original.get("operation_specs", {})
        if isinstance(operation_specs, dict):
            for spec in operation_specs.values():
                if isinstance(spec, dict) and spec.get("source_uid"):
                    canonical_by_uid[str(spec["source_uid"])] = tool.name
        if tool.uid:
            canonical_by_uid.setdefault(tool.uid, tool.name)

    def mapped_gold_names(original_gold: Iterable[str], record_index: int) -> set[str]:
        """Relabel gold tools through ToolScope's one-to-one phi mapping."""
        if gold_uid_by_record_index:
            resolved: set[str] = set()
            for raw_name in original_gold:
                name = str(raw_name)
                source_uid = f"{name}@@{record_index:05d}"
                if source_uid not in canonical_by_uid:
                    raise KeyError(
                        f"No canonical mapping for indexed gold tool {source_uid!r}"
                    )
                resolved.add(canonical_by_uid[source_uid])
            return resolved

        resolved: set[str] = set()
        for raw_name in original_gold:
            name = str(raw_name)
            mapped = name_mapping.get(name)
            if not mapped:
                resolved.add(name)
            elif len(mapped) == 1:
                resolved.add(next(iter(mapped)))
            else:
                resolved.update(mapped)
        return resolved

    def candidates_contain_gold(
        candidates: Iterable[str], valid_gold: set[str]
    ) -> bool:
        candidate_set = set(candidates)
        if multi_tool:
            return valid_gold.issubset(candidate_set)
        return bool(candidate_set & valid_gold)

    def decompose_with_retries(query: str) -> list[str]:
        last_error: Exception | None = None
        attempts = max(1, selection_retries)
        for attempt in range(1, attempts + 1):
            try:
                return decomposer.split(query)
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    delay = min(2 ** (attempt - 1), 8)
                    print(
                        f"[Query Decomposition] attempt {attempt}/{attempts} failed: "
                        f"{exc!r}; retrying in {delay}s",
                        flush=True,
                    )
                    time.sleep(delay)
        raise RuntimeError(
            f"Query decomposition failed after {attempts} attempts"
        ) from last_error

    def select_one(
        query: str,
        step: str,
        candidates: list[str],
        candidate_scores: dict[str, float],
    ) -> tuple[str | None, dict[str, Any]]:
        # Keep one client and audit state per worker; OpenAI clients are not
        # shared across concurrent requests.
        selector = ToolSelectionAgent(
            OpenAICompatibleLLM(llm_spec), prompts, merged_tools
        )
        last_error: Exception | None = None
        for attempt in range(1, max(1, selection_retries) + 1):
            try:
                selected = selector.select(
                    query, step, candidates, candidate_scores=candidate_scores
                )
                audit = dict(selector.audit[-1])
                if not str(audit.get("response", "")).strip():
                    raise RuntimeError("LLM returned empty tool-selection content")
                if selected is None:
                    raise RuntimeError(
                        "LLM did not return exactly one candidate function name"
                    )
                audit["error"] = None
                audit["attempts"] = attempt
                return selected, audit
            except Exception as exc:
                last_error = exc
                if attempt < selection_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        return None, {
            "query": query,
            "step": step,
            "candidates": candidates,
            "response": "",
            "error": repr(last_error),
            "attempts": max(1, selection_retries),
        }

    def process_query(
        index: int, record: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        query_id = str(record.get("id", index))
        if query_id in completed and completed[query_id].get("k_values") == k_values:
            row = completed[query_id]
            original_gold = solution_names(record)
            valid_gold = mapped_gold_names(original_gold, index - 1)
            row["gold_original"] = original_gold
            row["gold_after_mapping"] = sorted(valid_gold)
            if retry_failed_selections:
                retried = _retry_failed_selection_audits(
                    row, k_values, valid_gold, select_one
                )
                if retried:
                    print(
                        f"[Retry failed selections] {query_id}: {retried} request(s)",
                        flush=True,
                    )
            for metric in row.get("metrics", {}).values():
                selected = metric.get("selected")
                candidates = metric.get("candidates", [])
                scores = tool_selection_metrics(selected, [valid_gold])
                metric["success"] = None
                metric["path"] = scores["path"]
                metric["csr"] = scores["csr"]
                metric["candidate_recall"] = candidate_recall(candidates, [valid_gold])
                metric.pop("correct", None)
                metric["candidate_contains_gold"] = candidates_contain_gold(
                    candidates, valid_gold
                )
            return row, None
        query = query_text(record)
        use_decomposition = multi_tool
        subqueries = decompose_with_retries(query) if use_decomposition else [query]
        with retriever_lock:
            retrieval = retriever.retrieve(query, max_k, subqueries=subqueries)
        original_gold = solution_names(record)
        valid_gold = mapped_gold_names(original_gold, index - 1)
        row = {
            "query_id": query_id,
            **run_signature,
            "query": query,
            "gold_original": original_gold,
            "gold_after_mapping": sorted(valid_gold),
            "retrieval": retrieval,
            "deepseek_selections": {},
            "deepseek_selection_audit": {},
            "metrics": {},
            "query_decomposition_used": use_decomposition,
            "subqueries": subqueries,
        }
        with ThreadPoolExecutor(
            max_workers=max(1, int(selection_concurrency))
        ) as executor:
            for k in k_values:
                candidates = retrieval["selected"][:k]
                if k == 1:
                    direct_top_one = retrieval.get("direct_top_one", candidates)
                    selected = direct_top_one[0] if direct_top_one else None
                    direct_candidates = [selected] if selected else []
                    row["deepseek_selections"]["1"] = selected
                    row["deepseek_selection_audit"]["1"] = {
                        "method": "direct_top1",
                        "query": query,
                        "step": query,
                        "candidates": direct_candidates,
                        "response": selected or "",
                        "valid_response": selected is not None,
                        "error": None,
                        "attempts": 0,
                    }
                    row["metrics"]["1"] = {
                        "candidates": direct_candidates,
                        "selected": selected,
                        "success": None,
                        **tool_selection_metrics(selected, [valid_gold]),
                        "candidate_recall": candidate_recall(
                            direct_candidates, [valid_gold]
                        ),
                        "candidate_contains_gold": candidates_contain_gold(
                            direct_candidates, valid_gold
                        ),
                    }
                    continue
                futures: list[tuple[Any, str]] = []
                for step_index, step in enumerate(subqueries):
                    trace = retrieval["traces"][step_index]
                    score_by_name = {
                        item["tool"]: float(item["score"])
                        for item in trace["reranker_scores"]
                    }
                    future = executor.submit(
                        select_one,
                        query,
                        step,
                        candidates,
                        {name: score_by_name.get(name, 0.0) for name in candidates},
                    )
                    futures.append((future, step))
                selected_tools: list[str] = []
                audits: list[dict[str, Any]] = []
                for future, step in futures:
                    selected, audit = future.result()
                    audit["step"] = step
                    audits.append(audit)
                    if selected and selected not in selected_tools:
                        selected_tools.append(selected)
                row["deepseek_selections"][str(k)] = selected_tools
                row["deepseek_selection_audit"][str(k)] = audits
                row["metrics"][str(k)] = {
                    "candidates": candidates,
                    "selected": selected_tools,
                    "success": None,
                    **tool_selection_metrics(selected_tools, [valid_gold]),
                    "candidate_recall": candidate_recall(candidates, [valid_gold]),
                    "candidate_contains_gold": candidates_contain_gold(
                        candidates, valid_gold
                    ),
                }
        return row, None

    pending = [
        (index, record)
        for index, record in enumerate(records, start=1)
        if retry_failed_selections
        or not (
            str(record.get("id", index)) in completed
            and completed[str(record.get("id", index))].get("k_values") == k_values
        )
    ]
    with tqdm(total=len(records), desc="Query pipeline", unit="query") as progress:
        for index, record in enumerate(records, start=1):
            if (index, record) not in pending:
                row, _ = process_query(index, record)
                rows.append(row)
                progress.update(1)
        with ThreadPoolExecutor(max_workers=max(1, int(query_concurrency))) as executor:
            futures = {
                executor.submit(process_query, index, record): index
                for index, record in pending
            }
            for future in as_completed(futures):
                row, decomposition = future.result()
                if decomposition is not None:
                    _append_jsonl(output / "query_decompositions.jsonl", decomposition)
                rows.append(row)
                _append_jsonl(audit_path, row)
                progress.update(1)

    rows.sort(key=lambda row: str(row["query_id"]))
    _write_jsonl(audit_path, rows)
    for index, row in enumerate(rows, start=1):
        print("\n" + "-" * 100, flush=True)
        print(f"[Query] {index}/{len(records)} {row['query_id']}", flush=True)
        print("Query: " + str(row["query"]), flush=True)
        print(
            "Gold: " + json.dumps(row["gold_after_mapping"], ensure_ascii=False),
            flush=True,
        )
        for k in k_values:
            result = row["metrics"][str(k)]
            print(
                f"Candidates@{k}: "
                + json.dumps(result["candidates"], ensure_ascii=False),
                flush=True,
            )
            audit_entries = _audit_entries(
                row.get("deepseek_selection_audit", {}).get(str(k))
            )
            responses = [entry.get("response", "") for entry in audit_entries]
            response: Any = responses[0] if len(responses) == 1 else responses
            print(f"LLM raw response@{k}: {response}", flush=True)
            print(
                f"LLM@{k}: {result['selected']} | "
                f"CSR: {result['csr']:.0f} | Path: {result['path']:.4f}",
                flush=True,
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BFCL", help="dataset directory name")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--merge-cache",
        default=None,
        help="directory containing saved merged_tools.json and tool_mapping.json; defaults to --output",
    )
    parser.add_argument("--components", default=None)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--merge-only",
        action="store_true",
        help="run and save LLM cluster merging, but do not evaluate queries",
    )
    mode.add_argument(
        "--query-only",
        action="store_true",
        help="load saved merged_tools.json and tool_mapping.json; never rerun merging",
    )
    parser.add_argument("--max-clusters", type=int, default=None)
    parser.add_argument(
        "--merge-concurrency",
        type=int,
        default=5,
        help="maximum concurrent LLM cluster-merge requests",
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--queries", default=None, help="query JSON file to evaluate")
    parser.add_argument(
        "--query-concurrency",
        type=int,
        default=1,
        help="maximum concurrent queries; use selection-concurrency for subqueries within one query",
    )
    parser.add_argument(
        "--selection-concurrency",
        type=int,
        default=8,
        help="maximum concurrent LLM tool-selection requests",
    )
    parser.add_argument(
        "--selection-retries",
        type=int,
        default=3,
        help="attempts per LLM tool-selection request",
    )
    parser.add_argument(
        "--selection-timeout-seconds",
        type=float,
        default=60.0,
        help="timeout for each tool-selection API request",
    )
    parser.add_argument(
        "--retry-failed-selections",
        action="store_true",
        help="retry only failed or empty cached selections; requires --query-only --resume",
    )
    parser.add_argument(
        "--no-autocorrection",
        action="store_true",
        help="skip Auto-Correction and directly send each candidate cluster to Documentation Merger",
    )
    parser.add_argument(
        "--keep-name-collisions",
        action="store_true",
        help="preserve duplicate saved names for compatibility; default renames distinct variants",
    )
    args = parser.parse_args()
    if args.query_concurrency < 1 or args.selection_concurrency < 1:
        parser.error("query and selection concurrency must be at least 1")
    if args.selection_timeout_seconds <= 0:
        parser.error("--selection-timeout-seconds must be positive")
    if args.retry_failed_selections and not args.query_only:
        parser.error("--retry-failed-selections requires --query-only")
    if args.retry_failed_selections and not args.resume:
        parser.error("--retry-failed-selections requires --resume")
    dataset = args.dataset
    config = copy.deepcopy(load_config(PROJECT / "config.json"))
    autocorrection_enabled = _autocorrection_enabled(config, args.no_autocorrection)
    dataset_config = config.get("evaluation", {}).get("datasets", {}).get(dataset)
    dataset_files = {
        "BFCL": ("BFCL_osa.json", "BFCL_v.json"),
        "BFCL_PM": ("BFCL_pm_osa.json", "BFCL_pm_v.json"),
        "SealTool": ("SealTool_osa.json", "SealTool_v.json"),
    }
    if dataset_config:
        tool_file = str(dataset_config["tools"])
        query_file = str(dataset_config["queries"])
    elif dataset in dataset_files:
        tool_file, query_file = dataset_files[dataset]
    else:
        dataset_dir = WORKSPACE / "datasets" / dataset
        tool_candidates = sorted(dataset_dir.glob("*_osa.json"))
        query_candidates = sorted(dataset_dir.glob("*_v.json"))
        if not tool_candidates or not query_candidates:
            raise FileNotFoundError(
                f"Dataset {dataset!r} needs one *_osa.json and one *_v.json in {dataset_dir}"
            )
        tool_file = tool_candidates[0].name
        query_file = query_candidates[0].name
    output = Path(args.output or (PROJECT / f"outputs_{dataset.lower()}_full"))
    components_path = Path(
        args.components
        or (
            PROJECT
            / "outputs_entity_clusters_all_v17"
            / dataset
            / "connected_components.json"
        )
    )
    queries_path = Path(args.queries or (WORKSPACE / "datasets" / dataset / query_file))
    merge_cache = Path(args.merge_cache) if args.merge_cache else output
    _enable_llms(config)
    prompts = PromptStore(PROJECT / "prompts")
    tools = load_tools(WORKSPACE / "datasets" / dataset / tool_file)
    if args.query_only:
        print(
            f"{dataset} merge stage: SKIPPED (loading saved merged tools)", flush=True
        )
        merged_tools, mapping = _load_saved_merge(merge_cache)
        merge_summary_path = merge_cache / "merge_summary.json"
        merge_summary = (
            json.loads(merge_summary_path.read_text(encoding="utf-8"))
            if merge_summary_path.exists()
            else {}
        )
        cluster_rows = [
            {} for _ in range(int(merge_summary.get("proposed_clusters", 0)))
        ]
    else:
        components = json.loads(components_path.read_text(encoding="utf-8"))
        if args.max_clusters is not None:
            non_singletons = [item for item in components if int(item["size"]) > 1]
            allowed = {
                item["component_id"] for item in non_singletons[: args.max_clusters]
            }
            components = [
                item
                for item in components
                if int(item["size"]) == 1 or item["component_id"] in allowed
            ]
        print(
            f"{dataset} merge stage: {sum(int(item['size']) > 1 for item in components)} proposed clusters",
            flush=True,
        )
        merged_tools, mapping, cluster_rows = _cluster_stage(
            tools,
            config,
            prompts,
            components,
            output,
            args.resume,
            args.merge_concurrency,
            autocorrection_enabled,
        )
        _write_json(
            output / "merged_tools.json", [tool.to_dict() for tool in merged_tools]
        )
        _write_json(
            output / "tool_mapping.json",
            {key: sorted(value) for key, value in sorted(mapping.items())},
        )
    name_collisions: list[dict[str, Any]] = []
    if not args.keep_name_collisions:
        merged_tools, name_collisions = _normalize_name_collisions(
            merged_tools, mapping
        )
        _write_json(
            output / "merged_tools.json", [tool.to_dict() for tool in merged_tools]
        )
        _write_json(
            output / "tool_mapping.json",
            {key: sorted(value) for key, value in sorted(mapping.items())},
        )
    if args.merge_only:
        autocorrection_calls = sum(
            len(_audit_entries(row.get("autocorrection"))) for row in cluster_rows
        )
        documentation_calls = sum(
            len(_audit_entries(row.get("documentation_merger"))) for row in cluster_rows
        )
        merge_summary = {
            "dataset": dataset,
            "stage": "merge_only",
            "ablation": (
                "full" if autocorrection_enabled else "without_autocorrection"
            ),
            "input_tools": len(tools),
            "merged_tools": len(merged_tools),
            "proposed_clusters": len(cluster_rows),
            "autocorrection_enabled": autocorrection_enabled,
            "autocorrection_calls": autocorrection_calls,
            "documentation_merger_calls": documentation_calls,
            "files": {
                "clusters": "clusters.jsonl",
                "merged_tools": "merged_tools.json",
                "mapping": "tool_mapping.json",
            },
        }
        _write_json(output / "merge_summary.json", merge_summary)
        print(
            "\n" + json.dumps(merge_summary, ensure_ascii=False, indent=2), flush=True
        )
        return 0
    records = json.loads(queries_path.read_text(encoding="utf-8"))
    if args.max_queries is not None:
        records = records[: args.max_queries]
    query_rows = _query_stage(
        records,
        merged_tools,
        mapping,
        config,
        prompts,
        output,
        args.resume,
        args.retry_failed_selections,
        args.query_concurrency,
        args.selection_concurrency,
        args.selection_retries,
        args.selection_timeout_seconds,
        dataset,
    )
    k_values = [
        int(value)
        for value in config["evaluation"]
        .get("datasets", {})
        .get(dataset, {})
        .get("k", config["evaluation"].get("k_values", {}).get(dataset, [5]))
    ]
    error_rows: list[dict[str, Any]] = []
    for row in query_rows:
        for k in k_values:
            metric = row["metrics"][str(k)]
            if metric["csr"]:
                continue
            gold = set(row["gold_after_mapping"])
            candidates = set(metric["candidates"])
            audit = row.get("deepseek_selection_audit", {}).get(str(k), {})
            audit_entries = _audit_entries(audit)
            error_rows.append(
                {
                    "query_id": row["query_id"],
                    "k": k,
                    "query": row["query"],
                    "gold": sorted(gold),
                    "selected": metric.get("selected"),
                    "candidates": metric["candidates"],
                    "retrieval_scores": [
                        entry.get("candidate_scores", {}) for entry in audit_entries
                    ],
                    "error_type": (
                        "retrieval_miss"
                        if not (gold & candidates)
                        else "selection_confusion"
                    ),
                    "llm_response": [
                        entry.get("response", "") for entry in audit_entries
                    ],
                    "llm_error": [
                        entry.get("error")
                        for entry in audit_entries
                        if entry.get("error")
                    ],
                }
            )
    _write_jsonl_path = output / "selection_errors.jsonl"
    if _write_jsonl_path.exists():
        _write_jsonl_path.unlink()
    for error_row in error_rows:
        _append_jsonl(_write_jsonl_path, error_row)
    summary = {
        "dataset": dataset,
        "input_tools": len(tools),
        "merged_tools": len(merged_tools),
        "proposed_clusters": len(cluster_rows),
        "name_collisions_normalized": name_collisions,
        "queries": len(query_rows),
        "model": next(
            spec["model"]
            for spec in config["llms"]
            if spec.get("id") == "tool_selection"
        ),
        "query_decomposition_used": False,
        "metrics": {
            str(k): {
                "success": None,
                "path": round(
                    sum(row["metrics"][str(k)]["path"] for row in query_rows)
                    / len(query_rows),
                    4,
                ),
                "csr": round(
                    sum(row["metrics"][str(k)]["csr"] for row in query_rows)
                    / len(query_rows),
                    4,
                ),
                "candidate_recall": round(
                    sum(
                        row["metrics"][str(k)]["candidate_recall"] for row in query_rows
                    )
                    / len(query_rows),
                    4,
                ),
                "failed_or_empty_responses": sum(
                    not _audit_succeeded(row["deepseek_selection_audit"][str(k)])
                    for row in query_rows
                ),
            }
            for k in k_values
        },
        "metric_definitions": {
            "success": "Pending external LLM judge; the judge sees only query and selected tool documents.",
            "path": "Mean fraction of mapped gold tools covered by the final selected tool set.",
            "csr": "Fraction of queries whose final selected tool set exactly equals mapped gold.",
            "candidate_recall": "Mean fraction of mapped gold tools found in retrieved top-k candidates before tool selection.",
        },
        "files": {
            "clusters": "clusters.jsonl",
            "queries": "queries.jsonl",
            "merged_tools": "merged_tools.json",
            "mapping": "tool_mapping.json",
            "selection_errors": "selection_errors.jsonl",
        },
    }
    _write_json(output / "summary.json", summary)
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
