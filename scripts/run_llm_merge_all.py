"""Run Qwen Auto-Correction and Documentation Merger on saved candidate clusters."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from entrotool.toolscope_merger import (  # noqa: E402
    EntroTool,
    PromptStore,
    ToolRecord,
    load_config,
    load_tools,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["component_id"])] = row
    return result


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


def _enable_merge_llms(config: dict[str, Any]) -> None:
    for spec in config.get("llms", []):
        spec["enabled"] = spec.get("id") in {
            "merge_autocorrection",
            "documentation_merger",
        }


def _normalize_name_collisions(
    tools: list[ToolRecord], mapping: dict[str, set[str]]
) -> tuple[list[ToolRecord], list[dict[str, Any]]]:
    by_name: dict[str, list[ToolRecord]] = defaultdict(list)
    for tool in tools:
        by_name[tool.name].append(tool)
    normalized: list[ToolRecord] = []
    collisions: list[dict[str, Any]] = []
    for name, variants in by_name.items():
        if len(variants) == 1:
            normalized.extend(variants)
            continue
        renamed: list[dict[str, str]] = []
        mapped_names: set[str] = set()
        for index, tool in enumerate(sorted(variants, key=lambda item: item.uid)):
            new_name = name if index == 0 else f"{name}__variant_{index}"
            renamed.append({"uid": tool.uid, "name": new_name})
            mapped_names.add(new_name)
            if new_name != name:
                tool = _record_to_tool({**tool.to_dict(), "name": new_name})
            normalized.append(tool)
        mapping.setdefault(name, set()).update(mapped_names)
        collisions.append({"name": name, "count": len(variants), "renamed": renamed})
    return sorted(normalized, key=lambda tool: (tool.name, tool.uid)), collisions


def run_dataset(
    dataset: str,
    input_root: Path,
    output_root: Path,
    config: dict[str, Any],
    prompts: PromptStore,
    concurrency: int,
    resume: bool,
) -> dict[str, Any]:
    tools = load_tools(WORKSPACE / "datasets" / dataset / f"{dataset}_osa.json")
    components = json.loads(
        (input_root / dataset / "connected_components.json").read_text(encoding="utf-8")
    )
    output = output_root / dataset
    audit_path = output / "clusters.jsonl"
    completed = _read_jsonl(audit_path) if resume else {}
    if not resume and audit_path.exists():
        audit_path.unlink()
    by_uid = {tool.uid: tool for tool in tools}
    merge_components = [item for item in components if int(item["size"]) > 1]
    member_uids = {
        str(member["uid"])
        for component in merge_components
        for member in component["members"]
    }
    merged_tools = [tool for tool in tools if tool.uid not in member_uids]

    def process(component: dict[str, Any]) -> tuple[dict[str, Any], list[ToolRecord]]:
        component_id = str(component["component_id"])
        members = [by_uid[str(item["uid"])] for item in component["members"]]
        if component_id in completed:
            return completed[component_id], members
        pipeline = EntroTool(tools, config, prompt_store=prompts)
        corrected = pipeline._autocorrect(members)
        canonical = [
            pipeline._synthesize(group) if len(group) > 1 else group[0]
            for group in corrected
        ]
        return {
            "component_id": component_id,
            "members": [tool.uid for tool in members],
            "autocorrection": pipeline.autocorrection_audit,
            "corrected_groups": [[tool.uid for tool in group] for group in corrected],
            "documentation_merger": pipeline.documentation_audit,
            "canonical_tools": [tool.to_dict() for tool in canonical],
        }, members

    futures: dict[Any, dict[str, Any]] = {}
    rows: dict[str, tuple[dict[str, Any], list[ToolRecord]]] = {
        key: (row, [by_uid[uid] for uid in row["members"]])
        for key, row in completed.items()
    }
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        for component in merge_components:
            if str(component["component_id"]) not in completed:
                futures[executor.submit(process, component)] = component
        for future in as_completed(futures):
            component = futures[future]
            row, members = future.result()
            rows[str(component["component_id"])] = (row, members)
            _append_jsonl(audit_path, row)
            decision = row.get("autocorrection", [{}])[-1].get("response", "")
            print(
                f"[{dataset}] {component['component_id']} Auto-Correction: {decision}",
                flush=True,
            )

    mapping: dict[str, set[str]] = defaultdict(set)
    for tool in merged_tools:
        mapping[tool.name].add(tool.name)
    all_audits: list[dict[str, Any]] = []
    for component in merge_components:
        row, members = rows[str(component["component_id"])]
        canonical_tools = [_record_to_tool(item) for item in row["canonical_tools"]]
        merged_tools.extend(canonical_tools)
        all_audits.append(row)
        source_to_canonical: dict[str, str] = {}
        for canonical in canonical_tools:
            for spec in canonical.original.get("operation_specs", {}).values():
                if isinstance(spec, dict):
                    source_to_canonical[str(spec.get("source_uid", ""))] = (
                        canonical.name
                    )
        for member in members:
            mapping[member.name].add(source_to_canonical.get(member.uid, member.name))
    merged_tools, collisions = _normalize_name_collisions(merged_tools, mapping)
    _write_json(output / "merged_tools.json", [tool.to_dict() for tool in merged_tools])
    _write_json(
        output / "tool_mapping.json",
        {key: sorted(value) for key, value in sorted(mapping.items())},
    )
    summary = {
        "dataset": dataset,
        "input_tools": len(tools),
        "candidate_clusters": len(merge_components),
        "merged_tools": len(merged_tools),
        "reduction_percent": round(100 * (1 - len(merged_tools) / len(tools)), 6),
        "qwen_model": next(
            spec["model"]
            for spec in config["llms"]
            if spec.get("id") == "merge_autocorrection"
        ),
        "name_collision_groups": len(collisions),
        "files": {
            "clusters": "clusters.jsonl",
            "merged_tools": "merged_tools.json",
            "mapping": "tool_mapping.json",
        },
    }
    _write_json(output / "merge_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=["BFCL", "BFCL_PM", "SealTool"]
    )
    parser.add_argument(
        "--input-root", default=str(PROJECT / "outputs_graph_relaxed_v2")
    )
    parser.add_argument(
        "--output-root", default=str(PROJECT / "outputs_llm_merge_relaxed_v2")
    )
    parser.add_argument("--merge-concurrency", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = copy.deepcopy(load_config(PROJECT / "config.json"))
    _enable_merge_llms(config)
    prompts = PromptStore(PROJECT / "prompts")
    for dataset in args.datasets:
        run_dataset(
            dataset,
            Path(args.input_root),
            Path(args.output_root),
            config,
            prompts,
            args.merge_concurrency,
            args.resume,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
