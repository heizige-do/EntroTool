"""Judge finalized non-singleton tool merges with a configurable LLM."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from tqdm.auto import tqdm

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
EVALUATION_SCHEMA_VERSION = 7
sys.path.insert(0, str(PROJECT / "src"))

from entrotool.toolscope_merger import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    OpenAICompatibleLLM,
    load_config,
    load_tools,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        rows.append(value)
    return rows


def append_jsonl(path: Path, value: Mapping[str, Any], lock: Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def tool_block(tool: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"UID: {tool.get('uid', '')}",
            f"NAME: {tool.get('name', '')}",
            f"SIGNATURE: {tool.get('signature', '')}",
            f"DESCRIPTION: {tool.get('description', '')}",
            "ARGUMENTS: " + json.dumps(tool.get("arguments", {}), ensure_ascii=False),
        ]
    )


def merged_block(tool: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"NAME: {tool.get('name', '')}",
            f"SIGNATURE: {tool.get('signature', '')}",
            f"DOCUMENTATION:\n{tool.get('description', '')}",
            "ARGUMENTS: " + json.dumps(tool.get("arguments", {}), ensure_ascii=False),
        ]
    )


def build_reviews(
    merge_file: Path,
    tools_file: Path,
    prompt_template: str,
) -> list[dict[str, Any]]:
    tools = {tool.uid: tool.to_dict() for tool in load_tools(tools_file)}
    reviews: list[dict[str, Any]] = []
    for row in read_jsonl(merge_file):
        groups = row.get("corrected_groups", [])
        canonical_tools = row.get("canonical_tools", [])
        if len(groups) != len(canonical_tools):
            raise ValueError(
                f"{row.get('component_id')}: corrected_groups and canonical_tools differ in length"
            )
        for group_index, (group, canonical) in enumerate(zip(groups, canonical_tools)):
            member_uids = [str(uid) for uid in group]
            if len(member_uids) <= 1:
                continue
            missing = [uid for uid in member_uids if uid not in tools]
            if missing:
                raise KeyError(f"Missing source tools for {missing}")
            source_tools = [tools[uid] for uid in member_uids]
            source_block = "\n\n--- SOURCE TOOL ---\n".join(
                tool_block(tool) for tool in source_tools
            )
            prompt = prompt_template.replace("{SOURCE_TOOLS_BLOCK}", source_block)
            prompt = prompt.replace("{MERGED_TOOL_BLOCK}", merged_block(canonical))
            component_id = str(row.get("component_id", ""))
            reviews.append(
                {
                    "review_id": f"{component_id}:group:{group_index:03d}",
                    "component_id": component_id,
                    "group_index": group_index,
                    "member_uids": member_uids,
                    "source_tools": source_tools,
                    "merged_tool": canonical,
                    "prompt": prompt,
                }
            )
    return reviews


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-file", required=True, help="clusters.jsonl to judge")
    parser.add_argument("--dataset", required=True, help="dataset key in config.json")
    parser.add_argument("--tools", default=None, help="override source *_osa.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-id", default="merge_quality_judge")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    merge_file = Path(args.merge_file)
    output = (
        Path(args.output)
        if args.output
        else merge_file.with_name("merge_quality_judgments.jsonl")
    )
    config = load_config(PROJECT / "config.json")
    dataset_config = config.get("evaluation", {}).get("datasets", {}).get(args.dataset)
    if not dataset_config and not args.tools:
        raise KeyError(f"Unknown dataset {args.dataset!r}; pass --tools explicitly")
    tools_file = (
        Path(args.tools)
        if args.tools
        else (WORKSPACE / "datasets" / args.dataset / str(dataset_config["tools"]))
    )
    prompt_template = (PROJECT / "prompts" / "merge_quality_judge.txt").read_text(
        encoding="utf-8"
    )
    reviews = build_reviews(merge_file, tools_file, prompt_template)
    if args.max_groups is not None:
        reviews = reviews[: args.max_groups]

    completed = (
        {
            str(row.get("review_id")): row
            for row in read_jsonl(output)
            if row.get("review_id")
            and row.get("evaluation_schema_version") == EVALUATION_SCHEMA_VERSION
            and not row.get("error")
            and isinstance(row.get("merge_reasonable"), bool)
            and isinstance(row.get("functionally_complete"), bool)
        }
        if args.resume and output.exists()
        else {}
    )
    if not args.resume and output.exists():
        output.unlink()

    spec = next(
        (
            dict(item)
            for item in config.get("llms", [])
            if item.get("id") == args.model_id
        ),
        None,
    )
    if spec is None:
        raise KeyError(f"Unknown LLM id {args.model_id!r}")
    spec["enabled"] = True
    write_lock = Lock()

    def judge(review: dict[str, Any]) -> dict[str, Any]:
        if args.dry_run:
            return {
                **review,
                "model_id": args.model_id,
                "model": spec.get("model"),
                "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                "response": None,
                "parsed": None,
                "merge_reasonable": None,
                "functionally_complete": None,
                "error": None,
                "attempts": 0,
                "dry_run": True,
            }
        llm = OpenAICompatibleLLM(spec)
        last_error: Exception | None = None
        for attempt in range(1, max(1, args.retries) + 1):
            try:
                response = llm.complete(review["prompt"])
                parsed = parse_json_object(response)
                merge_reasonable = (parsed or {}).get("merge_reasonable")
                functionally_complete = (parsed or {}).get("functionally_complete")
                if not isinstance(merge_reasonable, bool) or not isinstance(
                    functionally_complete, bool
                ):
                    raise ValueError(
                        "Judge must return boolean merge_reasonable and functionally_complete"
                    )
                return {
                    **review,
                    "model_id": args.model_id,
                    "model": spec.get("model"),
                    "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                    "response": response,
                    "parsed": parsed,
                    "merge_reasonable": merge_reasonable,
                    "functionally_complete": functionally_complete,
                    "error": None,
                    "attempts": attempt,
                    "dry_run": False,
                }
            except Exception as exc:
                last_error = exc
                if attempt < max(1, args.retries):
                    time.sleep(min(2 ** (attempt - 1), 8))
        return {
            **review,
            "model_id": args.model_id,
            "model": spec.get("model"),
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "response": "",
            "parsed": None,
            "merge_reasonable": None,
            "functionally_complete": None,
            "error": repr(last_error),
            "attempts": max(1, args.retries),
            "dry_run": False,
        }

    pending = [item for item in reviews if item["review_id"] not in completed]
    results = dict(completed)
    completed_ids = {item["review_id"] for item in reviews} & set(completed)
    reasonable_count = sum(
        completed[review_id].get("merge_reasonable") is True
        for review_id in completed_ids
    )
    complete_count = sum(
        completed[review_id].get("functionally_complete") is True
        for review_id in completed_ids
    )
    error_count = sum(
        bool(completed[review_id].get("error")) for review_id in completed_ids
    )
    with tqdm(
        total=len(reviews),
        initial=len(completed_ids),
        desc="Merge quality judge",
        unit="merge",
        dynamic_ncols=True,
    ) as progress:
        progress.set_postfix(
            reasonable=reasonable_count,
            complete=complete_count,
            errors=error_count,
        )
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = {executor.submit(judge, item): item for item in pending}
            for future in as_completed(futures):
                result = future.result()
                results[result["review_id"]] = result
                append_jsonl(output, result, write_lock)
                reasonable_count += int(result.get("merge_reasonable") is True)
                complete_count += int(result.get("functionally_complete") is True)
                error_count += int(bool(result.get("error")))
                progress.set_postfix(
                    reasonable=reasonable_count,
                    complete=complete_count,
                    errors=error_count,
                )
                progress.update(1)

    ordered = [results[item["review_id"]] for item in reviews]
    output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered),
        encoding="utf-8",
    )
    judged = [
        item
        for item in ordered
        if item.get("merge_reasonable") is not None
        and item.get("functionally_complete") is not None
    ]
    summary = {
        "merge_file": str(merge_file),
        "dataset": args.dataset,
        "model_id": args.model_id,
        "model": spec.get("model"),
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "eligible_non_singleton_merges": len(reviews),
        "judged": len(judged),
        "merge_reasonable_true": sum(
            item["merge_reasonable"] is True for item in judged
        ),
        "functionally_complete_true": sum(
            item["functionally_complete"] is True for item in judged
        ),
        "both_true": sum(
            item["merge_reasonable"] is True and item["functionally_complete"] is True
            for item in judged
        ),
        "errors": sum(bool(item.get("error")) for item in ordered),
        "dry_run": args.dry_run,
        "output": str(output),
    }
    output.with_name("merge_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
