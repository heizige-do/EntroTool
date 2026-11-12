"""Run only ToolScopeMerger through Connected Components.

This entry point never imports ToolScopeRetriever, never opens ``*_v.json``,
and never invokes an LLM. It is intended for structural-cluster inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from entrotool.toolscope_merger import (  # noqa: E402
    load_config,
    run_merger_only,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Structural-Entropy ToolScopeMerger only"
    )
    parser.add_argument("--dataset", required=True, help="dataset directory name")
    parser.add_argument("--dataset-root", default=str(WORKSPACE / "datasets"))
    parser.add_argument(
        "--tools-file",
        default=None,
        help="custom tool JSON filename inside dataset directory; defaults to <dataset>_osa.json",
    )
    parser.add_argument("--config", default=str(PROJECT / "config.json"))
    parser.add_argument("--output-root", default=str(PROJECT / "outputs_merger_only"))
    parser.add_argument(
        "--matched-toolscope-size",
        action="store_true",
        help="optional tool-count-matched control rather than natural entropy stopping",
    )
    parser.add_argument(
        "--show-largest",
        type=int,
        default=10,
        help="number of largest non-singleton components printed to the console",
    )
    args = parser.parse_args()
    output = Path(args.output_root) / args.dataset
    summary = run_merger_only(
        args.dataset_root,
        args.dataset,
        load_config(args.config),
        output,
        matched_toolscope_size=args.matched_toolscope_size,
        tools_file=args.tools_file,
    )
    components = json.loads(
        (output / "connected_components.json").read_text(encoding="utf-8")
    )
    largest = sorted(
        (item for item in components if item["size"] > 1),
        key=lambda item: (-int(item["size"]), str(item["component_id"])),
    )[: max(0, args.show_largest)]
    print(
        json.dumps(
            {**summary, "output": str(output.resolve())}, ensure_ascii=False, indent=2
        )
    )
    print("\nLARGEST NON-SINGLETON CONNECTED COMPONENTS")
    print("=" * 88)
    for item in largest:
        print(
            json.dumps(
                {
                    "component_id": item["component_id"],
                    "size": item["size"],
                    "internal_edge_count": item["internal_edge_count"],
                    "tools": [member["name"] for member in item["members"]],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    print(f"\nDetailed JSON: {(output / 'connected_components.json').resolve()}")
    print(f"Manual review: {(output / 'connected_components_review.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
