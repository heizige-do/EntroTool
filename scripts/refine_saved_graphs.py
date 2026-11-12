"""Re-run entropy and automatic refinement from saved knowledge graphs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from entrotool.toolscope_merger import (  # noqa: E402
    KGEdge,
    StructuralEntropyMerger,
    load_config,
    load_tools,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=str(PROJECT / "outputs_mutual_contract"))
    parser.add_argument("--output-root", default=str(PROJECT / "outputs_auto_refined"))
    parser.add_argument(
        "--datasets", nargs="+", default=["BFCL", "BFCL_PM", "SealTool"]
    )
    args = parser.parse_args()
    config = load_config(PROJECT / "config.json")
    merger_config = config["merger"]
    for dataset in args.datasets:
        tools = load_tools(WORKSPACE / "datasets" / dataset / f"{dataset}_osa.json")
        graph_path = Path(args.input_root) / dataset / "knowledge_graph.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        edges = [
            KGEdge(
                source=edge["source"],
                target=edge["target"],
                weight=float(edge["weight"]),
                relations={key: float(value) for key, value in edge["relations"].items()},
                evidence=tuple(edge["evidence"]),
            )
            for edge in graph["edges_detail"]
        ]
        entropy = StructuralEntropyMerger(tools, edges, merger_config).fit()
        pruning_graph, components = entropy.pruning_graph()
        output = Path(args.output_root) / dataset
        output.mkdir(parents=True, exist_ok=True)
        (output / "entropy_pruning_graph.json").write_text(
            json.dumps(pruning_graph, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records = [
            {
                "component_id": f"component:{index:05d}",
                "size": len(component),
                "members": [tools[item].to_dict() for item in sorted(component)],
            }
            for index, component in enumerate(components)
        ]
        (output / "connected_components.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = {
            "dataset": dataset,
            "input_tools": len(tools),
            "input_graph_edges": len(edges),
            "entropy_partition_count": len(entropy.report_partition),
            "refined_component_count": len(components),
            "non_singleton_components": sum(len(item) > 1 for item in components),
            "tools_reduced_if_all_refined_components_merge": len(tools) - len(components),
            "automatic_refinement": entropy.refinement_stats,
        }
        (output / "merger_only_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
