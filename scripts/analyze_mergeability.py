"""Score merger-only components by automatic protocol-coverage proxies."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ACTION_WORDS = {
    "get", "fetch", "find", "search", "retrieve", "query", "list", "look",
    "calculate", "compute", "create", "add", "update", "delete", "remove",
    "book", "cancel", "check", "set", "send", "generate", "analyze",
}


def actions(name: str, description: str) -> set[str]:
    words = set(re.findall(r"[A-Za-z]+", f"{name} {description}".lower()))
    return words & ACTION_WORDS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "outputs_mutual_contract"))
    parser.add_argument(
        "--datasets", nargs="+", default=["BFCL", "BFCL_PM", "SealTool"]
    )
    parser.add_argument("--min-density", type=float, default=0.5)
    parser.add_argument("--min-core-coverage", type=float, default=0.5)
    parser.add_argument("--max-inflation", type=float, default=3.0)
    args = parser.parse_args()
    root = Path(args.root)
    for dataset in args.datasets:
        path = root / dataset / "connected_components.json"
        components = json.loads(path.read_text(encoding="utf-8"))
        scored = []
        for component in components:
            members = component["members"]
            if len(members) < 2:
                continue
            required = [set(member["required_arguments"]) for member in members]
            union = set().union(*required)
            core = set.intersection(*required)
            mean_required = sum(len(item) for item in required) / len(required)
            density = component["internal_edge_count"] / (len(members) * (len(members) - 1) / 2)
            inflation = len(union) / mean_required if mean_required else 1.0
            action_sets = [actions(member["name"], member["description"]) for member in members]
            common_actions = set.intersection(*action_sets) if all(action_sets) else set()
            action_consistent = len(common_actions) > 0
            pass_gate = (
                density >= args.min_density
                and len(core) / len(union) >= args.min_core_coverage if union else density >= args.min_density
            ) and inflation <= args.max_inflation and action_consistent
            scored.append({
                "component_id": component["component_id"],
                "size": len(members),
                "internal_edge_density": round(density, 6),
                "required_core_coverage": round(len(core) / len(union), 6) if union else 1.0,
                "schema_inflation": round(inflation, 6),
                "common_actions": sorted(common_actions),
                "action_consistent": action_consistent,
                "auto_merge_gate": bool(pass_gate),
                "tools": [member["name"] for member in members],
            })
        accepted = [row for row in scored if row["auto_merge_gate"]]
        report = {
            "dataset": dataset,
            "non_singleton_components": len(scored),
            "auto_mergeable_components": len(accepted),
            "mergeable_component_rate": round(len(accepted) / len(scored), 6) if scored else 0.0,
            "tools_reduced_if_accepted_only": sum(row["size"] - 1 for row in accepted),
            "gate": {
                "min_density": args.min_density,
                "min_core_coverage": args.min_core_coverage,
                "max_inflation": args.max_inflation,
                "requires_common_action": True,
            },
            "components": sorted(scored, key=lambda row: (-row["size"], row["component_id"])),
        }
        output = root / dataset / "mergeability_report.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: report[key] for key in report if key != "components"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
