from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "visualize_merged_tools.py"
SPEC = importlib.util.spec_from_file_location("visualize_merged_tools", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_groups_aligns_sources_and_canonical_tools() -> None:
    source_tools = {
        "alpha@@00000": {"uid": "alpha@@00000", "name": "alpha"},
        "beta@@00001": {"uid": "beta@@00001", "name": "beta"},
    }
    rows = [
        {
            "component_id": "component:00000",
            "members": ["alpha@@00000", "beta@@00001"],
            "autocorrection_enabled": False,
            "corrected_groups": [["alpha@@00000", "beta@@00001"]],
            "canonical_tools": [{"name": "alpha_beta"}],
        }
    ]

    groups, missing_count = MODULE.build_groups(rows, source_tools)

    assert missing_count == 0
    assert len(groups) == 1
    assert [tool["name"] for tool in groups[0]["sources"]] == ["alpha", "beta"]
    assert groups[0]["merged_tool"]["name"] == "alpha_beta"
    assert groups[0]["autocorrection_enabled"] is False


def test_build_groups_reports_missing_source_tool() -> None:
    rows = [
        {
            "component_id": "component:00000",
            "members": ["missing@@00000"],
            "corrected_groups": [["missing@@00000"]],
            "canonical_tools": [{"name": "missing"}],
        }
    ]

    groups, missing_count = MODULE.build_groups(rows, {})

    assert missing_count == 1
    assert groups[0]["sources"][0]["description"] == "Source tool not found"