from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_staged_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_staged_pipeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cluster_cache_is_isolated_by_autocorrection_mode(tmp_path: Path) -> None:
    audit_path = tmp_path / "clusters.jsonl"
    rows = [
        {"component_id": "component:00001", "autocorrection_enabled": True},
        {"component_id": "component:00002", "autocorrection_enabled": False},
        {"component_id": "component:legacy"},
    ]
    audit_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    enabled = MODULE._compatible_cluster_cache(audit_path, True)
    disabled = MODULE._compatible_cluster_cache(audit_path, False)

    assert set(enabled) == {"component:00001"}
    assert set(disabled) == {"component:00002"}


def test_autocorrection_is_controlled_by_config() -> None:
    enabled = {"merger": {"autocorrection": {"enabled": True}}}
    disabled = {"merger": {"autocorrection": {"enabled": False}}}

    assert MODULE._autocorrection_enabled(enabled, cli_disabled=False) is True
    assert MODULE._autocorrection_enabled(disabled, cli_disabled=False) is False
    assert MODULE._autocorrection_enabled(enabled, cli_disabled=True) is False
    assert MODULE._autocorrection_enabled({}, cli_disabled=False) is True
