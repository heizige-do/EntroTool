from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_staged_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_staged_pipeline", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_retry_validity_check_does_not_change_default_audit_semantics() -> None:
    audit = {"error": None, "response": "not-a-candidate", "valid_response": False}

    assert MODULE._audit_succeeded(audit)
    assert not MODULE._selection_audit_succeeded(audit)


def test_retry_failed_selection_audits_only_retries_failed_steps() -> None:
    row = {
        "query": "find a film and a book",
        "retrieval": {
            "selected": ["find_film", "find_book"],
            "traces": [
                {
                    "reranker_scores": [
                        {"tool": "find_film", "score": 0.9},
                        {"tool": "find_book", "score": 0.1},
                    ]
                },
                {
                    "reranker_scores": [
                        {"tool": "find_film", "score": 0.2},
                        {"tool": "find_book", "score": 0.8},
                    ]
                },
            ],
        },
        "deepseek_selection_audit": {
            "5": [
                {
                    "step": "find a film",
                    "response": "find_film",
                    "valid_response": True,
                    "error": None,
                },
                {
                    "step": "find a book",
                    "response": "",
                    "valid_response": False,
                    "error": "temporary failure",
                },
            ]
        },
        "deepseek_selections": {"5": ["find_film"]},
        "metrics": {
            "5": {
                "candidates": ["find_film", "find_book"],
                "selected": ["find_film"],
            }
        },
    }
    calls: list[tuple[str, str, list[str], dict[str, float]]] = []

    def fake_select_one(query, step, candidates, scores):
        calls.append((query, step, list(candidates), dict(scores)))
        return "find_book", {
            "query": query,
            "step": step,
            "candidates": list(candidates),
            "response": "find_book",
            "valid_response": True,
            "error": None,
            "attempts": 1,
        }

    retried = MODULE._retry_failed_selection_audits(
        row, [5], {"find_film", "find_book"}, fake_select_one
    )

    assert retried == 1
    assert len(calls) == 1
    assert calls[0][1] == "find a book"
    assert calls[0][2] == ["find_film", "find_book"]
    assert calls[0][3] == {"find_film": 0.2, "find_book": 0.8}
    assert row["deepseek_selections"]["5"] == ["find_film", "find_book"]
    assert row["metrics"]["5"]["selected"] == ["find_film", "find_book"]
