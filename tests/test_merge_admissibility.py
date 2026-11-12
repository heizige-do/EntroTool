import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from entrotool.toolscope_merger import (
    KGEdge,
    StructuralEntropyMerger,
    ToolRecord,
)


def test_relative_gate_preserves_duplicates_and_rejects_weaker_cross_intent_edges():
    tools = [
        ToolRecord(name=name, description=name, field="", arguments={})
        for name in ["area", "area_alt", "circumference", "circumference_alt"]
    ]
    edges = [
        KGEdge(
            source=tools[left].name,
            target=tools[right].name,
            weight=0.99 if (left, right) in {(0, 1), (2, 3)} else 0.93,
            relations={},
            evidence=(),
        )
        for left in range(4)
        for right in range(left + 1, 4)
    ]
    baseline = StructuralEntropyMerger(tools, edges).fit()
    constrained = StructuralEntropyMerger(
        tools,
        edges,
        {"merge_admissibility": {"enabled": True, "relative_margin": 0.02}},
    ).fit()
    assert {frozenset(group) for group in baseline.report_partition} == {
        frozenset({0, 1}),
        frozenset({2, 3}),
    }
    assert {frozenset(group) for group in constrained.report_partition} == {
        frozenset({0, 1}),
        frozenset({2, 3}),
    }
    assert {frozenset(group) for group in constrained.report_partition} == {
        frozenset({0, 1}),
        frozenset({2, 3}),
    }
    assert all(step["structural_gain"] > 0 for step in constrained.merge_history)


def test_relative_gate_prevents_transitive_bridge():
    tools = [
        ToolRecord(name=name, description=name, field="", arguments={})
        for name in ["first", "bridge", "last"]
    ]
    edges = [
        KGEdge(source="first", target="bridge", weight=0.99, relations={}, evidence=()),
        KGEdge(source="bridge", target="last", weight=0.98, relations={}, evidence=()),
    ]
    merger = StructuralEntropyMerger(
        tools,
        edges,
        {
            "min_structural_gain": -1,
            "automatic_refinement": {"enabled": False},
            "merge_admissibility": {"enabled": True},
        },
    ).fit()
    assert sorted(map(len, merger.report_partition)) == [1, 2]


def test_high_weight_equivalent_edges_are_processed_before_weaker_distractors():
    tools = [
        ToolRecord(
            name=name,
            description=description,
            field="",
            arguments={},
            uid=f"{name}@@{index:05d}",
        )
        for index, (name, description) in enumerate(
            [
                ("calculate_bmi", "Calculate body mass index from height and weight."),
                ("calculate_BMI", "Calculate body mass index from weight and height."),
                ("calculate_bmi", "Calculate BMI using a selected measurement system."),
                ("weight_in_space", "Calculate body weight on another planet."),
            ]
        )
    ]
    edges = [
        KGEdge(
            source=tools[0].uid,
            target=tools[1].uid,
            weight=0.974,
            relations={"semantic_contract_fallback": 1.0},
            evidence=(),
        ),
        KGEdge(
            source=tools[0].uid,
            target=tools[2].uid,
            weight=0.990,
            relations={"semantic_contract_fallback": 1.0},
            evidence=(),
        ),
        KGEdge(
            source=tools[1].uid,
            target=tools[2].uid,
            weight=0.976,
            relations={"semantic_contract_fallback": 1.0},
            evidence=(),
        ),
        KGEdge(
            source=tools[1].uid,
            target=tools[3].uid,
            weight=0.878,
            relations={},
            evidence=(),
        ),
    ]
    merger = StructuralEntropyMerger(
        tools,
        edges,
        {
            "min_structural_gain": -1,
            "merge_priority": "edge_weight_then_structural_gain",
            "merge_admissibility": {
                "enabled": True,
                "relative_margin": 2,
                "minimum_pair_cosine": 0.82,
                "minimum_anchor_overlap": 0.0,
                "enforce_action_compatibility": False,
            },
            "automatic_refinement": {"enabled": False},
        },
    ).fit()

    assert set(merger.merge_history[0]["leaves"]) == {"calculate_bmi"}
    bmi_cluster = next(group for group in merger.report_partition if 0 in group)
    assert bmi_cluster == {0, 1, 2}
    assert {3} in merger.report_partition


def test_anchor_gate_keeps_same_object_calculations_together():
    tools = [
        ToolRecord(name=name, description=description, field="", arguments={})
        for name, description in [
            ("calculate_triangle_area", "Calculate the area of a triangle."),
            ("compute_triangle_area", "Compute triangle area from its dimensions."),
        ]
    ]
    edges = [
        KGEdge(
            source=tools[0].name,
            target=tools[1].name,
            weight=0.99,
            relations={},
            evidence=(),
        )
    ]
    merger = StructuralEntropyMerger(
        tools,
        edges,
        {
            "min_structural_gain": -1,
            "merge_admissibility": {"enabled": True},
            "automatic_refinement": {"enabled": False},
        },
    ).fit()
    assert merger.report_partition == [{0, 1}]


def test_admissibility_accepts_a_direct_graph_edge():
    tools = [
        ToolRecord(name=name, description=description, field="", arguments={})
        for name, description in [
            ("calculate_bmi", "Calculate body mass index from height and weight."),
            ("calculate_diabetes_risk", "Calculate diabetes risk from medical data."),
        ]
    ]
    edges = [
        KGEdge(
            source=tools[0].name,
            target=tools[1].name,
            weight=0.99,
            relations={},
            evidence=(),
        )
    ]
    merger = StructuralEntropyMerger(
        tools,
        edges,
        {
            "min_structural_gain": -1,
            "merge_admissibility": {"enabled": True},
            "automatic_refinement": {"enabled": False},
        },
    ).fit()
    assert merger.report_partition == [{0, 1}]


def _quadratic_tools_and_edge():
    tools = [
        ToolRecord(
            name="algebra.quadratic_roots",
            description="Find the roots of a quadratic equation.",
            field="",
            arguments={},
        ),
        ToolRecord(
            name="solve_quadratic",
            description="Solve a quadratic equation and return its roots.",
            field="",
            arguments={},
        ),
    ]
    edge = KGEdge(
        source=tools[0].name,
        target=tools[1].name,
        weight=0.96,
        relations={"semantic_contract_fallback": 1.0},
        evidence=(),
    )
    return tools, [edge]


def test_refinement_action_compatibility_can_be_disabled():
    tools, edges = _quadratic_tools_and_edge()
    strict = StructuralEntropyMerger(
        tools,
        edges,
        {
            "min_structural_gain": -1,
            "merge_admissibility": {"enabled": False},
            "automatic_refinement": {
                "enabled": True,
                "minimum_pair_cosine": 0.82,
                "enforce_action_compatibility": True,
            },
        },
    ).fit()
    _, strict_components = strict.pruning_graph()
    relaxed = StructuralEntropyMerger(
        tools,
        edges,
        {
            "min_structural_gain": -1,
            "merge_admissibility": {"enabled": False},
            "automatic_refinement": {
                "enabled": True,
                "minimum_pair_cosine": 0.82,
                "enforce_action_compatibility": False,
            },
        },
    ).fit()
    graph, relaxed_components = relaxed.pruning_graph()
    assert sorted(map(len, strict_components)) == [1, 1]
    assert relaxed_components == [{0, 1}]
    assert graph["automatic_refinement"]["enforce_action_compatibility"] is False
