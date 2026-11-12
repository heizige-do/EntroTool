"""Create a shared-space t-SNE comparison of original and merged tools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from entrotool.toolscope_merger import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    KnowledgeGraphBuilder,
    load_config,
    load_tools,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize original and merged tool embeddings in a shared t-SNE space."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset key in config.json")
    parser.add_argument(
        "--merged-file",
        required=True,
        type=Path,
        help="Merged tools JSON file, such as merged_tools.json",
    )
    parser.add_argument(
        "--tools",
        type=Path,
        default=None,
        help="Override the original dataset *_osa.json file",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--device", default=None, help="Encoder device override")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.perplexity <= 0:
        raise ValueError("--perplexity must be positive")
    if args.clusters <= 0:
        raise ValueError("--clusters must be positive")

    config = load_config(PROJECT / "config.json")
    dataset_config = config.get("evaluation", {}).get("datasets", {}).get(args.dataset)
    if not dataset_config:
        known = ", ".join(sorted(config.get("evaluation", {}).get("datasets", {})))
        raise KeyError(f"Unknown dataset {args.dataset!r}; choose from: {known}")

    tools_path = args.tools or (
        WORKSPACE / "datasets" / args.dataset / str(dataset_config["tools"])
    )
    merged_path = args.merged_file
    if not tools_path.is_file():
        raise FileNotFoundError(f"Original tools file not found: {tools_path}")
    if not merged_path.is_file():
        raise FileNotFoundError(f"Merged tools file not found: {merged_path}")

    original_tools = load_tools(tools_path)
    merged_tools = load_tools(merged_path)
    semantic_config = config.get("merger", {}).get("semantic_graph", {})
    embedding_model = args.embedding_model or str(
        semantic_config.get("embedding_model", "thenlper/gte-large")
    )
    device = args.device or semantic_config.get("device")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install project requirements to use the embedding visualization script."
        ) from exc
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required; install it with `pip install matplotlib`."
        ) from exc

    document_builder = KnowledgeGraphBuilder(config.get("merger", {}))
    documents = [
        document_builder.tool_document(tool)
        for tool in [*original_tools, *merged_tools]
    ]
    encoder_options: dict[str, Any] = {}
    if device:
        encoder_options["device"] = device
    encoder = SentenceTransformer(embedding_model, **encoder_options)
    embeddings = encoder.encode(
        documents,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
        raise ValueError("Embedding model returned an invalid matrix shape")
    if len(documents) < 3:
        raise ValueError("At least three tools are required for t-SNE")
    if args.clusters > len(documents):
        raise ValueError("--clusters cannot exceed the total number of tools")

    effective_perplexity = min(args.perplexity, float(len(documents) - 1))
    cluster_ids = KMeans(
        n_clusters=args.clusters,
        random_state=args.random_state,
        n_init=10,
    ).fit_predict(embeddings)
    coordinates = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.random_state,
    ).fit_transform(embeddings)
    split = len(original_tools)
    original_xy = coordinates[:split]
    merged_xy = coordinates[split:]
    original_clusters = cluster_ids[:split]
    merged_clusters = cluster_ids[split:]

    output_path = args.output or merged_path.with_name(f"{merged_path.stem}_tsne.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    panels = [
        (axes[0], original_xy, original_clusters, f"Original {args.dataset} Toolset"),
        (
            axes[1],
            merged_xy,
            merged_clusters,
            f"Updated {args.dataset} Toolset",
        ),
    ]
    for axis, points, labels, title in panels:
        scatter = axis.scatter(
            points[:, 0],
            points[:, 1],
            s=18,
            alpha=0.82,
            c=labels,
            cmap="tab10",
            vmin=-0.5,
            vmax=args.clusters - 0.5,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel("Dimension 1")
        axis.set_ylabel("Dimension 2")
        axis.grid(color="#d8d8d8", linewidth=0.5, alpha=0.45)
        axis.spines[["top", "right"]].set_visible(False)
        colorbar = figure.colorbar(
            scatter, ax=axis, ticks=range(args.clusters), pad=0.04
        )
        colorbar.set_label("Cluster ID")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "dataset": args.dataset,
        "original_tools_file": str(tools_path),
        "merged_tools_file": str(merged_path),
        "original_tool_count": len(original_tools),
        "merged_tool_count": len(merged_tools),
        "embedding_model": embedding_model,
        "device": device,
        "cluster_method": "KMeans on joint normalized embeddings",
        "cluster_count": args.clusters,
        "perplexity": effective_perplexity,
        "random_state": args.random_state,
        "shared_joint_embedding_space": True,
        "output": str(output_path),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved t-SNE figure: {output_path}")
    print(f"Saved run metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
