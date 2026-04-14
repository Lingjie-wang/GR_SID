#!/usr/bin/env python3
"""Compute distribution metrics for semantic IDs (SIDs).

Examples:
    python -m src.analyze_sid_distribution \
        --input logs/inference/runs/2026-03-13/14-40-42/pickle/merged_predictions_tensor.pt \
        --num-layers 3 \
        --codebook-sizes 256,256,256

    python -m src.analyze_sid_distribution \
        --input some_run/pickle/merged_predictions.pkl \
        --prediction-key cluster_ids \
        --json-output sid_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Gini-based uniformity metrics for semantic IDs."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a SID tensor file (.pt) or merged predictions file (.pkl).",
    )
    parser.add_argument(
        "--prediction-key",
        default="cluster_ids",
        help="Prediction key to read from .pkl rows. Default: cluster_ids.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help=(
            "Number of SID layers to keep. Useful when the stored tensor includes an "
            "extra de-duplication column/row."
        ),
    )
    parser.add_argument(
        "--orientation",
        choices=("auto", "rows-as-items", "cols-as-items"),
        default="auto",
        help=(
            "Interpretation for .pt tensors. 'rows-as-items' means shape [N, L], "
            "'cols-as-items' means shape [L, N]."
        ),
    )
    parser.add_argument(
        "--drop-last-feature",
        choices=("auto", "yes", "no"),
        default="auto",
        help=(
            "Whether to drop the last SID feature. Use this when the file contains an "
            "extra duplicate-index feature created during de-duplication."
        ),
    )
    parser.add_argument(
        "--codebook-sizes",
        type=str,
        default=None,
        help=(
            "Optional comma-separated codebook sizes, e.g. 256,256,256. When provided, "
            "the script also reports per-layer coverage and full-space SID Gini "
            "(including zero-frequency SIDs)."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many most/least frequent SIDs or clusters to print. Default: 10.",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help=(
            "Optional path to write all metrics as JSON. If omitted, the script writes "
            "to the input file's parent run directory."
        ),
    )
    return parser.parse_args()


def parse_codebook_sizes(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--codebook-sizes cannot be empty.")
    sizes = [int(value) for value in values]
    if any(size <= 0 for size in sizes):
        raise ValueError("--codebook-sizes must be positive integers.")
    return sizes


def to_python_int_list(values: Iterable[Any]) -> list[int]:
    return [int(v) for v in values]


def to_2d_long_tensor(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D tensor-like value, got shape {tuple(tensor.shape)}.")
    return tensor.long()


def load_from_pickle(path: Path, prediction_key: str) -> torch.Tensor:
    with path.open("rb") as file:
        data = pickle.load(file)

    if isinstance(data, list):
        if not data:
            raise ValueError(f"Pickle file is empty: {path}")

        first_row = data[0]
        if isinstance(first_row, dict):
            if prediction_key not in first_row:
                raise KeyError(
                    f"Key '{prediction_key}' not found in pickle rows. "
                    f"Available keys: {sorted(first_row.keys())}"
                )
            rows = [torch.as_tensor(row[prediction_key]).view(-1) for row in data]
            return torch.stack(rows, dim=0).long()

        rows = [torch.as_tensor(row).view(-1) for row in data]
        return torch.stack(rows, dim=0).long()

    if isinstance(data, dict):
        if prediction_key not in data:
            raise KeyError(
                f"Key '{prediction_key}' not found in pickle dict. "
                f"Available keys: {sorted(data.keys())}"
            )
        return to_2d_long_tensor(data[prediction_key])

    return to_2d_long_tensor(data)


def infer_item_axis(shape: Sequence[int], orientation: str) -> int:
    if len(shape) != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(shape)}.")

    if orientation == "rows-as-items":
        return 0
    if orientation == "cols-as-items":
        return 1

    rows, cols = shape
    if rows <= 64 and cols > 64:
        return 1
    if cols <= 64 and rows > 64:
        return 0
    if rows >= cols:
        return 0
    return 1


def should_drop_last_feature(
    sid_matrix: torch.Tensor, drop_last_feature: str, num_layers: int | None
) -> bool:
    feature_dim = sid_matrix.shape[1]
    if num_layers is not None:
        if num_layers <= 0 or num_layers > feature_dim:
            raise ValueError(
                f"--num-layers must be in [1, {feature_dim}], got {num_layers}."
            )
        return num_layers < feature_dim

    if drop_last_feature == "yes":
        return feature_dim >= 2
    if drop_last_feature == "no":
        return False
    if feature_dim < 2:
        return False

    unique_counts = [int(torch.unique(sid_matrix[:, idx]).numel()) for idx in range(feature_dim)]
    last_unique = unique_counts[-1]
    prev_uniques = unique_counts[:-1]
    if not prev_uniques:
        return False

    prev_median = sorted(prev_uniques)[len(prev_uniques) // 2]
    last_feature = sid_matrix[:, -1]
    last_min = int(last_feature.min().item())
    last_max = int(last_feature.max().item())

    # GRID's de-duplication suffix typically has far fewer unique values than real SID layers.
    return bool(
        last_min >= 0
        and last_unique <= max(32, prev_median // 4)
        and last_max <= sid_matrix.shape[0]
    )


def load_sid_matrix(
    path: Path,
    prediction_key: str,
    orientation: str,
    drop_last_feature: str,
    num_layers: int | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    if path.suffix == ".pt":
        raw_tensor = torch.load(path, map_location="cpu")
        if not isinstance(raw_tensor, torch.Tensor):
            raise TypeError(
                f"Expected a torch.Tensor in {path}, got {type(raw_tensor).__name__}."
            )
        raw_tensor = raw_tensor.long()
    elif path.suffix == ".pkl":
        raw_tensor = load_from_pickle(path, prediction_key)
    else:
        raise ValueError("Unsupported input format. Please provide a .pt or .pkl file.")

    item_axis = infer_item_axis(raw_tensor.shape, orientation)
    sid_matrix = raw_tensor if item_axis == 0 else raw_tensor.transpose(0, 1)

    metadata: dict[str, Any] = {
        "input_path": str(path),
        "input_format": path.suffix,
        "raw_shape": list(raw_tensor.shape),
        "interpreted_shape_before_trim": list(sid_matrix.shape),
        "orientation_used": "rows-as-items" if item_axis == 0 else "cols-as-items",
    }

    drop_last = should_drop_last_feature(sid_matrix, drop_last_feature, num_layers)
    if num_layers is not None:
        sid_matrix = sid_matrix[:, :num_layers]
    elif drop_last:
        sid_matrix = sid_matrix[:, :-1]

    metadata["drop_last_feature"] = drop_last
    metadata["sid_shape"] = list(sid_matrix.shape)
    return sid_matrix.contiguous().long(), metadata


def gini_from_counts(counts: Iterable[int]) -> float:
    positive_counts = sorted(int(count) for count in counts if int(count) > 0)
    if not positive_counts:
        return 0.0

    num_distinct_ids = len(positive_counts)
    total = sum(positive_counts)
    cumulative = 0
    gap_sum = 0.0

    for index, count in enumerate(positive_counts, start=1):
        cumulative += count
        lorenz = cumulative / total
        gap_sum += index / num_distinct_ids - lorenz

    return 2.0 / num_distinct_ids * gap_sum


def gini_with_zeros(counts: Iterable[int], total_bins: int) -> float | None:
    positive_counts = sorted(int(count) for count in counts if int(count) > 0)
    if total_bins <= 0:
        raise ValueError("total_bins must be positive.")
    if len(positive_counts) > total_bins:
        raise ValueError("Observed non-zero bins exceed total_bins.")
    if not positive_counts:
        return 0.0

    total = sum(positive_counts)
    zero_bins = total_bins - len(positive_counts)
    cumulative = 0
    gap_sum = 0.0

    if zero_bins > 0:
        gap_sum += zero_bins * (zero_bins + 1) / (2.0 * total_bins)

    for offset, count in enumerate(positive_counts, start=1):
        index = zero_bins + offset
        cumulative += count
        lorenz = cumulative / total
        gap_sum += index / total_bins - lorenz

    return 2.0 / total_bins * gap_sum


def sid_counter_from_matrix(sid_matrix: torch.Tensor) -> Counter[tuple[int, ...]]:
    rows = [tuple(to_python_int_list(row.tolist())) for row in sid_matrix]
    return Counter(rows)


def summarize_counter(counter: Counter[Any], top_k: int) -> dict[str, Any]:
    counts = sorted(counter.values())
    if not counts:
        return {
            "unique_count": 0,
            "gini": 0.0,
            "min_count": 0,
            "max_count": 0,
            "avg_count": 0.0,
            "top_k": [],
            "bottom_k": [],
        }

    top_items = counter.most_common(top_k)
    bottom_items = sorted(counter.items(), key=lambda item: (item[1], item[0]))[:top_k]

    return {
        "unique_count": len(counter),
        "gini": gini_from_counts(counter.values()),
        "min_count": int(counts[0]),
        "max_count": int(counts[-1]),
        "avg_count": float(sum(counts) / len(counts)),
        "top_k": top_items,
        "bottom_k": bottom_items,
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return json_ready(value.item())
        return json_ready(value.tolist())
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return value


def default_json_output_path(input_path: Path) -> Path:
    parent = input_path.parent
    if parent.name in {"pickle", "csv"} and parent.parent != parent:
        parent = parent.parent
    return parent / "sid_distribution_metrics.json"


def compute_metrics(
    sid_matrix: torch.Tensor, top_k: int, codebook_sizes: list[int] | None
) -> dict[str, Any]:
    sid_counts = sid_counter_from_matrix(sid_matrix)
    sid_summary = summarize_counter(sid_counts, top_k=top_k)

    metrics: dict[str, Any] = {
        "num_items": int(sid_matrix.shape[0]),
        "num_layers": int(sid_matrix.shape[1]),
        "num_unique_sid": int(sid_summary["unique_count"]),
        "fraction_unique_sid": float(sid_summary["unique_count"] / sid_matrix.shape[0]),
        "sid_gini_observed": float(sid_summary["gini"]),
        "sid_count_stats": {
            "min": sid_summary["min_count"],
            "max": sid_summary["max_count"],
            "avg": sid_summary["avg_count"],
        },
        "top_sid": sid_summary["top_k"],
        "bottom_sid": sid_summary["bottom_k"],
        "per_layer": [],
    }

    if codebook_sizes is not None:
        if len(codebook_sizes) != sid_matrix.shape[1]:
            raise ValueError(
                f"--codebook-sizes has {len(codebook_sizes)} entries but SID has "
                f"{sid_matrix.shape[1]} layers."
            )
        total_sid_space = math.prod(codebook_sizes)
        metrics["sid_space_size"] = int(total_sid_space)
        metrics["sid_coverage"] = float(sid_summary["unique_count"] / total_sid_space)
        metrics["sid_gini_full_space"] = float(
            gini_with_zeros(sid_counts.values(), total_bins=total_sid_space)
        )

    for layer_idx in range(sid_matrix.shape[1]):
        layer_counter: Counter[int] = Counter(to_python_int_list(sid_matrix[:, layer_idx].tolist()))
        layer_summary = summarize_counter(layer_counter, top_k=top_k)
        layer_metrics: dict[str, Any] = {
            "layer_index": layer_idx,
            "unique_clusters": int(layer_summary["unique_count"]),
            "gini": float(layer_summary["gini"]),
            "count_stats": {
                "min": layer_summary["min_count"],
                "max": layer_summary["max_count"],
                "avg": layer_summary["avg_count"],
            },
            "top_clusters": layer_summary["top_k"],
            "bottom_clusters": layer_summary["bottom_k"],
        }
        if codebook_sizes is not None:
            layer_metrics["codebook_size"] = int(codebook_sizes[layer_idx])
            layer_metrics["coverage"] = float(
                layer_summary["unique_count"] / codebook_sizes[layer_idx]
            )
        metrics["per_layer"].append(layer_metrics)

    return metrics


def print_human_readable(metadata: dict[str, Any], metrics: dict[str, Any]) -> None:
    print("SID Distribution Metrics")
    print("=" * 80)
    print(f"input_path: {metadata['input_path']}")
    print(f"input_format: {metadata['input_format']}")
    print(f"raw_shape: {tuple(metadata['raw_shape'])}")
    print(
        "interpreted_sid_shape: "
        f"{tuple(metadata['sid_shape'])} ({metadata['orientation_used']})"
    )
    print(f"drop_last_feature: {metadata['drop_last_feature']}")
    print()

    print("Overall SID metrics")
    print("-" * 80)
    print(f"num_items: {metrics['num_items']}")
    print(f"num_layers: {metrics['num_layers']}")
    print(f"num_unique_sid: {metrics['num_unique_sid']}")
    print(f"fraction_unique_sid: {metrics['fraction_unique_sid']:.12f}")
    print(f"sid_gini_observed: {metrics['sid_gini_observed']:.12f}")
    if "sid_gini_full_space" in metrics:
        print(f"sid_space_size: {metrics['sid_space_size']}")
        print(f"sid_coverage: {metrics['sid_coverage']:.12f}")
        print(f"sid_gini_full_space: {metrics['sid_gini_full_space']:.12f}")
    print(f"sid_count_min: {metrics['sid_count_stats']['min']}")
    print(f"sid_count_max: {metrics['sid_count_stats']['max']}")
    print(f"sid_count_avg: {metrics['sid_count_stats']['avg']:.12f}")
    print(f"top_sid: {metrics['top_sid']}")
    print(f"bottom_sid: {metrics['bottom_sid']}")
    print()

    print("Per-layer metrics")
    print("-" * 80)
    for layer in metrics["per_layer"]:
        print(f"layer_{layer['layer_index']}_unique_clusters: {layer['unique_clusters']}")
        print(f"layer_{layer['layer_index']}_gini: {layer['gini']:.12f}")
        if "coverage" in layer:
            print(f"layer_{layer['layer_index']}_coverage: {layer['coverage']:.12f}")
        print(f"layer_{layer['layer_index']}_count_min: {layer['count_stats']['min']}")
        print(f"layer_{layer['layer_index']}_count_max: {layer['count_stats']['max']}")
        print(f"layer_{layer['layer_index']}_count_avg: {layer['count_stats']['avg']:.12f}")
        print(f"layer_{layer['layer_index']}_top_clusters: {layer['top_clusters']}")
        print(f"layer_{layer['layer_index']}_bottom_clusters: {layer['bottom_clusters']}")
        print()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    codebook_sizes = parse_codebook_sizes(args.codebook_sizes)

    sid_matrix, metadata = load_sid_matrix(
        path=input_path,
        prediction_key=args.prediction_key,
        orientation=args.orientation,
        drop_last_feature=args.drop_last_feature,
        num_layers=args.num_layers,
    )
    metrics = compute_metrics(
        sid_matrix=sid_matrix,
        top_k=args.top_k,
        codebook_sizes=codebook_sizes,
    )

    print_human_readable(metadata=metadata, metrics=metrics)

    output_path = Path(args.json_output) if args.json_output else default_json_output_path(
        input_path
    )
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(json_ready({"metadata": metadata, "metrics": metrics}), indent=2),
            encoding="utf-8",
        )
        print(f"json_output: {output_path}")


if __name__ == "__main__":
    main()
