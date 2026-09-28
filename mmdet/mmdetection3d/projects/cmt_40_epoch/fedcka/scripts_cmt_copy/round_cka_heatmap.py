"""
Compute per-round CKA scores using runner.py and visualize them as a heatmap.

Expected input folder layout:
- Any structure under --rounds_folder, containing .pth files
- Each round must map to exactly 4 checkpoints:
    1) unmerged pair (e.g. r1_modelA.pth, r1_modelB.pth)
    2) merged pair (e.g. r1_modelA_merged.pth, r1_modelB_merged.pth)
- Round id is detected from file path using patterns like: r1, round_1, round-1, Round1

Example (to compare model A and C)
-------
python round_cka_heatmap.py \
  --rounds_folder /path/to/round_checkpoints \
  --config /path/to/config.py \
  --data_dir /path/to/nuscenes_subset \
  --modality lidar_camera \
  --max_samples 10 \
  --output /path/to/cka_round_heatmap.png \
  --model_a modelA \
  --model_b modelC
"""

import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Keep import robust when launched from any working directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in os.sys.path:
    os.sys.path.insert(0, _THIS_DIR)

from runner import run_cka


def _extract_round_id(path: str) -> int:
    """Extract numeric round id from a checkpoint path."""
    normalized = os.path.basename(path).replace("\\", "/")
    # Accept forms like r1_modelA, round1_modelA, round_1_modelA, round-1_modelA.
    match = re.search(r"(?:^|[^a-z0-9])(?:r|round[_-]?)(\d+)(?=[^0-9]|$)", normalized, flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Could not detect round id in path: {path}. "
            "Use names containing r<number> or round<number>, e.g. r3_modelA.pth"
        )
    return int(match.group(1))


def _is_merged_checkpoint(path: str) -> bool:
    """Return True when a checkpoint is a merged model checkpoint."""
    name = os.path.basename(path).lower()
    return "_merged" in name or "-merged" in name


def _matches_model_tag(path: str, model_tag: str) -> bool:
    """Return True when basename contains the model tag as a standalone token."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    escaped = re.escape(model_tag.lower())
    return re.search(rf"(?:^|[^a-z0-9]){escaped}(?=[^a-z0-9]|$)", stem) is not None


def _collect_round_pairs(rounds_folder: str, model_a: str, model_b: str) -> List[Tuple[int, str, List[str]]]:
    """Find checkpoints and return sorted (round_id, state, [ckpt_a, ckpt_b]) rows.

    Row order per round is always: unmerged first, merged second.
    """
    all_ckpts: List[str] = []
    for root, _, files in os.walk(rounds_folder):
        for name in files:
            if name.endswith(".pth"):
                all_ckpts.append(os.path.join(root, name))

    if not all_ckpts:
        raise FileNotFoundError(f"No .pth files found in: {rounds_folder}")

    grouped_unmerged: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    grouped_merged: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for ckpt in sorted(all_ckpts):
        round_id = _extract_round_id(ckpt)
        abs_path = os.path.abspath(ckpt)
        if _matches_model_tag(abs_path, model_a):
            model_key = model_a
        elif _matches_model_tag(abs_path, model_b):
            model_key = model_b
        else:
            continue

        if _is_merged_checkpoint(abs_path):
            grouped_merged[round_id][model_key].append(abs_path)
        else:
            grouped_unmerged[round_id][model_key].append(abs_path)

    round_ids = sorted(set(grouped_unmerged.keys()) | set(grouped_merged.keys()))
    round_pairs: List[Tuple[int, str, List[str]]] = []
    for round_id in round_ids:
        unmerged_group = grouped_unmerged.get(round_id, {})
        merged_group = grouped_merged.get(round_id, {})

        unmerged_a = unmerged_group.get(model_a, [])
        unmerged_b = unmerged_group.get(model_b, [])
        merged_a = merged_group.get(model_a, [])
        merged_b = merged_group.get(model_b, [])

        if len(unmerged_a) != 1 or len(unmerged_b) != 1:
            raise ValueError(
                f"Round {round_id} expected exactly one unmerged checkpoint for each model "
                f"('{model_a}', '{model_b}'). Found {len(unmerged_a)} and {len(unmerged_b)}. "
                f"Files: {sorted(sum(unmerged_group.values(), []))}"
            )
        if len(merged_a) != 1 or len(merged_b) != 1:
            raise ValueError(
                f"Round {round_id} expected exactly one merged checkpoint for each model "
                f"('{model_a}', '{model_b}'). Found {len(merged_a)} and {len(merged_b)}. "
                f"Files: {sorted(sum(merged_group.values(), []))}"
            )

        # Enforce requested row order: unmerged first, merged second.
        round_pairs.append((round_id, "unmerged", [unmerged_a[0], unmerged_b[0]]))
        round_pairs.append((round_id, "merged", [merged_a[0], merged_b[0]]))

    if not round_pairs:
        raise FileNotFoundError(
            f"No matching checkpoints found for model tags '{model_a}' and '{model_b}' in {rounds_folder}"
        )

    return round_pairs


def _build_matrix(
    rows_per_round: Sequence[Tuple[int, str, Sequence[Tuple[int, str, float]]]]
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Convert per-round CKA tuples into a (1 - CKA) matrix and axis labels."""
    all_layers: List[str] = []
    for _, _, rows in rows_per_round:
        for _, layer_name, _ in rows:
            if layer_name not in all_layers:
                all_layers.append(layer_name)

    layer_to_col = {name: idx for idx, name in enumerate(all_layers)}
    matrix = np.full((len(rows_per_round), len(all_layers)), np.nan, dtype=np.float32)
    y_labels: List[str] = []

    for row_idx, (round_id, state, rows) in enumerate(rows_per_round):
        y_labels.append(f"Round {round_id} - {state}")
        for _, layer_name, score in rows:
            col_idx = layer_to_col[layer_name]
            matrix[row_idx, col_idx] = np.clip(1.0 - float(score), 0.0, 1.0)

    return matrix, y_labels, all_layers


def _plot_heatmap(matrix: np.ndarray, y_labels: List[str], x_labels: List[str], output: str) -> None:
    """Plot and save (1 - CKA) heatmap with red=0 and green=1."""
    fig_w = max(12, int(len(x_labels) * 0.25))
    fig_h = max(6, int(len(y_labels) * 0.45))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="RdYlGn", vmin=0.0, vmax=1.0)

    ax.set_title("1 - CKA Scores per Round (Red=0, Green=1)")
    ax.set_xlabel("Model Layers")
    ax.set_ylabel("Model Pair per Round")

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("1 - CKA Score")

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Round-wise CKA heatmap using runner.run_cka")
    parser.add_argument("--rounds_folder", required=True, type=str, help="Folder containing round checkpoint files")
    parser.add_argument("--config", required=True, type=str, help="Model config file")
    parser.add_argument("--data_dir", required=True, type=str, help="Input data directory for runner")
    parser.add_argument(
        "--modality",
        required=True,
        choices=["lidar", "camera", "lidar_camera"],
        help="Data modality for runner",
    )
    parser.add_argument("--max_samples", default=10, type=int, help="Max samples passed to run_cka")
    parser.add_argument("--output", default="round_cka_heatmap.png", type=str, help="Output image path")
    parser.add_argument("--output_dir", default="./temp_cka_rounds", type=str, help="Temp/output dir used by runner")
    parser.add_argument("--model_a", default="modelA", type=str, help="Model tag for first checkpoint in each pair")
    parser.add_argument("--model_b", default="modelB", type=str, help="Model tag for second checkpoint in each pair")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose runner output")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    rounds_folder = os.path.abspath(args.rounds_folder)
    if not os.path.isdir(rounds_folder):
        raise NotADirectoryError(f"rounds_folder not found: {rounds_folder}")

    if args.model_a.lower() == args.model_b.lower():
        raise ValueError("--model_a and --model_b must be different")

    round_pairs = _collect_round_pairs(rounds_folder, args.model_a, args.model_b)
    num_rounds = len(round_pairs) // 2
    print(f"Discovered {num_rounds} rounds and {len(round_pairs)} comparison rows")

    rows_per_round: List[Tuple[int, str, Sequence[Tuple[int, str, float]]]] = []
    for round_id, state, checkpoints in round_pairs:
        print(
            f"Running round {round_id} ({state}): "
            f"{os.path.basename(checkpoints[0])} vs {os.path.basename(checkpoints[1])}"
        )
        rows = run_cka(
            config=args.config,
            data_dir=args.data_dir,
            checkpoints=checkpoints,
            modality=args.modality,
            max_samples=args.max_samples,
            output_dir=os.path.join(args.output_dir, f"round_{round_id}_{state}"),
            verbose=args.verbose,
        )
        rows_per_round.append((round_id, state, rows))

    matrix, y_labels, x_labels = _build_matrix(rows_per_round)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _plot_heatmap(matrix, y_labels, x_labels, output_path)

    print(f"Saved heatmap: {output_path}")


if __name__ == "__main__":
    main()
