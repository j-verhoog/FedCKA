"""
CKA runner utility.

This module exposes a reusable API and CLI to compute CKA layer scores between
exactly two checkpoints, returning a list of tuples:
    (layer_index, layer_name, layer_cka_score)

It accepts the same key runtime arguments as main.py so it can be called from
any working directory.

Examples
--------
CLI usage from any directory:
    python "/YOUR_HOMEDIR_PATH_HERE/mmdet/mmdetection3d/projects/cmt_40_epoch/fedselect/scripts_cmt_copy/runner.py" \
        --config /path/to/config.py \
        --data_dir /path/to/data \
        --pt /path/to/modelA.pth /path/to/modelB.pth \
        --modality lidar_camera \
        --max_samples 10

Import usage in another Python file:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cka_runner",
        "/YOUR_HOMEDIR_PATH_HERE/mmdet/mmdetection3d/projects/cmt_40_epoch/fedselect/scripts_cmt_copy/runner.py",
    )
    cka_runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cka_runner)

    tuples = cka_runner.run_cka(
        config="/path/to/config.py",
        data_dir="/path/to/data",
        checkpoints=["/path/to/modelA.pth", "/path/to/modelB.pth"],
        modality="lidar_camera",
        max_samples=10,
        output_dir="./results"
    )
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from typing import Any, List, Sequence, Tuple


# Make local sibling imports robust even when launched from another cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Keep compatibility with existing container layouts used by this project.
_INTERNAL_PATH = "/opt/src/mmdetection3d"
if _INTERNAL_PATH not in sys.path:
    sys.path.insert(0, _INTERNAL_PATH)


def _collect_multimodal_samples(data_dir: str, max_samples: int) -> List[Tuple[str, str]]:
    """Collect paired lidar+camera files from NuScenes-like layout."""
    lidar_dir = os.path.join(data_dir, "samples", "LIDAR_TOP")
    camera_dir = os.path.join(data_dir, "samples", "CAM_FRONT")

    if not os.path.isdir(lidar_dir) or not os.path.isdir(camera_dir):
        raise ValueError(
            "Expected NuScenes structure with samples/LIDAR_TOP and samples/CAM_FRONT"
        )

    lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.bin")))
    camera_files = sorted(glob.glob(os.path.join(camera_dir, "*.jpg")))
    if len(lidar_files) == 0 or len(camera_files) == 0:
        raise ValueError(f"No lidar (.bin) or camera (.jpg) files found under {data_dir}")

    def extract_scene_timestamp(filepath: str) -> Tuple[str, str]:
        basename = os.path.basename(filepath)
        parts = basename.split("__")
        if len(parts) < 3:
            return "", ""
        scene_id = parts[0]
        timestamp_part = parts[2]
        while "." in timestamp_part:
            timestamp_part = os.path.splitext(timestamp_part)[0]
        return scene_id, timestamp_part

    camera_lookup = {}
    for cam_file in camera_files:
        scene_id, timestamp = extract_scene_timestamp(cam_file)
        if scene_id and timestamp:
            camera_lookup[(scene_id, timestamp)] = cam_file

    paired_samples: List[Tuple[str, str]] = []
    for lidar_file in lidar_files:
        scene_id, lidar_timestamp = extract_scene_timestamp(lidar_file)
        if not scene_id or not lidar_timestamp:
            continue

        best_camera_file = None
        min_time_diff = float("inf")
        lidar_ts = int(lidar_timestamp)

        for (cam_scene_id, cam_timestamp), cam_file in camera_lookup.items():
            if cam_scene_id != scene_id:
                continue
            cam_ts = int(cam_timestamp)
            time_diff = abs(lidar_ts - cam_ts)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                best_camera_file = cam_file

        if best_camera_file is not None:
            paired_samples.append((lidar_file, best_camera_file))

    def keep_train_samples(samples: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Filter pairs using the official NuScenes train scene split.

        This helper is deliberately fail-open: datasets without usable NuScenes
        metadata retain the previous sampling behavior.
        """
        try:
            from nuscenes.nuscenes import NuScenes
            from nuscenes.utils import splits

            version = None
            for candidate in ("v1.0-trainval", "v1.0-mini"):
                if os.path.isfile(os.path.join(data_dir, candidate, "sample_data.json")):
                    version = candidate
                    break
                if os.path.isfile(os.path.join(data_dir, candidate, "scene.json")):
                    version = candidate
                    break

            if version is None:
                return samples

            nusc = NuScenes(version=version, dataroot=data_dir, verbose=False)
            train_scene_names = set(
                splits.mini_train if version == "v1.0-mini" else splits.train
            )
            sample_data_by_filename = {
                os.path.basename(record["filename"]): record
                for record in nusc.sample_data
            }

            train_samples = []
            for lidar_file, camera_file in samples:
                lidar_record = sample_data_by_filename[os.path.basename(lidar_file)]
                sample_record = nusc.get("sample", lidar_record["sample_token"])
                scene_record = nusc.get("scene", sample_record["scene_token"])
                if scene_record["name"] in train_scene_names:
                    train_samples.append((lidar_file, camera_file))

            return train_samples
        except Exception as exc:
            print(
                f"Warning: Could not apply NuScenes train split filter ({exc}). "
                "Using all paired samples."
            )
            return samples

    # Filter before shuffling and applying max_samples so validation samples do
    # not consume the requested training sample budget.
    paired_samples = keep_train_samples(paired_samples)

    random.shuffle(paired_samples)

    if max_samples > 0:
        paired_samples = paired_samples[:max_samples]

    if len(paired_samples) == 0:
        raise ValueError(f"No paired lidar+camera files found in {data_dir}")

    return paired_samples


def _collect_samples(data_dir: str, modality: str, max_samples: int) -> List[Any]:
    """Collect sample paths in the same style as main.py."""
    if modality == "camera":
        img_files = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        if len(img_files) == 0:
            raise ValueError(f"No .jpg files found in {data_dir}")
        return img_files[:max_samples] if max_samples > 0 else img_files

    if modality == "lidar_camera":
        return _collect_multimodal_samples(data_dir, max_samples)

    if modality == "lidar":
        pcd_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if len(pcd_files) == 0:
            raise ValueError(f"No .bin files found in {data_dir}")
        return pcd_files[:max_samples] if max_samples > 0 else pcd_files

    raise ValueError("Modality must be 'lidar', 'camera', or 'lidar_camera'")


def run_cka(
    config: str,
    data_dir: str,
    checkpoints: List[str],
    modality: str,
    max_samples: int = 10,
    output_dir: str = ".",
    verbose: bool = False,
) -> List[Tuple[int, str, float]]:
    """
    Compute CKA scores and return sorted tuple rows.

    Returns
    -------
    List[Tuple[int, str, float]]
        One tuple per layer: (layer_index, layer_name, layer_cka_score)
    """
    if len(checkpoints) != 2:
        raise ValueError("Exactly two checkpoints must be provided via 'checkpoints'.")

    config = os.path.abspath(config)
    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isfile(config):
        raise FileNotFoundError(f"Config file not found: {config}")
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Data directory not found: {data_dir}")
    if max_samples < 0:
        raise ValueError(f"max_samples must be >= 0, got {max_samples}")

    normalized_checkpoints = [os.path.abspath(p) for p in checkpoints]
    missing = [p for p in normalized_checkpoints if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("Checkpoint file(s) not found: " + ", ".join(missing))

    sample_files = _collect_samples(data_dir=data_dir, modality=modality, max_samples=max_samples)
    os.makedirs(output_dir, exist_ok=True)

    try:
        from comparison_engine import ModelComparator
        from comparison_utils import default_hook_filter, sort_layers_by_network_order
        from metrics import CKAMetric
    except ImportError as exc:
        raise ImportError(
            "Failed to import local comparison modules. "
            "Run this with the same Python environment used for mmdetection3d. "
            f"Original error: {exc}"
        ) from exc

    start_time = time.time()
    print(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    comparator = ModelComparator(
        config_path=config,
        sample_files=sample_files,
        temp_dir=os.path.join(output_dir, "temp_cka"),
        verbose=verbose,
    )

    print(f"ModelComparator initialized in {time.time() - start_time:.2f} seconds.")
    cka_scores = comparator.compare_activation_metric(
        model_a_path=normalized_checkpoints[0],
        model_b_path=normalized_checkpoints[1],
        metric=CKAMetric(),
        hook_filter=default_hook_filter,
    )
    print(f"CKA comparison completed in {time.time() - start_time:.2f} seconds.")

    if len(cka_scores) == 0:
        raise RuntimeError("CKA comparison produced no layer scores.")

    ordered_layers = sort_layers_by_network_order(list(cka_scores.keys()))
    rows: List[Tuple[int, str, float]] = []
    for layer_idx, layer_name in enumerate(ordered_layers):
        rows.append((layer_idx, layer_name, float(cka_scores[layer_name])))

    return rows


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run only CKA comparison and output (index, layer_name, score) tuples"
    )
    parser.add_argument("--config", type=str, required=True, help="Model config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with input files")
    parser.add_argument("--pt", nargs="+", required=True, help="Two model checkpoints")
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=["lidar", "camera", "lidar_camera"],
        help="Data modality",
    )
    parser.add_argument("--labels", nargs="+", default=["ModelA", "ModelB"], help="Compatibility arg")
    parser.add_argument("--max_samples", type=int, default=10, help="Max samples to process")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for temp cache")
    parser.add_argument("--run_name", type=str, default=None, help="Compatibility arg")
    parser.add_argument("--verbose", default=False, action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of Python tuple-list literal",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if len(args.pt) != 2:
        raise ValueError("Comparison requires exactly 2 model checkpoints via --pt")

    try:
        rows = run_cka(
            config=args.config,
            data_dir=args.data_dir,
            checkpoints=args.pt,
            modality=args.modality,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"runner.py failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if args.json:
        # JSON does not support NaN cleanly; normalize to null for portability.
        json_rows = [
            [idx, name, (None if math.isnan(score) else score)]
            for idx, name, score in rows
        ]
        print(json.dumps(json_rows))
    else:
        print(rows)


if __name__ == "__main__":
    main()
