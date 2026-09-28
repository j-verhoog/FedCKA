"""
Main multi-metric comparison runner

Compares two architecturally identical models across all three metrics:
- CKA: Representational similarity (layer activations)
- W2: BatchNorm statistics divergence (running mean/variance)
- Effective Scale/Bias: BN affine parameter similarity (γ/β)

This generates comprehensive heatmaps for all metrics side-by-side.

Usage:
    python main.py --config <config> --pcd_dir <dir> --pt <model1> <model2> \
                   --labels <label1> <label2> --max_samples 100 --output_dir ./results
"""

import sys
import os
import glob
import argparse
import numpy as np
import re
from typing import List, Tuple

INTERNAL_PATH = '/opt/src/mmdetection3d'
if INTERNAL_PATH not in sys.path:
    sys.path.insert(0, INTERNAL_PATH)

from metrics import CKAMetric, W2BNStatsMetric, EffectiveScaleBiasMetric, BhattacharyyaMetric
from comparison_engine import ModelComparator
from comparison_utils import default_hook_filter
from visualization import ComparisonVisualizer, create_comparison_report

"""
# HOW TO RUN IT:
singularity shell --nv --cleanenv --bind /YOUR_HOMEDIR_PATH_HERE/mmdet:/workspace/mmdet --bind /YOUR_HOMEDIR_PATH_HERE/Desktop:/YOUR_HOMEDIR_PATH_HERE/Desktop /YOUR_HOMEDIR_PATH_HERE/mmdet/image/mmdet3d_v1rc5.sif
# Run in container:
export PATH=/opt/conda/envs/mmdet3d_v100rc5/bin:/opt/conda/bin:$PATH
export PYTHONPATH=/workspace/mmdet/mmdetection3d:$PYTHONPATH
export PYTHONPATH=/workspace/mmdet/mmdetection3d:/workspace/mmdet:\$PYTHONPATH
cd /workspace/mmdet/mmdetection3d/
python projects/analysis/scripts_cmt/main.py --config /YOUR_HOMEDIR_PATH_HERE/mmdet/mmdetection3d/projects/configs/cmt_voxel01_vov_1600x640_cbgs_syncbn.py --data_dir /YOUR_HOMEDIR_PATH_HERE/mmdet/datasets/v1.0-mini --pt /YOUR_HOMEDIR_PATH_HERE/Desktop/cmt_nocomm_modelA.pth /YOUR_HOMEDIR_PATH_HERE/Desktop/cmt_nocomm_modelB.pth --modality lidar_camera
"""

def collect_multimodal_samples(data_dir: str, max_samples: int) -> List[Tuple[str, str]]:
    """
    Collect paired lidar+camera files for multimodal processing.
    
    Args:
        data_dir: Path to nuscenes data directory containing samples/
        max_samples: Maximum number of sample pairs to collect (0 = all)
        
    Returns:
        List of (lidar_file, camera_file) tuples
    """
    lidar_dir = os.path.join(data_dir, "samples", "LIDAR_TOP")
    camera_dir = os.path.join(data_dir, "samples", "CAM_FRONT")
    
    if not os.path.exists(lidar_dir) or not os.path.exists(camera_dir):
        print(f"ERROR: Expected NuScenes structure with samples/LIDAR_TOP and samples/CAM_FRONT")
        return []
    
    # Get all lidar files
    lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.bin")))
    camera_files = sorted(glob.glob(os.path.join(camera_dir, "*.jpg")))
    
    if len(lidar_files) == 0 or len(camera_files) == 0:
        print(f"ERROR: No lidar (.bin) or camera (.jpg) files found")
        return []
    
    # Extract timestamp-based mapping
    def extract_scene_timestamp(filepath):
        """Extract scene_id and timestamp from nuscenes filename"""
        basename = os.path.basename(filepath)
        # Pattern: scene__SENSOR__timestamp.ext
        parts = basename.split('__')
        if len(parts) >= 3:
            scene_id = parts[0]
            timestamp_part = parts[2]
            # Handle double extensions like .pcd.bin by removing all extensions
            while '.' in timestamp_part:
                timestamp_part = os.path.splitext(timestamp_part)[0]
            return scene_id, timestamp_part
        return None, None
    
    # Build lookup table for camera files
    camera_lookup = {}
    for cam_file in camera_files:
        scene_id, timestamp = extract_scene_timestamp(cam_file)
        if scene_id and timestamp:
            key = (scene_id, timestamp)
            camera_lookup[key] = cam_file
    
    # Match lidar files with closest camera files
    paired_samples = []
    for lidar_file in lidar_files:
        scene_id, lidar_timestamp = extract_scene_timestamp(lidar_file)
        if not scene_id or not lidar_timestamp:
            continue
            
        # Find the closest camera timestamp for the same scene
        best_camera_file = None
        min_time_diff = float('inf')
        
        lidar_ts = int(lidar_timestamp)
        for (cam_scene_id, cam_timestamp), cam_file in camera_lookup.items():
            if cam_scene_id == scene_id:  # Same scene
                cam_ts = int(cam_timestamp)
                time_diff = abs(lidar_ts - cam_ts)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_camera_file = cam_file
        
        if best_camera_file is not None:
            paired_samples.append((lidar_file, best_camera_file))
    
    # Apply max_samples limit
    if max_samples > 0:
        paired_samples = paired_samples[:max_samples]
    
    print(f"Found {len(paired_samples)} lidar+camera pairs")
    return paired_samples


def main():
    parser = argparse.ArgumentParser(
        description="Multi-metric comparison of two architecturally identical models"
    )
    parser.add_argument('--config', type=str, required=True, help="Model config file")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory with .bin/.jpg files")
    parser.add_argument('--pt', nargs='+', required=True, help="Two model checkpoints")
    parser.add_argument('--modality', type=str, required=True, help="Data modality (lidar, camera, lidar_camera)")
    parser.add_argument('--labels', nargs='+', default=['ModelA', 'ModelB'], help="Two model labels")
    parser.add_argument('--max_samples', type=int, default=10, help="Max samples to process")
    parser.add_argument('--output_dir', type=str, default=".", help="Output directory")
    parser.add_argument('--run_name', type=str, default=None, help="Custom name for this comparison run")
    parser.add_argument('--verbose', default=False, action='store_true', help="Enable verbose logging")

    args = parser.parse_args()
    
    if len(args.pt) != 2 or len(args.labels) != 2:
        print("ERROR: Comparison requires exactly 2 models and 2 labels")
        return
    
    if args.modality not in ['lidar', 'camera', 'lidar_camera']:
        print("ERROR: Modality must be 'lidar', 'camera', or 'lidar_camera'")
        return
    
    if args.modality == 'camera':
        print(f"Running comparison for modality: {args.modality}")
        img_files = sorted(glob.glob(os.path.join(args.data_dir, "*.jpg")))
        if len(img_files) == 0:
            print(f"ERROR: No .jpg files found in {args.data_dir}")
            return
        if args.max_samples > 0:
            sample_files = img_files[:args.max_samples]
        else:
            sample_files = img_files
    elif args.modality == 'lidar_camera':
        print(f"Running comparison for modality: {args.modality}")
        # For multimodal, pair lidar and camera files by timestamp
        sample_files = collect_multimodal_samples(args.data_dir, args.max_samples)
        if len(sample_files) == 0:
            print(f"ERROR: No paired lidar+camera files found in {args.data_dir}")
            return
    else:  # lidar
        print(f"Running comparison for modality: {args.modality}")
        # Gather PCD files
        pcd_files = sorted(glob.glob(os.path.join(args.data_dir, "*.bin")))
        if len(pcd_files) == 0:
            print(f"ERROR: No .bin files found in {args.data_dir}")
            return
        if args.max_samples > 0:
            sample_files = pcd_files[:args.max_samples]
        else:
            sample_files = pcd_files
    
    print("=" * 80)
    print("MULTI-METRIC MODEL COMPARISON")
    print("=" * 80)
    print(f"Samples: {len(sample_files)}")
    print(f"Model A: {args.labels[0]}")
    print(f"Model B: {args.labels[1]}")
    print(f"Metrics: CKA, W2 BN Stats, Effective Scale/Bias")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run all three metrics
    comparator = ModelComparator(args.config, sample_files, os.path.join(args.output_dir, "temp"), verbose=args.verbose)
    
    metrics = [
        CKAMetric(),
        W2BNStatsMetric(),
        EffectiveScaleBiasMetric(),
        BhattacharyyaMetric()
    ]
    
    print("\n[Comparisons]")
    results = comparator.run_full_comparison(
        args.pt[0], args.pt[1],
        metrics,
        hook_filter=default_hook_filter
    )
    
    # Prepare visualization data
    if args.run_name:
        comp_title = args.run_name
    else:
        comp_title = f"{args.labels[0]}_vs_{args.labels[1]}"
    comparisons = [(comp_title, args.labels[0], args.labels[1])]
    
    viz_results = {comp_title: results}
    metric_names = ['CKA', 'W2 BN Stats', 'Effective Scale/Bias', 'Bhattacharyya Coeff']
    
    # Generate visualizations
    print("\n[Visualization]")
    visualizer = ComparisonVisualizer()
    output_files = visualizer.plot_all_metrics(
        viz_results, comparisons, metric_names,
        output_dir=args.output_dir
    )
    
    # 1. Generate the Consolidated Plot (The Main Goal)
    # Collect all layers for ordering
    all_layers = set()
    for m_scores in results.values():
        all_layers.update(m_scores.keys())
    from comparison_utils import sort_layers_by_network_order
    layer_order = sort_layers_by_network_order(list(all_layers))

    print(f'All layers to plot: {len(layer_order)}')
    print(f'Layer names: {layer_order}')
    

    consolidated_path = os.path.join(args.output_dir, "consolidated_comparison.png")
    visualizer.plot_consolidated(
        viz_results, 
        comparisons, 
        layer_order, 
        consolidated_path,
        run_name=args.run_name if args.run_name else f"{args.labels[0]} vs {args.labels[1]}"
    )
    print(f"  Saved Consolidated Plot: {consolidated_path}")

    # 2. Generate Row-Normalized Consolidated Plot (each comparison row per metric spans 0..1)
    consolidated_norm_path = os.path.join(args.output_dir, "consolidated_comparison_row_normalized.png")
    visualizer.plot_consolidated_row_normalized(
        viz_results,
        comparisons,
        layer_order,
        consolidated_norm_path,
        run_name=args.run_name if args.run_name else f"{args.labels[0]} vs {args.labels[1]}"
    )
    print(f"  Saved Row-Normalized Consolidated Plot: {consolidated_norm_path}")


    # Generate report
    report_path = create_comparison_report(
        viz_results, comparisons, metric_names,
        len(sample_files), args.output_dir
    )
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    for metric_name in metric_names:
        if metric_name in results:
            scores = results[metric_name]
            # Filter out NaN values for statistics
            valid_scores = [v for v in scores.values() if not np.isnan(v)]
            
            print(f"\n{metric_name}:")
            print(f"  Layers compared: {len(scores)}")
            print(f"  Valid scores:    {len(valid_scores)}")
            
            if len(valid_scores) > 0:
                print(f"  Mean score:      {np.mean(valid_scores):.4f}")
                print(f"  Std dev:         {np.std(valid_scores):.4f}")
                print(f"  Min score:       {np.min(valid_scores):.4f}")
                print(f"  Max score:       {np.max(valid_scores):.4f}")
            else:
                print(f"  No valid scores available (all NaN)")
                print(f"  This may indicate activation collection issues for this metric")
    
    print("\n" + "=" * 80)
    print(f"Report saved: {report_path}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()