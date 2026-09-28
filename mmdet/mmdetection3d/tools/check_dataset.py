# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import argparse
import os
import json
import warnings
from os import path as osp
from collections import Counter

import mmcv
from mmcv import Config, DictAction

from mmdet3d.datasets import build_dataset
from mmdet3d.utils import setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze dataset and CBGS sampling')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    import importlib
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                plg_lib = importlib.import_module(_module_path)
            else:
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                plg_lib = importlib.import_module(_module_path)
                
    plg_lib_base = importlib.import_module('mmdetection3d.mmdet3d')

    print(f"Building dataset from config: {args.config}...")
    dataset = build_dataset(cfg.data.train)

    print("\n" + "="*85)
    print("DATASET ANALYSIS")
    print("="*85)

    if hasattr(dataset, 'dataset') and hasattr(dataset, 'sample_indices'):
        original_dataset = dataset.dataset
        cbgs_indices = dataset.sample_indices
        classes = dataset.CLASSES

        len_without_cbgs = len(original_dataset)
        len_with_cbgs = len(dataset)
        added_samples = len_with_cbgs - len_without_cbgs

        print(f"Length WITHOUT CBGS: {len_without_cbgs} samples")
        print(f"Length WITH CBGS:    {len_with_cbgs} samples")
        print(f"Samples ADDED:       {added_samples} samples")
        print("-" * 85)

        # ---------------------------------------------------------
        # 1. CLASS DISTRIBUTION ANALYSIS
        # ---------------------------------------------------------
        print("Gathering class distribution data...")
        orig_cat_ids = {i: original_dataset.get_cat_ids(i) for i in range(len_without_cbgs)}
        
        counts_before = {i: 0 for i in range(len(classes))}
        counts_after = {i: 0 for i in range(len(classes))}

        for i, cat_ids in orig_cat_ids.items():
            for cat_id in cat_ids:
                counts_before[cat_id] += 1

        for orig_idx in cbgs_indices:
            cat_ids = orig_cat_ids[orig_idx]
            for cat_id in cat_ids:
                counts_after[cat_id] += 1

        print("\nCLASS DISTRIBUTION SUMMARY:")
        print(f"{'Class Name':<20} | {'Count BEFORE CBGS':<20} | {'Count AFTER CBGS':<20}")
        print("-" * 85)
        for i, cls_name in enumerate(classes):
            print(f"{cls_name:<20} | {counts_before[i]:<20} | {counts_after[i]:<20}")
        print("-" * 85)

        # ---------------------------------------------------------
        # 2. OVERSAMPLING FREQUENCY ANALYSIS (SAMPLE-LEVEL)
        # ---------------------------------------------------------
        index_counts = Counter(cbgs_indices)
        top_k = 15
        most_common = index_counts.most_common(top_k)
        
        print(f"\nTop {top_k} Most Oversampled INDIVIDUAL SAMPLES:")
        print(f"{'Original Index':<15} | {'Times Included':<15} | {'Contained Classes'}")
        print("-" * 85)
        for orig_idx, count in most_common:
            contained_classes = [classes[cat_id] for cat_id in orig_cat_ids[orig_idx]]
            cls_str = ", ".join(contained_classes)
            print(f"{orig_idx:<15} | {count:<15} | {cls_str}")
        print("-" * 85)

        # ---------------------------------------------------------
        # 3. EXACT SCENE DISTRIBUTION ANALYSIS (METADATA-BASED)
        # ---------------------------------------------------------
        if hasattr(original_dataset, 'data_infos'):
            print("\nReading exact scene metadata from raw nuScenes JSONs...")
            
            # Safely grab the data_root and version from the dataset config
            data_root = getattr(original_dataset, 'data_root', './data/nuscenes')
            version = getattr(original_dataset, 'version', 'v1.0-trainval')
            
            sample_json_path = osp.join(data_root, version, 'sample.json')
            scene_json_path = osp.join(data_root, version, 'scene.json')

            sample2scene = {}
            scene2name = {}

            # Read raw JSON metadata to guarantee accuracy
            if osp.exists(sample_json_path) and osp.exists(scene_json_path):
                with open(sample_json_path, 'r') as f:
                    samples_meta = json.load(f)
                    for s in samples_meta:
                        sample2scene[s['token']] = s['scene_token']
                
                with open(scene_json_path, 'r') as f:
                    scenes_meta = json.load(f)
                    for s in scenes_meta:
                        scene2name[s['token']] = s['name']
            else:
                print(f"WARNING: Could not find JSONs at {sample_json_path}. Make sure paths are correct.")

            orig_scene_names = {}
            for i in range(len_without_cbgs):
                info = original_dataset.data_infos[i]
                sample_token = info.get('token', '')
                
                # Map Sample Token -> Scene Token -> Scene Name
                scene_token = sample2scene.get(sample_token, 'Unknown_Hash')
                scene_name = scene2name.get(scene_token, scene_token)
                orig_scene_names[i] = scene_name

            scene_counts_before = Counter(orig_scene_names.values())
            scene_counts_after = Counter([orig_scene_names[idx] for idx in cbgs_indices])

            print("\nSCENE OVERSAMPLING SUMMARY (ALL Scenes):")
            print(f"{'Scene Name':<20} | {'Frames BEFORE':<15} | {'Frames AFTER':<15} | {'Multiplier':<10}")
            print("-" * 85)
            
            sorted_scenes = sorted(scene_counts_after.items(), key=lambda x: x[1], reverse=True)
            for scene_id, after_count in sorted_scenes:
                before_count = scene_counts_before[scene_id]
                multiplier = after_count / before_count if before_count > 0 else 0
                print(f"{str(scene_id):<20} | {before_count:<15} | {after_count:<15} | {multiplier:.2f}x")
            
            print("-" * 85)
            print(f"Total Unique Scenes Counted: {len(scene_counts_before)}")
        else:
            print("\nDataset does not have 'data_infos' attribute, skipping scene analysis.")

    else:
        print("The dataset does not appear to be wrapped in a CBGSDataset.")
        print("Make sure your config uses type='CustomCBGSDataset' in data.train.")
        print(f"Current Dataset length: {len(dataset)}")

    print("="*85 + "\n")


if __name__ == '__main__':
    main()