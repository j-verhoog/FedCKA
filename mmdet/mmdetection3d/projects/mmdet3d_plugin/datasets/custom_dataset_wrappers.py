# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmdet3d.datasets.builder import DATASETS, build_dataset


@DATASETS.register_module()
class CustomCBGSDataset(object):
    """A wrapper of class sampled dataset with ann_file path. Implementation of
    paper `Class-balanced Grouping and Sampling for Point Cloud 3D Object
    Detection <https://arxiv.org/abs/1908.09492.>`_.

    Balance the number of scenes under different classes.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be class sampled.
        enable_cap (bool): Whether to apply subset bounding caps. Default: False.
        max_proportion (float): Max percentage of dataset length a single frame can be duplicated.
        max_scene_frames (int): Max total frames a single scene can contribute to an epoch.
        Settable with:
        data = dict(
            train=dict(
            type='CustomCBGSDataset',
            enable_cap=True,               
            max_proportion=0.006,          
            max_scene_frames=120,          
            dataset=dict(...)
            )
        )
    """

    def __init__(self, dataset, enable_cap=False, max_proportion=0.006, max_scene_frames=120):
        # FIX: If the builder passed a config dict instead of an object, build it here!
        if isinstance(dataset, dict):
            dataset = build_dataset(dataset)

        self.dataset = dataset
        self.enable_cap = enable_cap
        self.max_proportion = max_proportion
        self.max_scene_frames = max_scene_frames
        
        self.CLASSES = dataset.CLASSES
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.sample_indices = self._get_sample_indices()
        # self.dataset.data_infos = self.data_infos
        if hasattr(self.dataset, 'flag'):
            self.flag = np.array(
                [self.dataset.flag[ind] for ind in self.sample_indices],
                dtype=np.uint8)

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx in range(len(self.dataset)):
            sample_cat_ids = self.dataset.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                class_sample_idxs[cat_id].append(idx)
        
        duplicated_samples = sum([len(v) for _, v in class_sample_idxs.items()])
        
        # FIX 1: Prevent ZeroDivisionError if the entire subset is empty
        class_distribution = {
            k: len(v) / duplicated_samples if duplicated_samples > 0 else 0.0
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []
        frac = 1.0 / len(self.CLASSES)
        
        # FIX 2: Prevent ZeroDivisionError for classes with 0 instances
        ratios = [frac / v if v > 0 else 0.0 for v in class_distribution.values()]
        
        # --- NEW: Setup sample-level dynamic cap ---
        dynamic_cap = float('inf')
        if self.enable_cap:
            dynamic_cap = max(1.0, len(self.dataset) * self.max_proportion)

        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            # FIX 3: Prevent np.random.choice from crashing on empty lists
            if len(cls_inds) > 0:
                
                # Apply sample cap if enabled
                bounded_ratio = min(ratio, dynamic_cap) if self.enable_cap else ratio
                
                sample_indices += np.random.choice(
                    cls_inds, 
                    int(len(cls_inds) * bounded_ratio)
                ).tolist()
                
        # --- NEW: Scene-level cap (Only drops duplicates) ---
        if self.enable_cap:
            import json
            import os.path as osp
            from collections import defaultdict
            
            # Read metadata safely to ensure correct scene mapping
            data_root = getattr(self.dataset, 'data_root', './data/nuscenes')
            version = getattr(self.dataset, 'version', 'v1.0-trainval')
            sample_json_path = osp.join(data_root, version, 'sample.json')
            
            sample2scene = {}
            if osp.exists(sample_json_path):
                with open(sample_json_path, 'r') as f:
                    samples_meta = json.load(f)
                    for s in samples_meta:
                        sample2scene[s['token']] = s['scene_token']

            scene_to_indices = defaultdict(list)
            for idx in sample_indices:
                info = self.dataset.data_infos[idx]
                sample_token = info.get('token', '')
                scene_token = sample2scene.get(sample_token, f'Unknown_{idx}')
                scene_to_indices[scene_token].append(idx)
                
            capped_sample_indices = []
            for scene_token, indices in scene_to_indices.items():
                if len(indices) <= self.max_scene_frames:
                    capped_sample_indices.extend(indices)
                else:
                    unique_indices = list(set(indices))
                    kept_indices = unique_indices.copy()
                    
                    # Calculate how many extra duplicates we are allowed to keep
                    allowed_duplicates = max(0, self.max_scene_frames - len(unique_indices))
                    
                    if allowed_duplicates > 0:
                        # Isolate only the duplicates
                        duplicates_only = indices.copy()
                        for u_idx in unique_indices:
                            duplicates_only.remove(u_idx)
                            
                        # Downsample ONLY the duplicates
                        kept_duplicates = np.random.choice(duplicates_only, allowed_duplicates, replace=False).tolist()
                        kept_indices.extend(kept_duplicates)
                        
                    capped_sample_indices.extend(kept_indices)
                    
            # Shuffle so chunks of the exact same scene aren't loaded sequentially
            np.random.shuffle(capped_sample_indices)
            sample_indices = capped_sample_indices

        return sample_indices

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        ori_idx = self.sample_indices[idx]
        return self.dataset[ori_idx]

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.sample_indices)