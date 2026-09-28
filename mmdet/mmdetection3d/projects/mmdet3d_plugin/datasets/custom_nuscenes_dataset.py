# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
# Modified for FedCKA by Jolle Verhoog, 2026.

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmcv.utils import print_log # <--- ADD THIS IMPORT

@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, *args, return_gt_info=False, **kwargs):
        super(CustomNuScenesDataset, self).__init__(*args, **kwargs)
        self.return_gt_info = return_gt_info

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            img_sweeps=None if 'img_sweeps' not in info else info['img_sweeps'],
            radar_info=None if 'radars' not in info else info['radars']
        )

        if self.return_gt_info:
            input_dict['info'] = info

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict
    
    def evaluate(self, 
                 results, 
                 metric='bbox', 
                 logger=None, 
                 jsonfile_prefix=None, 
                 result_names=['pts_bbox'], 
                 show=False, 
                 out_dir=None, 
                 pipeline=None, 
                 min_gt_count=1, 
                 **kwargs): 
        """Evaluate and inject filtered mAP and NDS."""
        
        # 1. Run standard evaluation
        metric_dict = super().evaluate( 
            results=results, 
            metric=metric, 
            logger=logger, 
            jsonfile_prefix=jsonfile_prefix, 
            result_names=result_names, 
            show=show, 
            out_dir=out_dir, 
            pipeline=pipeline, 
            **kwargs) 

        # 2. Count Ground Truth instances per class
        class_counts = {cls: 0 for cls in self.CLASSES} 
        for i in range(len(self)): 
            ann_info = self.get_ann_info(i) 
            if 'gt_names' in ann_info: 
                for name in ann_info['gt_names']: 
                    if name in class_counts: 
                        class_counts[name] += 1 

        # 3. Identify valid classes
        valid_classes = [cls for cls, count in class_counts.items() if count >= min_gt_count] 

        if not valid_classes: 
            return metric_dict 

        # 4. Recalculate Filtered metrics
        for res_name in result_names: 
            prefix = f'{res_name}_NuScenes' 
            
            # --- FIXED: Average AP over the 4 distance thresholds for each class ---
            filtered_aps = []
            dist_ths = ['0.5', '1.0', '2.0', '4.0']
            for cls in valid_classes:
                cls_aps = [metric_dict[f'{prefix}/{cls}_AP_dist_{dist}'] 
                           for dist in dist_ths 
                           if f'{prefix}/{cls}_AP_dist_{dist}' in metric_dict]
                if cls_aps:
                    filtered_aps.append(np.mean(cls_aps))
            
            filtered_map = np.mean(filtered_aps) if filtered_aps else 0.0 
            metric_dict[f'{prefix}/filtered_mAP'] = filtered_map 
            
            # Recalculate NDS components
            tp_metric_names = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err'] 
            tp_sums = 0.0 
            
            for tp_name in tp_metric_names: 
                filtered_tps = [metric_dict[f'{prefix}/{cls}_{tp_name}'] 
                                for cls in valid_classes  
                                if f'{prefix}/{cls}_{tp_name}' in metric_dict and not np.isnan(metric_dict[f'{prefix}/{cls}_{tp_name}'])] 
                
                mean_tp = np.mean(filtered_tps) if filtered_tps else 1.0 
                tp_sums += (1.0 - min(1.0, mean_tp)) 

            filtered_nds = (5.0 * filtered_map + tp_sums) / 10.0 
            metric_dict[f'{prefix}/filtered_NDS'] = filtered_nds 
            
            # --- FIXED: Safe logging using MMCV's print_log ---
            dropped = set(self.CLASSES) - set(valid_classes) 
            log_msg = (f"\n[Filtered Evaluation] Valid GT Threshold >= {min_gt_count}\n"
                       f"Dropped classes: {list(dropped)}\n"
                       f"Filtered mAP: {filtered_map:.4f}, Filtered NDS: {filtered_nds:.4f}\n")
            print_log(log_msg, logger=logger)

        return metric_dict