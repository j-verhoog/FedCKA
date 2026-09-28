"""
Common utilities for model comparison:
- Hooks for extracting layer outputs
- Data loading and inference
- Visualization helpers
"""

import os
import re
import sys
import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, List, Callable
from torch.nn.modules.batchnorm import _BatchNorm
from tqdm import tqdm

# Import from parent directory
try:
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmdet3d.apis import init_model 
    from mmdet3d.datasets.pipelines import Compose
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
except ImportError as e:
    print(f"FATAL IMPORT ERROR: {e}")
    sys.exit(1)


# class ForwardHook:
#     """
#     Generic hook for capturing layer outputs during forward pass.
#     Automatically extracts tensor from complex structures.
#     """
    
#     def __init__(self):
#         self.output = None
#         self.module_ref = None
#         self.call_count = 0
    
#     def __call__(self, module, input, output):
#         """Called during forward pass."""
#         self.module_ref = module
#         self.call_count += 1
        
#         if output is None:
#             self.output = None
#             return
        
#         # Unwrap structures more aggressively
#         if isinstance(output, (list, tuple)):
#             # Try to find the first tensor in the structure
#             for item in output:
#                 if torch.is_tensor(item):
#                     output = item
#                     break
#             else:
#                 # No tensor found, keep original
#                 output = output[0] if len(output) > 0 else None
        
#         if output is None:
#             self.output = None
#             return
        
#         # Handle custom tensor wrappers
#         if hasattr(output, "features"):
#             output = output.features
#         elif hasattr(output, "data"):
#             output = output.data
        
#         # Store as tensor
#         if torch.is_tensor(output):
#             try:
#                 # Keep on GPU if possible to avoid memory transfer overhead
#                 self.output = output.detach().float()
#                 if self.call_count == 1:  # Debug first call
#                     print(f"    Hook captured: {output.shape} from {module.__class__.__name__}")
#             except Exception as e:
#                 print(f"    Hook failed to capture from {module.__class__.__name__}: {e}")
#                 self.output = None
#         else:
#             self.output = None
    
#     def get_output(self) -> torch.Tensor:
#         """Get captured output (or None)."""
# #         return self.output
# class ForwardHook:
#     """
#     Lightweight hook that captures sparse features and indices 
#     without triggering OOMs or CUDA asserts.
#     """
#     def __init__(self):
#         self.output = None
    
#     def __call__(self, module, input, output):
#         if output is None:
#             return
            
#         # Unwrap structures
#         if isinstance(output, (list, tuple)):
#             for item in output:
#                 if torch.is_tensor(item) or hasattr(item, "indices"):
#                     output = item
#                     break
#             else:
#                 output = output[0] if len(output) > 0 else None
                
#         if output is None:
#             return

#         # 1. Catch Sparse Tensors and package them lightly
#         if hasattr(output, "indices") and hasattr(output, "features") and hasattr(output, "spatial_shape"):
#             batch_size = getattr(output, "batch_size", 1)
#             self.output = {
#                 "is_sparse": True,
#                 "features": output.features.detach().cpu(), # Keep on CPU to save GPU VRAM for caching
#                 "indices": output.indices.detach().cpu().long(),
#                 "shape": output.spatial_shape,
#                 "batch_size": batch_size
#             }
            
#         # 2. Catch Standard Tensors
#         elif hasattr(output, "features"):
#             self.output = output.features.detach().cpu()
#         elif hasattr(output, "data"):
#             self.output = output.data.detach().cpu()
#         elif torch.is_tensor(output):
#             self.output = output.detach().cpu()
            
#     def get_output(self):
#         return self.output
class ForwardHook:
    """
    Stateful hook that captures sparse features and indices,
    and caches them to re-attach to subsequent BatchNorm layers
    that temporarily strip spatial context.
    """
    
    # Class-level cache to share state across all hook instances
    # during a sequential forward pass.
    _last_sparse_info = None
    
    def __init__(self):
        self.output = None
        self.call_count = 0
    
    def __call__(self, module, input, output):
        self.call_count += 1
        
        if output is None:
            return
            
        # Unwrap structures
        if isinstance(output, (list, tuple)):
            for item in output:
                if torch.is_tensor(item) or hasattr(item, "indices"):
                    output = item
                    break
            else:
                output = output[0] if len(output) > 0 else None
                
        if output is None:
            return

        # 1. Catch Sparse Tensors and CACHE their spatial context
        if hasattr(output, "indices") and hasattr(output, "features") and hasattr(output, "spatial_shape"):
            batch_size = getattr(output, "batch_size", 1)
            features = output.features.detach().cpu()
            indices = output.indices.detach().cpu().long()
            
            # Update global cache for subsequent BN layers
            ForwardHook._last_sparse_info = {
                "num_features": features.shape[0],
                "indices": indices,
                "shape": output.spatial_shape,
                "batch_size": batch_size
            }
            
            self.output = {
                "is_sparse": True,
                "features": features,
                "indices": indices,
                "shape": output.spatial_shape,
                "batch_size": batch_size
            }
            
        # 2. Catch Standard Tensors (e.g., from BatchNorm inside sparse blocks)
        elif hasattr(output, "features"):
            tensor_out = output.features.detach().cpu()
            self._handle_dense_or_cached(tensor_out)
        elif hasattr(output, "data"):
            tensor_out = output.data.detach().cpu()
            self._handle_dense_or_cached(tensor_out)
        elif torch.is_tensor(output):
            tensor_out = output.detach().cpu()
            self._handle_dense_or_cached(tensor_out)
            
    def _handle_dense_or_cached(self, tensor_out: torch.Tensor):
        """
        Check if this dense tensor is actually a stripped sparse feature.
        If N matches the cached active voxel count, re-attach indices.
        """
        cache = ForwardHook._last_sparse_info
        
        # If we have a cache, and the tensor is 2D [N, C], and N matches exactly
        if (cache is not None and 
            tensor_out.dim() == 2 and 
            tensor_out.shape[0] == cache["num_features"]):
            
            # RE-ATTACH THE SPATIAL CONTEXT
            self.output = {
                "is_sparse": True,
                "features": tensor_out,
                "indices": cache["indices"],
                "shape": cache["shape"],
                "batch_size": cache["batch_size"]
            }
        else:
            # Truly dense layer, leave as standard tensor
            self.output = tensor_out

    def get_output(self):
        return self.output
def load_model_and_pipeline(config_path: str, checkpoint_path: str):
    """
    Load a model and its data pipeline.
    
    Sets model to eval mode and disables BatchNorm running stats updates
    to ensure no model modification during comparison.
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to checkpoint
        
    Returns:
        Tuple of (model, pipeline)
    """
    print(f"Loading weights: {os.path.basename(checkpoint_path)} ...")
    
    cfg = Config.fromfile(config_path)

    # Handle custom imports first
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    import importlib
    # import modules from plugin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(f"Loading plugin module: {_module_path}")
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(config_path)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(f"Loading plugin module: {_module_path}")
                plg_lib = importlib.import_module(_module_path)
                
    plg_lib_base = importlib.import_module('mmdetection3d.mmdet3d')
    
    model = init_model(config_path, checkpoint_path, device='cuda:0')
    model.eval()
    
    test_pipeline = Compose(cfg.data.test.pipeline)
    return model, test_pipeline


def register_hooks(
    model: nn.Module,
    filter_func: Callable[[str, nn.Module], bool]
) -> Tuple[Dict[str, ForwardHook], List[torch.utils.hooks.RemovableHandle]]:
    """
    Register hooks on model layers matching filter criteria.
    
    Args:
        model: PyTorch model
        filter_func: Function(layer_name, module) -> bool to select layers
        
    Returns:
        Tuple of (hooks_dict, handles_list)
    """
    hooks = {}
    handles = []
    
    print("  Registering hooks on layers...")
    hooked_count = 0
    
    for name, module in model.named_modules():
        if filter_func(name, module):
            h = ForwardHook()
            handles.append(module.register_forward_hook(h))
            hooks[name] = h
            hooked_count += 1
            
            # Debug: print first few and last few layers being hooked
            if hooked_count <= 5 or hooked_count % 50 == 0:
                print(f"    Hooked {hooked_count}: {name} ({module.__class__.__name__})")
    
    print(f"  Total hooks registered: {hooked_count}")
    return hooks, handles

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Any, List, Callable
from torch.nn.modules.batchnorm import _BatchNorm
from tqdm import tqdm

# --- NEW IMPORTS FOR NUSCENES ---
try:
    from nuscenes.nuscenes import NuScenes
except ImportError:
    print("Warning: 'nuscenes-devkit' not found. Camera comparison will require dummy intrinsics.")

# --- LAZY LOADING HELPERS TO AVOID RELOADING DB ---
_NUSC_INSTANCE = None
_NUSC_FILE_INDEX = None
_NUSC_CACHE = {}
_NUSC_VERSION_PREFERENCE = ("v1.0-trainval", "v1.0-mini")


def _candidate_nuscenes_configs(file_path: str) -> List[Tuple[str, str]]:
    """Find plausible (dataroot, version) pairs by walking ancestors."""
    candidates: List[Tuple[str, str]] = []
    seen = set()

    current_dir = os.path.dirname(os.path.abspath(file_path))
    for _ in range(12):
        for version_name in _NUSC_VERSION_PREFERENCE:
            json_path = os.path.join(current_dir, version_name, "sample_data.json")
            if os.path.isfile(json_path):
                key = (current_dir, version_name)
                if key not in seen:
                    candidates.append(key)
                    seen.add(key)

        base = os.path.basename(current_dir)
        if base in ("v1.0-trainval", "v1.0-mini"):
            parent_dir = os.path.dirname(current_dir)
            json_path = os.path.join(current_dir, base, "sample_data.json")
            if os.path.isfile(json_path):
                key = (current_dir, base)
                if key not in seen:
                    candidates.append(key)
                    seen.add(key)
            elif os.path.isfile(os.path.join(parent_dir, base, "sample_data.json")):
                key = (parent_dir, base)
                if key not in seen:
                    candidates.append(key)
                    seen.add(key)

        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            break
        current_dir = parent

    return candidates


def _get_or_build_nuscenes_index(dataroot: str, version: str):
    """Load/cache NuScenes instance and filename index for one dataset version."""
    key = (dataroot, version)
    if key in _NUSC_CACHE:
        return _NUSC_CACHE[key]

    print(f"Initializing NuScenes DB from: {dataroot} ({version}) ...")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    file_index = {}
    for record in nusc.sample_data:
        fname = os.path.basename(record['filename'])
        file_index[fname] = record

    _NUSC_CACHE[key] = (nusc, file_index)
    return _NUSC_CACHE[key]

def _get_nuscenes_metadata(file_path: str):
    """
    Load NuScenes metadata and fetch intrinsics for a file.
    Supports v1.0-mini and v1.0-trainval by discovering available layouts.
    """
    global _NUSC_INSTANCE, _NUSC_FILE_INDEX

    filename_only = os.path.basename(file_path)

    # Discover possible roots/versions and choose one containing this file.
    # Preference is explicit: v1.0-trainval, then v1.0-mini.
    candidates = _candidate_nuscenes_configs(file_path)
    if not candidates:
        # Fallback to active index if it already has this file.
        if _NUSC_FILE_INDEX and filename_only in _NUSC_FILE_INDEX:
            sd_record = _NUSC_FILE_INDEX[filename_only]
            cs_record = _NUSC_INSTANCE.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            return np.array(cs_record['camera_intrinsic'], dtype=np.float32)

        print(f"Warning: Could not locate a NuScenes root/version for {file_path}. Using dummy intrinsics.")
        return None

    pref_rank = {ver: i for i, ver in enumerate(_NUSC_VERSION_PREFERENCE)}
    candidates = sorted(candidates, key=lambda item: pref_rank.get(item[1], 999))

    for dataroot, version in candidates:
        try:
            nusc, file_index = _get_or_build_nuscenes_index(dataroot, version)
        except Exception as exc:
            print(f"Warning: Failed to initialize NuScenes ({version}) at {dataroot}: {exc}")
            continue

        if filename_only in file_index:
            _NUSC_INSTANCE = nusc
            _NUSC_FILE_INDEX = file_index
            sd_record = file_index[filename_only]
            cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            return np.array(cs_record['camera_intrinsic'], dtype=np.float32)

    # Secondary fallback to current active index.
    if _NUSC_FILE_INDEX and filename_only in _NUSC_FILE_INDEX:
        sd_record = _NUSC_FILE_INDEX[filename_only]
        cs_record = _NUSC_INSTANCE.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        return np.array(cs_record['camera_intrinsic'], dtype=np.float32)

    print(
        f"Warning: Could not map file to NuScenes metadata across candidates: "
        f"{filename_only}. Using dummy intrinsics."
    )
    
    return None


def get_nuscenes_calibration_info(lidar_path: str, camera_path: str):
    """
    Get NuScenes calibration info for lidar-camera pair, computing necessary transformation matrices.
    This replicates the coordinate transformation logic from inference_multi_modality_detector.
    """
    global _NUSC_INSTANCE, _NUSC_FILE_INDEX
    
    try:
        # Check imports
        try:
            from pyquaternion import Quaternion
        except ImportError:
            print("Warning: pyquaternion not available. Install with: pip install pyquaternion")
            return None
            
        # Initialize NuScenes if needed
        _get_nuscenes_metadata(camera_path)
        
        if _NUSC_INSTANCE is None:
            print("Warning: NuScenes DB not available")
            return None
            
        # Get camera and lidar sample data records
        camera_filename = os.path.basename(camera_path)
        lidar_filename = os.path.basename(lidar_path)
        
        camera_sd = _NUSC_FILE_INDEX.get(camera_filename)
        lidar_sd = _NUSC_FILE_INDEX.get(lidar_filename)
        
        if not camera_sd or not lidar_sd:
            print(f"Warning: Could not find sample_data for {camera_filename} or {lidar_filename}")
            return None
            
        # Get calibrated sensor records
        camera_cs = _NUSC_INSTANCE.get('calibrated_sensor', camera_sd['calibrated_sensor_token'])
        lidar_cs = _NUSC_INSTANCE.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        
        # Get ego pose records
        camera_ep = _NUSC_INSTANCE.get('ego_pose', camera_sd['ego_pose_token'])
        lidar_ep = _NUSC_INSTANCE.get('ego_pose', lidar_sd['ego_pose_token'])
        
        # Convert to numpy arrays
        cam_intrinsic = np.array(camera_cs['camera_intrinsic'], dtype=np.float32)
        cam_translation = np.array(camera_cs['translation'], dtype=np.float32) 
        cam_rotation = np.array(camera_cs['rotation'], dtype=np.float32)
        
        lidar_translation = np.array(lidar_cs['translation'], dtype=np.float32)
        lidar_rotation = np.array(lidar_cs['rotation'], dtype=np.float32)
        
        # Build transformation matrices (following CustomNuScenesDataset conventions)
        from pyquaternion import Quaternion
        
        # Lidar to ego transformation
        lidar_quat = Quaternion(lidar_rotation)
        lidar2ego_rotation = lidar_quat.rotation_matrix
        lidar2ego_translation = lidar_translation.reshape(3, 1)
        lidar2ego = np.hstack([lidar2ego_rotation, lidar2ego_translation])
        lidar2ego = np.vstack([lidar2ego, [0, 0, 0, 1]])
        
        # Ego to camera transformation  
        cam_quat = Quaternion(cam_rotation)
        ego2cam_rotation = cam_quat.rotation_matrix.T  # Inverse rotation
        ego2cam_translation = -ego2cam_rotation @ cam_translation.reshape(3, 1)
        ego2cam = np.hstack([ego2cam_rotation, ego2cam_translation])
        ego2cam = np.vstack([ego2cam, [0, 0, 0, 1]])
        
        # Full lidar to camera transformation (following CMT format)
        lidar2cam_rt = ego2cam @ lidar2ego
        
        # CMT expects 4x4 viewpad matrix with intrinsics embedded
        viewpad = np.eye(4)
        viewpad[:cam_intrinsic.shape[0], :cam_intrinsic.shape[1]] = cam_intrinsic
        
        # Lidar to image transformation (CMT format: viewpad @ lidar2cam_rt.T)
        lidar2img_rt = viewpad @ lidar2cam_rt.T
        
        calibration_info = {
            'lidar2img': lidar2img_rt.astype(np.float32),
            'camera_intrinsic': cam_intrinsic,
            'lidar2cam': lidar2cam_rt.T.astype(np.float32)  # CMT expects transposed
        }
        
        return calibration_info
        
    except Exception as e:
        print(f"Warning: Failed to get NuScenes calibration info: {e}")
        return None


def get_camera_intrinsics(camera_path: str):
    """
    Legacy function - kept for compatibility.
    """
    return _get_nuscenes_metadata(camera_path)

# --- MAIN FUNCTION ---

def process_single_sample(
    model: nn.Module,
    pipeline: Any,
    sample_path,  # Can be str (single file) or tuple (lidar_file, camera_file)
    hooks: Dict[str, Any],
    verbose=False
) -> Dict[str, torch.Tensor]:
    """
    Run inference on a single sample using exact same logic as inference_multi_modality_detector.
    """
    from mmdet3d.apis import inference_detector
    from mmdet3d.core import get_box_type, Box3DMode
    from mmcv.parallel import collate, scatter
    from copy import deepcopy
    from mmdet3d.datasets.pipelines import Compose
    import mmcv
    import os
    import numpy as np
    import re
    
    # Handle multimodal input
    is_multimodal = isinstance(sample_path, (tuple, list)) and len(sample_path) == 2
    
    try:
        if is_multimodal:
            lidar_path, camera_path = sample_path
            print(f"Processing multimodal sample: {os.path.basename(lidar_path)} + {os.path.basename(camera_path)}") if verbose else None
            
            # EXACT REPLICATION OF inference_multi_modality_detector LOGIC
            cfg = model.cfg
            device = next(model.parameters()).device  # model device
            
            # Build the data pipeline  
            test_pipeline = deepcopy(cfg.data.test.pipeline)
            test_pipeline = Compose(test_pipeline)
            box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
            
            # Get NuScenes calibration data (this is the missing piece!)
            calibration_info = get_nuscenes_calibration_info(lidar_path, camera_path)
            
            # Create data dict for CMT multimodal model (following CustomNuScenesDataset format)
            data = dict(
                pts_filename=lidar_path,
                img_filename=[camera_path],  # List for multi-view
                img_prefix=None,  # Set to None when using full paths
                img_info=dict(filename=[os.path.basename(camera_path)]),  # List format
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d,
                ann_info=dict(axis_align_matrix=np.eye(4)),
                sweeps=[],
                timestamp=[0],
                img_timestamp=[0],  # CMT requires this
                img_fields=[],
                bbox3d_fields=[],
                pts_mask_fields=[],
                pts_seg_fields=[],
                bbox_fields=[],
                mask_fields=[],
                seg_fields=[])
            
            # Add CMT-required calibration matrices
            if calibration_info is not None:
                # Format as lists for multi-view (even for single view)
                cam_intrinsic_4x4 = np.eye(4)
                cam_intrinsic_4x4[:3, :3] = calibration_info['camera_intrinsic']
                
                data.update(dict(
                    cam_intrinsic=[cam_intrinsic_4x4],
                    lidar2img=[calibration_info['lidar2img']],
                    lidar2cam=[calibration_info['lidar2cam']]
                ))
                print(f"    Added calibration matrices (shapes: intrinsic={cam_intrinsic_4x4.shape}, lidar2img={calibration_info['lidar2img'].shape}, lidar2cam={calibration_info['lidar2cam'].shape})") if verbose else None
            else:
                # Fallback dummy values if calibration fails
                dummy_intrinsic = np.eye(4)
                dummy_transform = np.eye(4)  # CMT expects 4x4 matrices
                data.update(dict(
                    cam_intrinsic=[dummy_intrinsic],
                    lidar2img=[dummy_transform],
                    lidar2cam=[dummy_transform]
                ))
                print("    Using dummy calibration matrices (calibration failed)")
            
            # Process through pipeline
            data = test_pipeline(data)
            
            # No need for additional coordinate transforms - CMT handles this through data structure
            
            # Collate and scatter (same as original)
            data = collate([data], samples_per_gpu=1)
            if next(model.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [device.index])[0]
            else:
                # this is a workaround to avoid the bug of MMDataParallel
                data['img_metas'] = data['img_metas'][0].data
                data['points'] = data['points'][0].data
                data['img'] = data['img'][0].data
            
            # Forward the model (exactly as in original)
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
                
        else:
            # For unimodal, use standard inference
            result = inference_detector(model, sample_path)
        
        # Collect the hooked outputs
        return {name: h.get_output() for name, h in hooks.items() if h.get_output() is not None}

    except Exception as e:
        print(f"Error processing sample: {e}")
        import traceback
        traceback.print_exc()
        return {}

def get_bn_module_refs(
    model: nn.Module
) -> Dict[str, _BatchNorm]:
    """
    Get reference to all BatchNorm modules in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict mapping module names to _BatchNorm modules
    """
    bn_modules = {}
    
    for name, module in model.named_modules():
        if isinstance(module, _BatchNorm):
            bn_modules[name] = module
    
    return bn_modules


def default_hook_filter(name: str, module: nn.Module) -> bool:
    """
    Enhanced filter for selecting layers to hook.
    Includes convolutions, linear layers, normalization layers, and transformer components.
    
    Args:
        name: Layer name
        module: Module instance
        
    Returns:
        bool: Whether to hook this layer
    """
    cls = module.__class__.__name__.lower()
    
    # Skip activation functions and utility layers
    skip_types = (
        nn.Dropout, nn.Dropout2d, nn.Dropout3d,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.ELU, nn.Tanh,
        nn.Sigmoid, nn.Softmax, nn.LogSoftmax,
        nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
        nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
        nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
        nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
        nn.Upsample
    )
    
    if isinstance(module, skip_types):
        return False
    
    # Include main computational layer types
    include_types = (
        # Convolutional layers
        nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        # Linear layers  
        nn.Linear,
        # Normalization layers
        _BatchNorm, nn.SyncBatchNorm,
        nn.GroupNorm, nn.LayerNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        # Attention layers
        nn.MultiheadAttention,
    )
    
    if isinstance(module, include_types):
        return True
    
    # Check by name patterns for additional layer types
    include_patterns = [
        # Normalization patterns
        "batchnorm", "syncbn", "norm", "layernorm", "groupnorm", "instancenorm",
        # Transformer patterns  
        "attention", "attn", "multihead", "self_attn", "cross_attn",
        # Custom layer patterns
        "embedding", "projection", "ffn", "mlp",
    ]
    
    if any(pattern in cls or pattern in name.lower() for pattern in include_patterns):
        return True
        
    # Include layers with learnable parameters (weights/bias)
    has_params = any(p.requires_grad for p in module.parameters()) if hasattr(module, 'parameters') else False
    if has_params and len(list(module.children())) == 0:  # Leaf modules only
        # Additional check to avoid hooking containers
        if not isinstance(module, (nn.ModuleList, nn.ModuleDict, nn.Sequential)):
            return True
    
    return False


import re
from typing import List, Tuple

# def sort_layers_by_network_order(layers: List[str]) -> List[str]:
#     """
#     Sort layers by typical network order for 3D Detection (FCOS3D/PointPillars).
#     Ensures numerical layers (10 vs 2) and architectural stages follow data flow.
#     """
#     # 1. Strict chronological prefix order
#     prefix_order = [
#         "pts_voxel_layer", 
#         "pts_voxel_encoder", 
#         "pts_middle_encoder",
#         "pts_backbone", 
#         "backbone", 
#         "pts_neck", 
#         "neck", 
#         "pts_bbox_head", 
#         "bbox_head", 
#         "head"
#     ]
    
#     # 2. Sub-layer priority to mimic data flow within the Neck
#     # Lateral connections happen before FPN/Upsample/Extra convs
#     sub_priority = {
#         "lateral_convs": 0,
#         "fpn_convs": 1,
#         "upsample_layers": 2,
#         "extra_convs": 3
#     }

#     def natural_keys(text: str):
#         """Helper to sort 'layer10' after 'layer2'."""
#         return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

#     def sort_key(layer_name: str) -> Tuple:
#         # Determine Main Stage
#         prefix_idx = 999
#         for i, prefix in enumerate(prefix_order):
#             if layer_name.startswith(prefix):
#                 prefix_idx = i
#                 break
        
#         # Determine Sub-Stage priority (mostly for the Neck)
#         internal_priority = 99
#         for sub_prefix, priority in sub_priority.items():
#             if sub_prefix in layer_name:
#                 internal_priority = priority
#                 break
                
#         # Return hierarchy: (Stage, Sub-Stage, Natural Alphanumeric Order)
#         return (prefix_idx, internal_priority, natural_keys(layer_name))

#     return sorted(layers, key=sort_key)

import re
from typing import List, Tuple

def sort_layers_by_network_order(layers: List[str]) -> List[str]:
    """
    Sort layers following the Cross-modal-transformer flow:
    Image Backbone -> LiDAR Backbone -> Multi-modal Transformer -> Regression Heads.
    """
    # 1. Main Architectural Stages in order of data flow
    prefix_order = [
        # Image Branch
        "img_backbone",
        "img_neck",
        # LiDAR Branch
        "pts_voxel_layer", 
        "pts_voxel_encoder", 
        "pts_middle_encoder",
        "pts_backbone", 
        "pts_neck", 
        # Multimodal / Head
        "pts_bbox_head.shared_conv",
        "pts_bbox_head.transformer",
        "pts_bbox_head.bev_embedding",
        "pts_bbox_head.rv_embedding",
        "pts_bbox_head.reference_points",
        "pts_bbox_head.task_heads",  # Final regression
        "pts_bbox_head",
        "bbox_head",
        "head"
    ]
    
    # 2. Sub-layer priority for internal logic (Neck and Heads)
    sub_priority = {
        # Neck internal flow
        "lateral_convs": 0,
        "fpn_convs": 1,
        "deblocks": 2, # Common in LiDAR necks
        "upsample_layers": 3,
        "extra_convs": 4,
        # Head internal flow (Regression types)
        "cls_logits": 10,
        "center": 11,
        "height": 12,
        "dim": 13,
        "rot": 14,
        "vel": 15
    }

    def natural_keys(text: str):
        """Alphanumeric sort helper (layer2 < layer10)."""
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

    def sort_key(layer_name: str) -> Tuple:
        # Determine Main Stage
        prefix_idx = 999
        for i, prefix in enumerate(prefix_order):
            if layer_name.startswith(prefix):
                prefix_idx = i
                break
        
        # Determine Sub-Stage priority
        internal_priority = 99
        for sub_prefix, priority in sub_priority.items():
            if sub_prefix in layer_name:
                internal_priority = priority
                break
                
        # Return hierarchy: (Stage Index, Sub-Stage Priority, Alphanumeric Name)
        return (prefix_idx, internal_priority, natural_keys(layer_name))

    return sorted(layers, key=sort_key)