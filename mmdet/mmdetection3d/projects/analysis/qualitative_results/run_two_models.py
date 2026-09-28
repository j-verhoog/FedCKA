# Quick guide to reconstruct qualitative results from paper:
# 1. Run this script with a config and two checkpoint paths:
#    python run_two_models.py CONFIG.py CHECKPOINT_A.pth CHECKPOINT_B.pth
# 2. Set --target-img to the image filename to visualize and use --camera to
#    select the camera (for example, CAM_FRONT).
# 3. Use --out /YOUR_PATH_HERE/comparison_output.jpg to choose the output
#    location. The script writes separate GT, model A, model B, and BEV images.
# 4. Use --score-thr, --draw-space, --projection-mode, and --debug to adjust
#    filtering, image coordinates, projection diagnostics, and debug output.
# 5. Replace /YOUR_PATH_HERE/ and provide paths appropriate to your machine.

import argparse
import importlib
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

import mmcv
from mmcv import Config
from mmcv.parallel import collate, MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from nuscenes.utils.geometry_utils import view_points


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize and compare 3D predictions on 2D images")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("checkpoint_a", help="first checkpoint file (.pth)")
    parser.add_argument("checkpoint_b", help="second checkpoint file (.pth)")

    parser.add_argument("--target-img", default="1542799615612460.jpg", help="Image filename to search for")
    parser.add_argument(
        "--out",
        default="/YOUR_PATH_HERE/comparison_output.jpg",
        help="Output image filename",
    )
    parser.add_argument("--score-thr", type=float, default=0.35, help="Bounding box score threshold")
    parser.add_argument("--camera", default="CAM_FRONT", help="Camera name to visualize")
    parser.add_argument("--line-thickness", type=int, default=2)

    parser.add_argument(
        "--draw-space",
        choices=["processed", "padded", "raw"],
        default="padded",
        help=(
            "processed: resize raw image to img_shape. "
            "padded: resize to img_shape and place on pad_shape canvas. "
            "raw: draw on raw image; only correct if projection matrix is also raw-image based."
        ),
    )

    parser.add_argument(
        "--projection-mode",
        choices=["auto", "lidar2img", "cam2img", "lidar2img_with_img_aug", "img_aug_with_lidar2img"],
        default="auto",
        help=(
            "auto: use lidar2img for LiDAR boxes and cam2img for camera boxes. "
            "Other modes are for debugging projection/augmentation mismatch."
        ),
    )

    parser.add_argument(
        "--z-shift",
        type=float,
        default=0.0,
        help="Optional debug shift added to box z coordinate before projecting. Use e.g. -0.5 or 0.5 to test z-origin issues.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extensive metadata, matrices, and box diagnostics.",
    )

    parser.add_argument(
        "--max-debug-boxes",
        type=int,
        default=5,
        help="Number of box tensors to print for debugging.",
    )

    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.5,
        help="Font scale for class name labels on bounding boxes.",
    )

    parser.add_argument(
        "--lidar-range",
        type=float,
        default=50.0,
        help="Maximum range in meters for BEV X and Y axes (e.g., 50 will show -50 to +50 meters).",
    )

    parser.add_argument(
        "--bev-gt-label",
        type=str,
        default="ENTER_TEXT_HERE",
        help="Label for GT subplot in BEV visualization.",
    )

    parser.add_argument(
        "--bev-model-a-label",
        type=str,
        default="ENTER_TEXT_HERE",
        help="Label for Model A subplot in BEV visualization.",
    )

    parser.add_argument(
        "--bev-model-b-label",
        type=str,
        default="ENTER_TEXT_HERE",
        help="Label for Model B subplot in BEV visualization.",
    )

    parser.add_argument(
        "--draw-class-names",
        action="store_true",
        default=False,
        help="Draw class names on bounding boxes in camera images.",
    )

    return parser.parse_args()


def import_plugins(cfg, config_path):
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    if hasattr(cfg, "plugin") and cfg.plugin:
        if hasattr(cfg, "plugin_dir"):
            plugin_dir = cfg.plugin_dir
            module_dir = os.path.dirname(plugin_dir).split("/")
        else:
            module_dir = os.path.dirname(config_path).split("/")

        module_path = module_dir[0]
        for m in module_dir[1:]:
            module_path += "." + m

        importlib.import_module(module_path)

    try:
        importlib.import_module("mmdetection3d.mmdet3d")
    except Exception as e:
        print(f"[WARN] Could not import mmdetection3d.mmdet3d: {e}")


def prepare_cfg(cfg):
    cfg.model.pretrained = None

    if hasattr(cfg, "data") and hasattr(cfg.data, "test"):
        cfg.data.test.test_mode = True

    if hasattr(cfg.model, "train_cfg"):
        cfg.model.train_cfg = None

    return cfg


def find_sample_idx(dataset, target_img, camera):
    for i, info in enumerate(dataset.data_infos):
        cams = info.get("cams", {})
        if camera in cams and target_img in cams[camera].get("data_path", ""):
            return i

    for i, info in enumerate(dataset.data_infos):
        cams = info.get("cams", {})
        for cam_name, cam_info in cams.items():
            if target_img in cam_info.get("data_path", ""):
                print(f"[WARN] Target image found in {cam_name}, not requested camera {camera}. Using this sample anyway.")
                return i

    return -1


def get_data_container_payload(x):
    if hasattr(x, "data"):
        return x.data
    return x


def unwrap_singletons(x):
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return x


def get_img_meta(data):
    metas = data["img_metas"]
    metas = get_data_container_payload(metas[0])
    metas = unwrap_singletons(metas)

    if isinstance(metas, (list, tuple)):
        return metas[0]

    return metas


def ensure_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def find_camera_index(img_meta, target_img, camera):
    filenames = ensure_list(img_meta.get("filename", []))

    for i, fn in enumerate(filenames):
        if target_img in str(fn):
            return i

    for i, fn in enumerate(filenames):
        if camera in str(fn):
            return i

    print("[WARN] Could not identify camera index from filename. Falling back to index 0.")
    return 0


def get_meta_value_for_camera(img_meta, key, cam_idx, required=False):
    if key not in img_meta:
        if required:
            raise KeyError(f"Required key '{key}' not found in img_meta.")
        return None

    value = img_meta[key]

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            if required:
                raise ValueError(f"img_meta['{key}'] is empty.")
            return None

        if cam_idx >= len(value):
            print(
                f"[WARN] cam_idx={cam_idx} out of range for img_meta['{key}'] "
                f"with len={len(value)}. Falling back to index 0."
            )
            return value[0]

        return value[cam_idx]

    return value


def as_np_matrix(x, name):
    if x is None:
        return None

    x = np.asarray(x, dtype=np.float64)

    if x.shape == (3, 3):
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = x
        return out

    if x.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = x
        return out

    if x.shape == (4, 4):
        return x

    raise ValueError(f"{name} should be 3x3, 3x4, or 4x4, but got shape {x.shape}.")


def print_matrix_like_meta(img_meta, cam_idx):
    print("\n[DEBUG] ===== MATRIX-LIKE IMG_META KEYS =====")

    keys = list(img_meta.keys())
    for k in keys:
        lk = k.lower()
        if (
            "img" in lk
            or "lidar" in lk
            or "cam" in lk
            or "matrix" in lk
            or "rot" in lk
            or "trans" in lk
            or "scale" in lk
            or "crop" in lk
            or "pad" in lk
            or "shape" in lk
        ):
            v = img_meta[k]
            print(f"\n[DEBUG] key = {k}")
            print(f"[DEBUG] type = {type(v)}")

            try:
                vv = v[cam_idx] if isinstance(v, (list, tuple)) and len(v) > cam_idx else v
                arr = np.asarray(vv)
                print(f"[DEBUG] selected value type = {type(vv)}")
                print(f"[DEBUG] selected value shape = {arr.shape}")
                print(f"[DEBUG] selected value =\n{arr}")
            except Exception as e:
                print(f"[DEBUG] could not print value for key {k}: {e}")

    print("[DEBUG] ===== END MATRIX-LIKE IMG_META KEYS =====\n")


def print_img_meta_summary(img_meta, cam_idx):
    print("\n[INFO] ===== IMG_META SUMMARY =====")
    print(f"[INFO] img_meta keys: {list(img_meta.keys())}")

    for key in [
        "filename",
        "ori_filename",
        "ori_shape",
        "img_shape",
        "pad_shape",
        "scale_factor",
        "crop_offset",
        "img_norm_cfg",
    ]:
        value = get_meta_value_for_camera(img_meta, key, cam_idx, required=False)
        print(f"[INFO] {key}: {value}")

    print("[INFO] ===== END IMG_META SUMMARY =====\n")

def load_image_for_camera(img_meta, cam_idx, draw_space):
    filenames = ensure_list(img_meta.get("filename", []))

    if len(filenames) == 0:
        raise ValueError("No filenames found in img_meta.")

    if cam_idx >= len(filenames):
        raise IndexError(f"cam_idx={cam_idx}, but only {len(filenames)} filenames available.")

    img_path = filenames[cam_idx]
    raw_img = cv2.imread(img_path)

    if raw_img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    raw_h, raw_w = raw_img.shape[:2]

    img_shape = get_meta_value_for_camera(img_meta, "img_shape", cam_idx, required=True)
    img_h, img_w = int(img_shape[0]), int(img_shape[1])

    pad_shape = get_meta_value_for_camera(img_meta, "pad_shape", cam_idx, required=False)

    print(f"[INFO] Camera index: {cam_idx}")
    print(f"[INFO] Image path: {img_path}")
    print(f"[INFO] Raw image shape: {raw_img.shape}")
    print(f"[INFO] img_shape: {img_shape}")
    print(f"[INFO] pad_shape: {pad_shape}")

    if draw_space == "raw":
        print("[WARN] Drawing on raw image. This is only correct if projection matrix is also in raw image coordinates.")
        return raw_img, img_path

    scale_w = img_w / raw_w
    scale_h_direct = img_h / raw_h

    resized_h = int(round(raw_h * scale_w))
    resized_w = img_w

    print(f"[INFO] Width-based resize scale: {scale_w:.6f}")
    print(f"[INFO] Direct height scale would be: {scale_h_direct:.6f}")
    print(f"[INFO] Aspect-preserving resized shape: ({resized_h}, {resized_w})")

    resized = cv2.resize(raw_img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    if resized_h > img_h:
        crop_top = resized_h - img_h
        crop_bottom = crop_top + img_h

        print(f"[INFO] Cropping resized image vertically:")
        print(f"[INFO]   resized height = {resized_h}")
        print(f"[INFO]   target height  = {img_h}")
        print(f"[INFO]   crop_top      = {crop_top}")
        print(f"[INFO]   crop_bottom   = {crop_bottom}")

        img = resized[crop_top:crop_bottom, :, :]

    elif resized_h < img_h:
        print("[WARN] Aspect-preserving resize is smaller than img_shape height. Padding bottom.")
        img = np.zeros((img_h, img_w, 3), dtype=resized.dtype)
        img[:resized_h, :resized_w] = resized

    else:
        print("[INFO] Aspect-preserving resized image already matches img_shape height.")
        img = resized

    if img.shape[:2] != (img_h, img_w):
        raise RuntimeError(f"Image shape after resize/crop is {img.shape}, expected {(img_h, img_w, 3)}")

    if draw_space == "processed":
        print("[INFO] Drawing on processed aspect-preserving resized/cropped image.")
        return img, img_path

    if draw_space == "padded":
        if pad_shape is None:
            print("[WARN] pad_shape missing. Falling back to processed img_shape.")
            return img, img_path

        pad_h, pad_w = int(pad_shape[0]), int(pad_shape[1])

        if (pad_h, pad_w) == (img_h, img_w):
            print("[INFO] pad_shape equals img_shape. Drawing on processed image.")
            return img, img_path

        print("[INFO] Placing processed image on padded canvas.")
        canvas = np.zeros((pad_h, pad_w, 3), dtype=img.dtype)
        copy_h = min(img_h, pad_h)
        copy_w = min(img_w, pad_w)
        canvas[:copy_h, :copy_w] = img[:copy_h, :copy_w]
        return canvas, img_path

    raise ValueError(f"Unknown draw_space: {draw_space}")


def extract_boxes_scores(result_dict):
    if "pts_bbox" in result_dict:
        result_dict = result_dict["pts_bbox"]

    boxes = result_dict.get("boxes_3d", None)
    scores = result_dict.get("scores_3d", None)

    if boxes is None:
        raise KeyError("Could not find boxes_3d in model output.")

    return boxes, scores


def filter_boxes_by_score(boxes_3d, scores_3d, score_thr):
    if boxes_3d is None:
        return None, None

    if scores_3d is None:
        return boxes_3d, None

    if torch.is_tensor(scores_3d):
        mask = scores_3d.detach().cpu() > score_thr
    else:
        mask = torch.tensor(np.asarray(scores_3d) > score_thr)

    if len(mask) != len(boxes_3d):
        raise ValueError(f"Score length {len(mask)} does not match box length {len(boxes_3d)}.")

    return boxes_3d[mask], mask


def clone_and_shift_boxes_z(boxes_3d, z_shift):
    if z_shift == 0.0 or boxes_3d is None:
        return boxes_3d

    boxes_shifted = boxes_3d.clone()
    boxes_shifted.tensor[:, 2] += z_shift
    return boxes_shifted


def debug_print_boxes(name, boxes_3d, scores_3d, max_boxes):
    print(f"\n[DEBUG] ===== BOX DEBUG: {name} =====")

    if boxes_3d is None:
        print("[DEBUG] boxes_3d is None")
        print(f"[DEBUG] ===== END BOX DEBUG: {name} =====\n")
        return

    print(f"[DEBUG] type: {type(boxes_3d)}")
    print(f"[DEBUG] len: {len(boxes_3d)}")

    if hasattr(boxes_3d, "box_dim"):
        print(f"[DEBUG] box_dim: {boxes_3d.box_dim}")

    if hasattr(boxes_3d, "origin"):
        print(f"[DEBUG] origin: {boxes_3d.origin}")

    if hasattr(boxes_3d, "with_yaw"):
        print(f"[DEBUG] with_yaw: {boxes_3d.with_yaw}")

    try:
        tensor = boxes_3d.tensor.detach().cpu()
        print(f"[DEBUG] tensor shape: {tuple(tensor.shape)}")
        print(f"[DEBUG] first {max_boxes} boxes tensor:")
        print(tensor[:max_boxes])
        print(f"[DEBUG] z min/mean/max: {tensor[:, 2].min().item():.4f} / {tensor[:, 2].mean().item():.4f} / {tensor[:, 2].max().item():.4f}")
        print(f"[DEBUG] h/dim-z min/mean/max: {tensor[:, 5].min().item():.4f} / {tensor[:, 5].mean().item():.4f} / {tensor[:, 5].max().item():.4f}")
    except Exception as e:
        print(f"[DEBUG] could not print tensor: {e}")

    if scores_3d is not None:
        try:
            scores = scores_3d.detach().cpu() if torch.is_tensor(scores_3d) else torch.tensor(scores_3d)
            print(f"[DEBUG] scores shape: {tuple(scores.shape)}")
            print(f"[DEBUG] first {max_boxes} scores:")
            print(scores[:max_boxes])
            print(f"[DEBUG] score min/mean/max: {scores.min().item():.4f} / {scores.mean().item():.4f} / {scores.max().item():.4f}")
        except Exception as e:
            print(f"[DEBUG] could not print scores: {e}")

    try:
        corners = boxes_3d.corners.detach().cpu()
        print(f"[DEBUG] corners shape: {tuple(corners.shape)}")
        print(f"[DEBUG] first box corners:")
        print(corners[0])
    except Exception as e:
        print(f"[DEBUG] could not print corners: {e}")

    print(f"[DEBUG] ===== END BOX DEBUG: {name} =====\n")


def get_img_aug_matrix(img_meta, cam_idx):
    candidate_keys = [
        "img_aug_matrix",
        "img_aug_mat",
        "img_transform",
        "img_transformation",
    ]

    for key in candidate_keys:
        value = get_meta_value_for_camera(img_meta, key, cam_idx, required=False)
        if value is not None:
            try:
                return as_np_matrix(value, key), key
            except Exception as e:
                print(f"[WARN] Found {key}, but could not convert to matrix: {e}")

    return None, None


def get_projection_matrix(img_meta, cam_idx, boxes_3d, projection_mode):
    box_type = type(boxes_3d).__name__

    lidar2img_raw = get_meta_value_for_camera(img_meta, "lidar2img", cam_idx, required=False)
    cam2img_raw = get_meta_value_for_camera(img_meta, "cam2img", cam_idx, required=False)

    lidar2img = as_np_matrix(lidar2img_raw, "lidar2img") if lidar2img_raw is not None else None
    cam2img = as_np_matrix(cam2img_raw, "cam2img") if cam2img_raw is not None else None

    img_aug, img_aug_key = get_img_aug_matrix(img_meta, cam_idx)

    if projection_mode == "lidar2img":
        if lidar2img is None:
            raise KeyError("projection-mode=lidar2img requested, but lidar2img is missing.")
        return lidar2img, "lidar2img"

    if projection_mode == "cam2img":
        if cam2img is None:
            raise KeyError("projection-mode=cam2img requested, but cam2img is missing.")
        return cam2img, "cam2img"

    if projection_mode == "lidar2img_with_img_aug":
        if lidar2img is None:
            raise KeyError("projection-mode=lidar2img_with_img_aug requested, but lidar2img is missing.")
        if img_aug is None:
            raise KeyError("projection-mode=lidar2img_with_img_aug requested, but no img_aug_matrix-like key was found.")
        return lidar2img @ img_aug, f"lidar2img @ {img_aug_key}"

    if projection_mode == "img_aug_with_lidar2img":
        if lidar2img is None:
            raise KeyError("projection-mode=img_aug_with_lidar2img requested, but lidar2img is missing.")
        if img_aug is None:
            raise KeyError("projection-mode=img_aug_with_lidar2img requested, but no img_aug_matrix-like key was found.")
        return img_aug @ lidar2img, f"{img_aug_key} @ lidar2img"

    if projection_mode != "auto":
        raise ValueError(f"Unknown projection_mode: {projection_mode}")

    if "Camera" in box_type:
        if cam2img is not None:
            return cam2img, "auto: cam2img"
        if lidar2img is not None:
            print("[WARN] Camera boxes detected, but cam2img missing. Falling back to lidar2img.")
            return lidar2img, "auto fallback: lidar2img"
        raise KeyError("Neither cam2img nor lidar2img available for Camera boxes.")

    if lidar2img is not None:
        return lidar2img, "auto: lidar2img"

    if cam2img is not None:
        print("[WARN] LiDAR boxes detected, but lidar2img missing. Falling back to cam2img.")
        return cam2img, "auto fallback: cam2img"

    raise KeyError("Neither lidar2img nor cam2img available.")


def clip_line_to_image(pt1, pt2, width, height):
    x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
    x2, y2 = int(round(pt2[0])), int(round(pt2[1]))

    ok, clipped_pt1, clipped_pt2 = cv2.clipLine((0, 0, width, height), (x1, y1), (x2, y2))

    if not ok:
        return None, None

    return clipped_pt1, clipped_pt2


def project_corners_to_image(corners_3d, proj_mat):
    points = corners_3d.T
    projected = view_points(points, proj_mat, normalize=False)

    depths = projected[2, :]

    if np.all(depths <= 1e-5):
        return None, None, depths

    valid_depth = depths > 1e-5
    safe_depths = np.clip(depths, 1e-5, 1e8)

    corners_2d = projected[:2, :] / safe_depths[None, :]
    corners_2d = corners_2d.T

    return corners_2d, valid_depth, depths


def draw_box_edges(image, corners_2d, valid_depth, color, thickness, class_name=None, font_scale=0.5, draw_class_names=False):
    height, width = image.shape[:2]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    drawn_any = False

    for a, b in edges:
        if not (valid_depth[a] and valid_depth[b]):
            continue

        pt1, pt2 = clip_line_to_image(corners_2d[a], corners_2d[b], width, height)

        if pt1 is None:
            continue

        cv2.line(image, pt1, pt2, color, thickness)
        drawn_any = True

    if drawn_any and class_name is not None and draw_class_names:
        min_corner = np.min(corners_2d[valid_depth], axis=0)
        text_pos = (int(round(min_corner[0])), int(round(min_corner[1])) - 5)
        cv2.putText(image, class_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

    return drawn_any


def debug_project_first_boxes(name, corners_3d, proj_mat, image_shape, max_boxes):
    h, w = image_shape[:2]

    print(f"\n[DEBUG] ===== PROJECTION DEBUG: {name} =====")
    print(f"[DEBUG] image size used for drawing: width={w}, height={h}")

    for i in range(min(max_boxes, corners_3d.shape[0])):
        corners_2d, valid_depth, depths = project_corners_to_image(corners_3d[i], proj_mat)

        if corners_2d is None:
            print(f"[DEBUG] box {i}: all corners behind camera. depths={depths}")
            continue

        finite = np.all(np.isfinite(corners_2d))
        min_xy = np.nanmin(corners_2d, axis=0)
        max_xy = np.nanmax(corners_2d, axis=0)

        print(f"[DEBUG] box {i}:")
        print(f"[DEBUG]   finite={finite}")
        print(f"[DEBUG]   valid_depth={valid_depth}")
        print(f"[DEBUG]   depth min/max={np.min(depths):.4f}/{np.max(depths):.4f}")
        print(f"[DEBUG]   2d min xy={min_xy}, max xy={max_xy}")
        print(f"[DEBUG]   image bounds x=[0,{w}], y=[0,{h}]")

    print(f"[DEBUG] ===== END PROJECTION DEBUG: {name} =====\n")


def project_and_draw(
    boxes_3d,
    scores_3d,
    image,
    color,
    img_meta,
    cam_idx,
    score_thr,
    thickness,
    name,
    projection_mode,
    z_shift,
    debug,
    max_debug_boxes,
    class_names=None,
    font_scale=0.5,
    draw_class_names=False,
):
    if boxes_3d is None or len(boxes_3d) == 0:
        print(f"[INFO] {name}: no boxes.")
        return 0

    boxes_3d, _ = filter_boxes_by_score(boxes_3d, scores_3d, score_thr)

    if boxes_3d is None or len(boxes_3d) == 0:
        print(f"[INFO] {name}: no boxes after threshold {score_thr}.")
        return 0

    boxes_3d = clone_and_shift_boxes_z(boxes_3d, z_shift)

    if z_shift != 0.0:
        print(f"[WARN] {name}: debug z-shift applied: {z_shift}")

    if debug:
        debug_print_boxes(name, boxes_3d, scores_3d, max_debug_boxes)

    proj_mat, proj_name = get_projection_matrix(img_meta, cam_idx, boxes_3d, projection_mode)

    corners_3d = boxes_3d.corners
    if torch.is_tensor(corners_3d):
        corners_3d = corners_3d.detach().cpu().numpy()
    else:
        corners_3d = np.asarray(corners_3d)

    print(f"[INFO] {name}: drawing {corners_3d.shape[0]} boxes.")
    print(f"[INFO] {name}: box type = {type(boxes_3d).__name__}")
    print(f"[INFO] {name}: projection = {proj_name}")
    print(f"[INFO] {name}: projection matrix shape = {proj_mat.shape}")

    if debug:
        print(f"[DEBUG] {name}: projection matrix =\n{proj_mat}")
        debug_project_first_boxes(name, corners_3d, proj_mat, image.shape, max_debug_boxes)

    num_drawn = 0

    for i in range(corners_3d.shape[0]):
        corners_2d, valid_depth, _ = project_corners_to_image(corners_3d[i], proj_mat)

        if corners_2d is None:
            continue

        if not np.all(np.isfinite(corners_2d)):
            continue

        class_name = None
        if class_names is not None and i < len(class_names):
            class_name = class_names[i]

        if draw_box_edges(image, corners_2d, valid_depth, color, thickness, class_name=class_name, font_scale=font_scale, draw_class_names=draw_class_names):
            num_drawn += 1

    print(f"[INFO] {name}: actually drawn {num_drawn} boxes.")
    return num_drawn


def run_inference(model, data):
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    if isinstance(result, (list, tuple)):
        return result[0]

    return result


def filter_gt_boxes_by_visibility(gt_boxes, ann_info, good_visibility_only=True):
    """Filter GT boxes to only include those with GOOD visibility."""
    if gt_boxes is None:
        return None, None

    visibility = ann_info.get("gt_bboxes_3d_visibility", None)
    
    if visibility is None:
        print("[WARN] No visibility information found in annotations. Using all GT boxes.")
        return gt_boxes, None

    if not good_visibility_only:
        return gt_boxes, None

    visibility = np.asarray(visibility)
    good_visibility_mask = visibility == 4  # 4 represents GOOD visibility in nuscenes
    
    if not np.any(good_visibility_mask):
        print("[WARN] No boxes with GOOD visibility found. Using all boxes.")
        return gt_boxes, None
    
    filtered_boxes = gt_boxes[good_visibility_mask]
    print(f"[INFO] Filtered GT boxes: {len(gt_boxes)} -> {len(filtered_boxes)} (GOOD visibility only)")
    
    return filtered_boxes, good_visibility_mask


def get_class_names_for_boxes(boxes_3d, ann_info, dataset):
    """Get class names for each box from annotation info."""
    if boxes_3d is None or len(boxes_3d) == 0:
        return None
    
    class_names = ann_info.get("gt_names", None)
    
    if class_names is None:
        return None
    
    return list(class_names[:len(boxes_3d)])


def load_lidar_points_from_nusc(img_meta, nusc=None):
    """Load LiDAR points using nuscenes devkit (optional).
    
    This method is skipped if nuscenes data is not available.
    """
    return None


def load_lidar_points(data):
    """Extract LiDAR points from MMDetection3D data dictionary.
    
    This handles the complex nested structure of points in the data dict.
    """
    points = data.get("points", None)
    
    if points is None:
        print("[DEBUG] No 'points' key in data")
        return None
    
    try:
        print(f"[DEBUG] Initial points type: {type(points)}")
        
        # Recursively unwrap single-element lists/tuples
        depth = 0
        while isinstance(points, (list, tuple)) and len(points) == 1 and depth < 10:
            points = points[0]
            depth += 1
            print(f"[DEBUG] Unwrapped single-element list/tuple (depth {depth}): {type(points)}")
        
        # Now handle DataContainer if present
        if hasattr(points, "data"):
            print(f"[DEBUG] Unwrapping DataContainer...")
            points = points.data
            print(f"[DEBUG] After DataContainer.data: {type(points)}")
            
            # Recursively unwrap single-element lists/tuples again
            depth = 0
            while isinstance(points, (list, tuple)) and len(points) == 1 and depth < 10:
                points = points[0]
                depth += 1
                print(f"[DEBUG] Unwrapped single-element list/tuple after DataContainer (depth {depth}): {type(points)}")
        
        # Handle list of multiple tensors (stacked sweeps)
        if isinstance(points, (list, tuple)) and len(points) > 1:
            if all(isinstance(p, torch.Tensor) for p in points):
                print(f"[DEBUG] Detected {len(points)} tensors, concatenating...")
                points = torch.cat(points, dim=0)
                print(f"[DEBUG] Concatenated tensor shape: {points.shape}")
        
        print(f"[DEBUG] Before conversion to numpy: type={type(points)}")
        
        # Convert tensor to numpy
        if torch.is_tensor(points):
            print(f"[DEBUG] Converting tensor to numpy, shape: {points.shape}")
            points = points.cpu().numpy()
        elif not isinstance(points, np.ndarray):
            print(f"[DEBUG] Converting to numpy array from {type(points)}")
            points = np.asarray(points)
        
        print(f"[DEBUG] After conversion: shape={points.shape}, dtype={points.dtype}")
        
        # Validate and extract x, y
        if points.ndim < 2:
            print(f"[DEBUG] Array has {points.ndim} dimensions, need at least 2")
            return None
        
        if points.shape[0] == 0:
            print(f"[DEBUG] Array has 0 points")
            return None
        
        if points.shape[1] < 2:
            print(f"[DEBUG] Array has only {points.shape[1]} columns, need at least 2 (x, y)")
            return None
        
        # Extract x, y coordinates
        xy_points = points[:, :2].astype(np.float32)
        print(f"[INFO] Successfully extracted {xy_points.shape[0]} LiDAR points for BEV")
        return xy_points
    
    except Exception as e:
        import traceback
        print(f"[DEBUG] Error in load_lidar_points: {e}")
        traceback.print_exc()
        return None


def get_bottom_4_points(corners_3d):
    """Extract bottom 4 corners from 8 corners of a box.
    
    Returns points ordered by angle from centroid to form a proper rectangle.
    """
    # Get z coordinates
    z_coords = corners_3d[:, 2]
    
    # Find indices of 4 lowest z values (bottom corners)
    bottom_indices = np.argsort(z_coords)[:4]
    bottom_pts = corners_3d[bottom_indices, :2]  # x, y only
    
    # Sort points by angle from centroid to form a proper polygon
    centroid = bottom_pts.mean(axis=0)
    angles = np.arctan2(bottom_pts[:, 1] - centroid[1], bottom_pts[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    
    return bottom_pts[sorted_indices]


def draw_boxes_on_bev(ax, boxes_3d, color, alpha=0.7, linewidth=2):
    """Draw boxes on BEV plot using bottom 4 points."""
    if boxes_3d is None or len(boxes_3d) == 0:
        return 0
    
    corners_3d = boxes_3d.corners
    if torch.is_tensor(corners_3d):
        corners_3d = corners_3d.detach().cpu().numpy()
    else:
        corners_3d = np.asarray(corners_3d)
    
    num_drawn = 0
    
    for i in range(corners_3d.shape[0]):
        try:
            bottom_pts = get_bottom_4_points(corners_3d[i])
            
            if not np.all(np.isfinite(bottom_pts)):
                continue
            
            # Close the loop for rectangle
            points = np.vstack([bottom_pts, bottom_pts[0:1]])
            ax.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth, alpha=alpha)
            ax.fill(bottom_pts[:, 0], bottom_pts[:, 1], color=color, alpha=alpha*0.3)
            num_drawn += 1
        except Exception as e:
            continue
    
    return num_drawn


def visualize_bev(img_meta, data, gt_boxes, boxes_a, boxes_b, scores_a, scores_b, 
                  score_thr, output_path, lidar_range=50.0, 
                  gt_label="ENTER_TEXT_HERE", model_a_label="ENTER_TEXT_HERE", model_b_label="ENTER_TEXT_HERE"):
    """Create BEV visualization with three subplots.
    
    Args:
        lidar_range: Maximum range in meters for X and Y axes (e.g., 50 shows -50 to +50m)
        gt_label: Label for GT subplot
        model_a_label: Label for Model A subplot
        model_b_label: Label for Model B subplot
    """
    
    # Load LiDAR points from data
    lidar_points = load_lidar_points(data)
    
    if lidar_points is None:
        print("[WARN] Could not extract LiDAR points for BEV visualization.")
        return
    
    # Filter boxes by score
    gt_boxes_filtered, _ = filter_boxes_by_score(gt_boxes, None, score_thr)
    boxes_a_filtered, _ = filter_boxes_by_score(boxes_a, scores_a, score_thr)
    boxes_b_filtered, _ = filter_boxes_by_score(boxes_b, scores_b, score_thr)
    
    # Create figure with three subplots (no figure title)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Helper function to plot BEV
    def plot_bev(ax, lidar_pts, boxes, title, color):
        # Plot LiDAR points
        if lidar_pts is not None and len(lidar_pts) > 0:
            ax.scatter(lidar_pts[:, 0], lidar_pts[:, 1], c='gray', s=1, alpha=0.3, label='LiDAR')
        
        # Draw boxes
        num_boxes = draw_boxes_on_bev(ax, boxes, color=color)
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        # Set fixed limits based on lidar_range parameter
        ax.set_xlim(-lidar_range, lidar_range)
        ax.set_ylim(-lidar_range, lidar_range)
    
    # Plot GT
    plot_bev(axes[0], lidar_points, gt_boxes_filtered, gt_label, color='red')
    
    # Plot Model A
    plot_bev(axes[1], lidar_points, boxes_a_filtered, model_a_label, color='green')
    
    # Plot Model B
    plot_bev(axes[2], lidar_points, boxes_b_filtered, model_b_label, color='blue')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"[INFO] BEV visualization saved to {output_path}")
    plt.close()


def print_model_output_debug(name, result):
    print(f"\n[DEBUG] ===== MODEL OUTPUT DEBUG: {name} =====")
    print(f"[DEBUG] result type: {type(result)}")

    if isinstance(result, dict):
        print(f"[DEBUG] result keys: {list(result.keys())}")
        if "pts_bbox" in result:
            print(f"[DEBUG] pts_bbox keys: {list(result['pts_bbox'].keys())}")
        else:
            print(f"[DEBUG] result direct keys: {list(result.keys())}")

    print(f"[DEBUG] ===== END MODEL OUTPUT DEBUG: {name} =====\n")


def main():
    args = parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    import_plugins(cfg, args.config)
    cfg = prepare_cfg(cfg)

    dataset = build_dataset(cfg.data.test)

    sample_idx = find_sample_idx(dataset, args.target_img, args.camera)

    if sample_idx == -1:
        raise ValueError(f"Sample containing {args.target_img} not found in the test dataset split.")

    print(f"[INFO] Found target sample at dataset index {sample_idx}")

    data = dataset[sample_idx]
    data = collate([data], samples_per_gpu=1)

    img_meta = get_img_meta(data)
    cam_idx = find_camera_index(img_meta, args.target_img, args.camera)

    print_img_meta_summary(img_meta, cam_idx)

    if args.debug:
        print_matrix_like_meta(img_meta, cam_idx)

    img, img_path = load_image_for_camera(img_meta, cam_idx, args.draw_space)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script because MMDataParallel(model.cuda()) is used.")

    model_a = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    model_b = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    load_checkpoint(model_a, args.checkpoint_a, map_location="cpu")
    load_checkpoint(model_b, args.checkpoint_b, map_location="cpu")

    model_a = MMDataParallel(model_a.cuda(), device_ids=[0])
    model_b = MMDataParallel(model_b.cuda(), device_ids=[0])

    model_a.eval()
    model_b.eval()

    result_a = run_inference(model_a, data)
    result_b = run_inference(model_b, data)

    if args.debug:
        print_model_output_debug("Model A", result_a)
        print_model_output_debug("Model B", result_b)

    boxes_a, scores_a = extract_boxes_scores(result_a)
    boxes_b, scores_b = extract_boxes_scores(result_b)

    ann_info = dataset.get_ann_info(sample_idx)
    gt_boxes = ann_info.get("gt_bboxes_3d", None)

    if gt_boxes is None:
        print("[WARN] No gt_bboxes_3d found for this sample. Only predictions will be drawn.")
    else:
        gt_boxes, _ = filter_gt_boxes_by_visibility(gt_boxes, ann_info, good_visibility_only=True)

    if args.debug:
        debug_print_boxes("GT raw before projection", gt_boxes, None, args.max_debug_boxes)
        debug_print_boxes("Model A raw before projection", boxes_a, scores_a, args.max_debug_boxes)
        debug_print_boxes("Model B raw before projection", boxes_b, scores_b, args.max_debug_boxes)

    # Get class names for GT boxes
    gt_class_names = get_class_names_for_boxes(gt_boxes, ann_info, dataset)

    # Create three separate images
    img_gt = img.copy()
    img_model_a = img.copy()
    img_model_b = img.copy()

    # Draw on separate images
    project_and_draw(
        gt_boxes,
        None,
        img_gt,
        color=(255, 0, 0),
        img_meta=img_meta,
        cam_idx=cam_idx,
        score_thr=args.score_thr,
        thickness=args.line_thickness,
        name="GT",
        projection_mode=args.projection_mode,
        z_shift=args.z_shift,
        debug=args.debug,
        max_debug_boxes=args.max_debug_boxes,
        class_names=gt_class_names,
        font_scale=args.font_scale,
        draw_class_names=args.draw_class_names,
    )

    project_and_draw(
        boxes_a,
        scores_a,
        img_model_a,
        color=(0, 255, 0),
        img_meta=img_meta,
        cam_idx=cam_idx,
        score_thr=args.score_thr,
        thickness=args.line_thickness,
        name="Model A",
        projection_mode=args.projection_mode,
        z_shift=args.z_shift,
        debug=args.debug,
        max_debug_boxes=args.max_debug_boxes,
        class_names=None,
        font_scale=args.font_scale,
        draw_class_names=args.draw_class_names,
    )

    project_and_draw(
        boxes_b,
        scores_b,
        img_model_b,
        color=(0, 0, 255),
        img_meta=img_meta,
        cam_idx=cam_idx,
        score_thr=args.score_thr,
        thickness=args.line_thickness,
        name="Model B",
        projection_mode=args.projection_mode,
        z_shift=args.z_shift,
        debug=args.debug,
        max_debug_boxes=args.max_debug_boxes,
        class_names=None,
        font_scale=args.font_scale,
        draw_class_names=args.draw_class_names,
    )

    # Save three separate images
    base_out = args.out
    if base_out.endswith(".jpg") or base_out.endswith(".png"):
        base_out = base_out.rsplit(".", 1)[0]
    
    out_gt = f"{base_out}_gt.jpg"
    out_model_a = f"{base_out}_model_a.jpg"
    out_model_b = f"{base_out}_model_b.jpg"

    success_gt = cv2.imwrite(out_gt, img_gt)
    success_a = cv2.imwrite(out_model_a, img_model_a)
    success_b = cv2.imwrite(out_model_b, img_model_b)

    if not (success_gt and success_a and success_b):
        raise IOError(f"Failed to write one or more output images.")

    print(f"[INFO] GT image saved to {out_gt}")
    print(f"[INFO] Model A image saved to {out_model_a}")
    print(f"[INFO] Model B image saved to {out_model_b}")

    # Generate BEV visualizations
    out_bev = f"{base_out}_bev.jpg"
    visualize_bev(
        img_meta, data, gt_boxes, boxes_a, boxes_b, scores_a, scores_b,
        args.score_thr, out_bev, lidar_range=args.lidar_range,
        gt_label=args.bev_gt_label, model_a_label=args.bev_model_a_label,
        model_b_label=args.bev_model_b_label
    )


if __name__ == "__main__":
    main()