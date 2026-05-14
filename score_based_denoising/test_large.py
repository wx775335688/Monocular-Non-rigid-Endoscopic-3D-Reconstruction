import os
import time
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from utils.misc import *
from utils.denoise import *
from utils.transforms import *
from models.denoise import *

# ----------------------------------------------------------------------
# 请把相机内参写死在这里
# 例如：
# CAM_INTRINSICS = {
#     "fx": 718.856,
#     "fy": 718.856,
#     "cx": 607.1928,
#     "cy": 185.2157,
# }
# ----------------------------------------------------------------------
CAM_INTRINSICS = {
    "fx": 751.706129,
    "fy": 766.315580,
    "cx": 338.665467,
    "cy": 257.986032,
}


def _determine_resolution(args):
    if args.image_path and os.path.exists(args.image_path):
        img = Image.open(args.image_path)
        return img.size[1], img.size[0]  # (H, W)
    if args.height is not None and args.width is not None:
        return args.height, args.width
    raise ValueError('Image resolution unknown. Provide --image_path or both --height and --width.')


def _load_intrinsics():
    fx = CAM_INTRINSICS.get("fx")
    fy = CAM_INTRINSICS.get("fy")
    cx = CAM_INTRINSICS.get("cx")
    cy = CAM_INTRINSICS.get("cy")
    if None in (fx, fy, cx, cy):
        raise ValueError('请在 CAM_INTRINSICS 中写入 fx/fy/cx/cy 的具体值。')
    return fx, fy, cx, cy


def _looks_like_pixel_space(points, H, W):
    if points.shape[0] == 0:
        return False
    coords = points[:, :2]
    if not np.allclose(coords, np.round(coords)):
        return False
    if coords[:, 0].min() < 0 or coords[:, 0].max() >= W:
        return False
    if coords[:, 1].min() < 0 or coords[:, 1].max() >= H:
        return False
    return True

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/ckpt.pt')
parser.add_argument('--input_npy', type=str, required=True,
                    help='Input sparse depth (.npy) - can be 2D depth map (H, W) or point cloud (N, 3)')
parser.add_argument('--output_npy', type=str, required=True,
                    help='Output denoised depth map (.npy) with shape (H, W)')
parser.add_argument('--image_path', type=str, default=None,
                    help='Path to RGB image to get resolution. Required if input is point cloud format.')
parser.add_argument('--height', type=int, default=None,
                    help='Image height (required if input is point cloud and no image_path provided)')
parser.add_argument('--width', type=int, default=None,
                    help='Image width (required if input is point cloud and no image_path provided)')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=2020)
# Denoiser parameters
parser.add_argument('--cluster_size', type=int, default=30000)
args = parser.parse_args()
seed_all(args.seed)

# Model
ckpt = torch.load(args.ckpt, map_location=args.device)
model = DenoiseNet(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

# Load input data
input_data = np.load(args.input_npy)
print(f'[INFO] Loaded input data with shape: {input_data.shape}')

# Determine input format and get image resolution
if input_data.ndim == 2 and input_data.shape[1] == 3:
    # Point cloud format (N, 3) - determine if coordinates are already pixel indices or 3D points
    print('[INFO] Input is point cloud format (N, 3), converting to depth map...')
    H, W = _determine_resolution(args)
    print(f'[INFO] Using image resolution: {H}x{W}')

    if _looks_like_pixel_space(input_data, H, W):
        print('[INFO] Detected pixel-space coordinates [u, v, depth]')
        u_coords = input_data[:, 0].astype(np.int32)
        v_coords = input_data[:, 1].astype(np.int32)
        depths = input_data[:, 2].astype(np.float32)
    else:
        print('[INFO] Detected camera-space coordinates [x, y, z]; projecting with intrinsics')
        intrinsics = _load_intrinsics()
        if intrinsics is None:
            raise ValueError('Camera intrinsics required. Provide --camera_matrix (.npy) or --fx/--fy/--cx/--cy.')
        fx, fy, cx, cy = intrinsics
        x, y, depths = input_data[:, 0].astype(np.float32), input_data[:, 1].astype(np.float32), input_data[:, 2].astype(
            np.float32)
        valid = depths > 0
        x, y, depths = x[valid], y[valid], depths[valid]
        u_coords = np.round((x / depths) * fx + cx).astype(np.int32)
        v_coords = np.round((y / depths) * fy + cy).astype(np.int32)

    depth_map = np.zeros((H, W), dtype=np.float32)
    valid_mask = (u_coords >= 0) & (u_coords < W) & (v_coords >= 0) & (v_coords < H) & (depths > 0)
    u_coords = u_coords[valid_mask]
    v_coords = v_coords[valid_mask]
    depths = depths[valid_mask]
    depth_map[v_coords, u_coords] = depths

    row_idx, col_idx = v_coords, u_coords
    print(f'[INFO] Converted {len(depths)} points to depth map of size {H}x{W}')

elif input_data.ndim == 2:
    # Depth map format (H, W)
    depth_map = input_data.astype(np.float32)
    H, W = depth_map.shape
    print(f'[INFO] Input is depth map format: {H}x{W}')
    mask = depth_map > 0
    if not mask.any():
        raise ValueError('Input depth map contains no positive depth values.')
    row_idx, col_idx = np.nonzero(mask)
else:
    raise ValueError(f'Unsupported input format. Expected 2D array (H, W) or point cloud (N, 3), got shape {input_data.shape}')

# Extract points for denoising: [col, row, depth]
pts = np.stack([col_idx, row_idx, depth_map[row_idx, col_idx]], axis=1).astype(np.float32)
original_depths = pts[:, 2].copy()
print(f'[INFO] Original depth range: [{original_depths.min():.4f}, {original_depths.max():.4f}]')

pcl = torch.from_numpy(pts).float()
pcl, center, scale = NormalizeUnitSphere.normalize(pcl)
print(f'[INFO] Normalization center: {center}, scale: {scale}')
pcl = pcl.to(args.device)

print('[INFO] Start large point cloud denoising...')
pcl_denoised = denoise_large_pointcloud(
    model=model,
    pcl=pcl,
    cluster_size=args.cluster_size,
    seed=args.seed
)
pcl_denoised = pcl_denoised.cpu()
pcl_denoised = pcl_denoised * scale + center
pcl_denoised_np = pcl_denoised.numpy()
print('[INFO] Finished denoising.')

# Extract denoised depths (only use the depth channel, keep original pixel coordinates)
denoised_depths = pcl_denoised_np[:, 2]
print(f'[INFO] Denoised depth range: [{denoised_depths.min():.4f}, {denoised_depths.max():.4f}]')
print(f'[INFO] Valid denoised depths (>0): {(denoised_depths > 0).sum()} / {len(denoised_depths)}')

# Ensure all depths are positive (clamp negative values to 0)
denoised_depths = np.maximum(denoised_depths, 0.0)

# Create output depth map with same resolution as input
denoised_depth = np.zeros((H, W), dtype=np.float32)
# Use original pixel coordinates to place denoised depth values
denoised_depth[row_idx, col_idx] = denoised_depths

print(f'[INFO] Saving denoised depth map ({H}x{W}) to: {args.output_npy}')
np.save(args.output_npy, denoised_depth)
print(f'[INFO] Output shape: {denoised_depth.shape}, valid pixels: {(denoised_depth > 0).sum()}')
