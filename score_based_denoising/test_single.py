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
    # "fx": 751.706129,
    # "fy": 766.315580,
    # "cx": 338.665467,
    # "cy": 257.986032,

    "fx": 583.16,
    "fy": 583.24,
    "cx": 318.95,
    "cy": 255.44,
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
parser.add_argument('--input_npy', type=str, default=None,
                    help='Input sparse depth (.npy) - can be 2D depth map (H, W) or point cloud (N, 3)')
parser.add_argument('--input_ply', type=str, default=None,
                    help='Input sparse point cloud (.ply) - alternative to --input_npy')
parser.add_argument('--output_npy', type=str, required=True,
                    help='Output denoised depth map (.npy) with shape (H, W)')
parser.add_argument('--output_xyz', type=str, default=None,
                    help='Output denoised point cloud (.xyz) - alternative format')
parser.add_argument('--image_path', type=str, default=None,
                    help='Path to RGB image to get resolution. Required if input is point cloud format.')
parser.add_argument('--height', type=int, default=None,
                    help='Image height (required if input is point cloud and no image_path provided)')
parser.add_argument('--width', type=int, default=None,
                    help='Image width (required if input is point cloud and no image_path provided)')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=2020)
args = parser.parse_args()
seed_all(args.seed)

# Model
ckpt = torch.load(args.ckpt, map_location=args.device)
model = DenoiseNet(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

# Load input data (支持PLY或NPY)
def load_ply_to_xyz(ply_path):
    """加载PLY文件并转换为点云数组"""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points).astype(np.float32)
        print(f'[INFO] Loaded PLY file: {ply_path} with {len(points)} points')
        return points
    except ImportError:
        # Fallback: 手动解析PLY文件
        print('[INFO] open3d not available, using manual PLY parser')
        points = []
        with open(ply_path, 'r') as f:
            lines = f.readlines()
            header_end = False
            num_vertices = 0
            for i, line in enumerate(lines):
                if 'element vertex' in line:
                    num_vertices = int(line.split()[-1])
                if 'end_header' in line:
                    header_end = True
                    continue
                if header_end and len(points) < num_vertices:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            points.append([x, y, z])
                        except ValueError:
                            continue
        points = np.array(points, dtype=np.float32)
        print(f'[INFO] Loaded PLY file: {ply_path} with {len(points)} points (manual parser)')
        return points

def save_xyz(points, path):
    """保存点云为XYZ格式"""
    np.savetxt(path, points, fmt='%.6f')
    print(f'[INFO] Saved XYZ file: {path} with {len(points)} points')

if args.input_ply:
    input_data = load_ply_to_xyz(args.input_ply)
    # 转换为(N, 3)格式
    if input_data.ndim == 1:
        input_data = input_data.reshape(-1, 3)
elif args.input_npy:
    input_data = np.load(args.input_npy)
else:
    raise ValueError('Either --input_npy or --input_ply must be provided')
    
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

# 使用相机内参将像素坐标转换到相机坐标系 (x, y, z)，再做去噪
fx, fy, cx, cy = _load_intrinsics()
depth_values = depth_map[row_idx, col_idx].astype(np.float32)
x_coords = (col_idx.astype(np.float32) - cx) * depth_values / fx
y_coords = (row_idx.astype(np.float32) - cy) * depth_values / fy
pts_xyz = np.stack([x_coords, y_coords, depth_values], axis=1).astype(np.float32)

original_depths = pts_xyz[:, 2].copy()
print(f'[INFO] Original depth range: [{original_depths.min():.4f}, {original_depths.max():.4f}]')

pcl = torch.from_numpy(pts_xyz).float()
pcl, center, scale = NormalizeUnitSphere.normalize(pcl)
print(f'[INFO] Normalization center: {center}, scale: {scale}')
pcl = pcl.to(args.device)

print('[INFO] Start denoising...')
pcl_denoised = patch_based_denoise(model, pcl).cpu()
pcl_denoised = pcl_denoised * scale + center
pcl_denoised_np = pcl_denoised.numpy()
print('[INFO] Finished denoising.')

# 将去噪后的点重新投影回深度图
denoised_z = pcl_denoised_np[:, 2]
denoised_x = pcl_denoised_np[:, 0]
denoised_y = pcl_denoised_np[:, 1]
print(f'[INFO] Denoised depth range: [{denoised_z.min():.4f}, {denoised_z.max():.4f}]')
print(f'[INFO] Valid denoised depths (>0): {(denoised_z > 0).sum()} / {len(denoised_z)}')

# 过滤无效/负深度
valid_mask = denoised_z > 0
denoised_x, denoised_y, denoised_z = denoised_x[valid_mask], denoised_y[valid_mask], denoised_z[valid_mask]

# 投影到像素坐标
u_proj = np.round((denoised_x * fx) / denoised_z + cx).astype(np.int32)
v_proj = np.round((denoised_y * fy) / denoised_z + cy).astype(np.int32)
pixel_valid = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H)
u_proj, v_proj, denoised_z = u_proj[pixel_valid], v_proj[pixel_valid], denoised_z[pixel_valid]

# z-buffer: 对同一像素取最近深度
denoised_depth = np.full((H, W), np.inf, dtype=np.float32)
np.minimum.at(denoised_depth, (v_proj, u_proj), denoised_z)
denoised_depth[~np.isfinite(denoised_depth)] = 0.0

print(f'[INFO] Saving denoised depth map ({H}x{W}) to: {args.output_npy}')
np.save(args.output_npy, denoised_depth)
print(f'[INFO] Output shape: {denoised_depth.shape}, valid pixels: {(denoised_depth > 0).sum()}')

# 如果指定了output_xyz，也保存为XYZ格式（直接使用去噪后的3D点）
if args.output_xyz:
    save_xyz(pcl_denoised_np, args.output_xyz)
