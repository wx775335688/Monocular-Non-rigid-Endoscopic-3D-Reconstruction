"""
THPG-guided anchor-preserving inference script for score-based point-cloud denoising.

This file keeps the original score-based denoising backbone unchanged and adds an
optional post-denoising refinement stage for registration-oriented endoscopic
reconstruction:

1) anchor preservation: keeps high-confidence sparse/depth anchors close to their
   original pseudo-metric positions;
2) THPG/pixel-topology preservation: preserves local edge lengths defined by an
   imported THPG edge file or automatically built image-domain neighbor edges;
3) semantic/depth boundary awareness: suppresses topology constraints across
   semantic masks or large depth jumps;
4) optional depth affine correction: aligns refined depths back to the original
   anchor depth convention.

Run without --innovation_mode to reproduce the original behavior as closely as
possible. Run with --innovation_mode to enable the proposed refinement.
"""

import os
import json
import argparse
import numpy as np
import torch
from PIL import Image

from utils.misc import *
from utils.denoise import *
from utils.transforms import *
from models.denoise import *

# ----------------------------------------------------------------------
# 请把相机内参写死在这里
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
    return float(fx), float(fy), float(cx), float(cy)


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


def load_ply_to_xyz(ply_path):
    """加载PLY文件并转换为点云数组。"""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points).astype(np.float32)
        print(f'[INFO] Loaded PLY file: {ply_path} with {len(points)} points')
        return points
    except ImportError:
        print('[INFO] open3d not available, using manual PLY parser')
        points = []
        with open(ply_path, 'r') as f:
            lines = f.readlines()
            header_end = False
            num_vertices = 0
            for line in lines:
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
    """保存点云为XYZ格式。"""
    np.savetxt(path, points, fmt='%.6f')
    print(f'[INFO] Saved XYZ file: {path} with {len(points)} points')


def _load_optional_array(path, name):
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f'{name} not found: {path}')
    arr = np.load(path)
    print(f'[INFO] Loaded {name}: {path}, shape={arr.shape}')
    return arr


def _vector_from_map_or_vector(arr, row_idx, col_idx, H, W, N, name):
    """支持(H,W)图，也支持长度为N的一维向量。"""
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape == (H, W):
        return arr[row_idx, col_idx]
    if arr.ndim == 1 and arr.shape[0] == N:
        return arr
    if arr.ndim == 2 and arr.shape[0] == N and arr.shape[1] == 1:
        return arr[:, 0]
    raise ValueError(
        f'{name} must be a map with shape {(H, W)} or a vector with length {N}, got {arr.shape}.'
    )


def _build_index_map(row_idx, col_idx, H, W):
    index_map = np.full((H, W), -1, dtype=np.int64)
    index_map[row_idx, col_idx] = np.arange(len(row_idx), dtype=np.int64)
    return index_map


def _load_or_build_edges(args, row_idx, col_idx, depth_values, H, W, N, semantic_vec=None):
    """
    若提供--thpg_edges_npy，则读取前端THPG边；否则根据像素4邻域自动构建伪拓扑边。
    自动边不会跨越语义边界；若设置--boundary_depth_thresh，也不会跨越明显深度跳变。
    """
    if args.thpg_edges_npy:
        edges = np.load(args.thpg_edges_npy)
        edges = np.asarray(edges, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1] < 2:
            raise ValueError('--thpg_edges_npy must have shape (M, 2) or (M, >=2).')
        edges = edges[:, :2]
        valid = (
            (edges[:, 0] >= 0) & (edges[:, 0] < N) &
            (edges[:, 1] >= 0) & (edges[:, 1] < N) &
            (edges[:, 0] != edges[:, 1])
        )
        edges = edges[valid]
        print(f'[INFO] Loaded THPG edges: {len(edges)} valid edges')
    else:
        index_map = _build_index_map(row_idx, col_idx, H, W)
        candidates = []
        # right neighbor
        valid_r = col_idx + 1 < W
        i_r = np.where(valid_r)[0]
        j_r = index_map[row_idx[i_r], col_idx[i_r] + 1]
        keep_r = j_r >= 0
        if keep_r.any():
            candidates.append(np.stack([i_r[keep_r], j_r[keep_r]], axis=1))
        # down neighbor
        valid_d = row_idx + 1 < H
        i_d = np.where(valid_d)[0]
        j_d = index_map[row_idx[i_d] + 1, col_idx[i_d]]
        keep_d = j_d >= 0
        if keep_d.any():
            candidates.append(np.stack([i_d[keep_d], j_d[keep_d]], axis=1))
        if len(candidates) == 0:
            edges = np.zeros((0, 2), dtype=np.int64)
        else:
            edges = np.concatenate(candidates, axis=0).astype(np.int64)
        print(f'[INFO] Auto-built image-domain topology edges: {len(edges)}')

    if len(edges) == 0:
        return edges, np.zeros((0,), dtype=np.float32)

    edge_weights = np.ones((len(edges),), dtype=np.float32)

    if semantic_vec is not None:
        same_sem = semantic_vec[edges[:, 0]] == semantic_vec[edges[:, 1]]
        edge_weights *= same_sem.astype(np.float32)
        print(f'[INFO] Semantic-boundary filtering kept {int((edge_weights > 0).sum())}/{len(edge_weights)} edges')

    if args.boundary_depth_thresh > 0:
        dz = np.abs(depth_values[edges[:, 0]] - depth_values[edges[:, 1]])
        same_depth = dz <= args.boundary_depth_thresh
        edge_weights *= same_depth.astype(np.float32)
        print(f'[INFO] Depth-boundary filtering kept {int((edge_weights > 0).sum())}/{len(edge_weights)} edges')

    valid_edges = edge_weights > 0
    edges = edges[valid_edges]
    edge_weights = edge_weights[valid_edges]

    if len(edges) > args.max_edges:
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(len(edges), size=args.max_edges, replace=False)
        edges = edges[chosen]
        edge_weights = edge_weights[chosen]
        print(f'[INFO] Randomly sampled edges to max_edges={args.max_edges}')

    return edges, edge_weights.astype(np.float32)


def _weighted_affine_match_depth(z_src, z_ref, weights):
    """求解 min sum_i w_i * (s*z_src_i + b - z_ref_i)^2。"""
    z_src = z_src.astype(np.float64)
    z_ref = z_ref.astype(np.float64)
    weights = weights.astype(np.float64)
    valid = np.isfinite(z_src) & np.isfinite(z_ref) & np.isfinite(weights) & (weights > 1e-6)
    if valid.sum() < 3:
        return 1.0, 0.0
    x = z_src[valid]
    y = z_ref[valid]
    w = weights[valid]
    sw = w.sum()
    mx = (w * x).sum() / sw
    my = (w * y).sum() / sw
    var_x = (w * (x - mx) ** 2).sum()
    if var_x < 1e-12:
        return 1.0, 0.0
    cov_xy = (w * (x - mx) * (y - my)).sum()
    s = cov_xy / var_x
    b = my - s * mx
    return float(s), float(b)


def _topology_error(points, reference, edges, edge_weights):
    if edges is None or len(edges) == 0:
        return None
    i, j = edges[:, 0], edges[:, 1]
    ref_len = np.linalg.norm(reference[j] - reference[i], axis=1)
    cur_len = np.linalg.norm(points[j] - points[i], axis=1)
    err = np.abs(cur_len - ref_len)
    w = edge_weights.astype(np.float64)
    if w.sum() <= 1e-12:
        return float(err.mean())
    return float((err * w).sum() / w.sum())


def _anchor_drift(points, reference, weights):
    w = weights.astype(np.float64)
    valid = w > 1e-6
    if valid.sum() == 0:
        return None
    drift = np.linalg.norm(points[valid] - reference[valid], axis=1)
    return float((drift * w[valid]).sum() / w[valid].sum())


def _apply_topology_preservation(points, reference, edges, edge_weights, confidence, weight=0.2, iters=5):
    """
    迭代保持局部边长，作为THPG拓扑保持项的轻量实现。
    参考边长来自去噪前点云，因此不会引入新的监督标签。
    """
    if edges is None or len(edges) == 0 or weight <= 0 or iters <= 0:
        return points

    points = points.copy()
    i = edges[:, 0]
    j = edges[:, 1]
    ref_vec = reference[j] - reference[i]
    ref_len = np.linalg.norm(ref_vec, axis=1).astype(np.float32)
    valid = ref_len > 1e-8
    i, j = i[valid], j[valid]
    ref_len = ref_len[valid]
    ew = edge_weights[valid].astype(np.float32)
    ew *= np.sqrt(np.clip(confidence[i], 0.0, 1.0) * np.clip(confidence[j], 0.0, 1.0))
    valid_w = ew > 1e-6
    i, j, ref_len, ew = i[valid_w], j[valid_w], ref_len[valid_w], ew[valid_w]
    if len(i) == 0:
        return points

    for _ in range(iters):
        diff = points[j] - points[i]
        cur_len = np.linalg.norm(diff, axis=1).astype(np.float32)
        valid_len = cur_len > 1e-8
        if not valid_len.any():
            break

        ii = i[valid_len]
        jj = j[valid_len]
        cur = cur_len[valid_len]
        ref = ref_len[valid_len]
        w = ew[valid_len]
        d = diff[valid_len]

        corr = ((cur - ref) / cur)[:, None] * d
        step = (0.5 * weight * w)[:, None] * corr

        accum = np.zeros_like(points, dtype=np.float32)
        wsum = np.zeros((points.shape[0], 1), dtype=np.float32)
        np.add.at(accum, ii, step)
        np.add.at(accum, jj, -step)
        np.add.at(wsum, ii, w[:, None])
        np.add.at(wsum, jj, w[:, None])
        points += accum / (wsum + 1e-6)

    return points


def thpg_guided_anchor_preserving_refinement(
    args,
    original_points,
    denoised_points,
    row_idx,
    col_idx,
    depth_values,
    H,
    W,
    fx,
    fy,
    cx,
    cy,
):
    """在原始score-based输出之后增加THPG-guided约束。"""
    N = original_points.shape[0]
    report = {}

    anchor_mask_map = _load_optional_array(args.anchor_mask_npy, 'anchor mask')
    anchor_conf_arr = _load_optional_array(args.anchor_conf_npy, 'anchor confidence')
    semantic_mask_map = _load_optional_array(args.semantic_mask_npy, 'semantic mask')

    anchor_mask_vec = _vector_from_map_or_vector(anchor_mask_map, row_idx, col_idx, H, W, N, 'anchor_mask')
    anchor_conf_vec = _vector_from_map_or_vector(anchor_conf_arr, row_idx, col_idx, H, W, N, 'anchor_conf')
    semantic_vec = _vector_from_map_or_vector(semantic_mask_map, row_idx, col_idx, H, W, N, 'semantic_mask')

    if anchor_mask_vec is None:
        # 没有显式锚点时，把所有输入点视为弱伪锚点，用于抑制score-denoise造成的整体漂移/收缩。
        anchor_mask_vec = np.ones((N,), dtype=np.float32)
        base_anchor_weight = args.anchor_weight
        print('[INFO] No anchor mask provided. All input points are treated as weak pseudo-anchors.')
    else:
        anchor_mask_vec = (anchor_mask_vec > 0).astype(np.float32)
        base_anchor_weight = args.strong_anchor_weight
        print(f'[INFO] Explicit anchor mask: {int(anchor_mask_vec.sum())}/{N} anchor points')

    if anchor_conf_vec is None:
        anchor_conf_vec = np.ones((N,), dtype=np.float32)
    else:
        anchor_conf_vec = np.asarray(anchor_conf_vec, dtype=np.float32)
        # 兼容0-255置信图
        if anchor_conf_vec.max() > 1.0:
            anchor_conf_vec = anchor_conf_vec / 255.0
        anchor_conf_vec = np.clip(anchor_conf_vec, 0.0, 1.0)

    anchor_weights = np.clip(base_anchor_weight * anchor_mask_vec * anchor_conf_vec, 0.0, 1.0).astype(np.float32)

    edges, edge_weights = _load_or_build_edges(
        args=args,
        row_idx=row_idx,
        col_idx=col_idx,
        depth_values=depth_values,
        H=H,
        W=W,
        N=N,
        semantic_vec=semantic_vec,
    )

    refined = denoised_points.astype(np.float32).copy()

    report['anchor_drift_before'] = _anchor_drift(refined, original_points, anchor_weights)
    report['topology_error_before'] = _topology_error(refined, original_points, edges, edge_weights)

    # 1) THPG/pixel-topology preserving correction.
    refined = _apply_topology_preservation(
        refined,
        original_points,
        edges,
        edge_weights,
        confidence=anchor_conf_vec,
        weight=args.topology_weight,
        iters=args.topology_iters,
    )

    # 2) Anchor-preserving correction.
    # 高置信锚点被强约束，低置信/非锚点更多保留score-based去噪结果。
    refined = (1.0 - anchor_weights[:, None]) * refined + anchor_weights[:, None] * original_points

    # 3) Optional affine depth correction to keep pseudo-metric depth convention.
    if args.preserve_depth_affine:
        s, b = _weighted_affine_match_depth(refined[:, 2], original_points[:, 2], anchor_weights)
        refined[:, 2] = s * refined[:, 2] + b
        report['depth_affine_scale'] = s
        report['depth_affine_shift'] = b
        print(f'[INFO] Applied weighted depth affine correction: scale={s:.6f}, shift={b:.6f}')

    # 4) Optional ray consistency for depth-map output.
    # 若输出是深度图，保持像素位置不变通常比重新投影更稳定，可减少空洞和z-buffer冲突。
    if args.keep_pixel_positions:
        refined[:, 2] = np.maximum(refined[:, 2], 1e-6)
        refined[:, 0] = (col_idx.astype(np.float32) - cx) * refined[:, 2] / fx
        refined[:, 1] = (row_idx.astype(np.float32) - cy) * refined[:, 2] / fy
        print('[INFO] Kept original pixel positions and updated XYZ along camera rays.')

    report['anchor_drift_after'] = _anchor_drift(refined, original_points, anchor_weights)
    report['topology_error_after'] = _topology_error(refined, original_points, edges, edge_weights)
    report['num_points'] = int(N)
    report['num_edges'] = int(len(edges))
    report['anchor_weight_mean'] = float(anchor_weights.mean())
    report['anchor_weight_nonzero'] = int((anchor_weights > 1e-6).sum())

    return refined.astype(np.float32), report


def points_to_depth_by_projection(points, H, W, fx, fy, cx, cy):
    """将3D点重新投影回深度图，采用z-buffer保留最近深度。"""
    z = points[:, 2]
    x = points[:, 0]
    y = points[:, 1]
    valid_mask = z > 0
    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]

    u_proj = np.round((x * fx) / z + cx).astype(np.int32)
    v_proj = np.round((y * fy) / z + cy).astype(np.int32)
    pixel_valid = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H)
    u_proj, v_proj, z = u_proj[pixel_valid], v_proj[pixel_valid], z[pixel_valid]

    depth = np.full((H, W), np.inf, dtype=np.float32)
    np.minimum.at(depth, (v_proj, u_proj), z)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def points_to_depth_keep_pixels(points, row_idx, col_idx, H, W):
    """保持输入有效像素位置，只更新深度值。"""
    depth = np.zeros((H, W), dtype=np.float32)
    z = points[:, 2]
    valid = z > 0
    depth[row_idx[valid], col_idx[valid]] = z[valid].astype(np.float32)
    return depth


def parse_args():
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

    # ---------------- Innovation options ----------------
    parser.add_argument('--innovation_mode', action='store_true',
                        help='Enable THPG-guided anchor-preserving refinement after vanilla score denoising.')
    parser.add_argument('--anchor_mask_npy', type=str, default=None,
                        help='Optional anchor mask. Shape can be (H,W) or (N,). Nonzero entries are high-confidence anchors.')
    parser.add_argument('--anchor_conf_npy', type=str, default=None,
                        help='Optional anchor/tracking confidence. Shape can be (H,W) or (N,). Values are expected in [0,1] or [0,255].')
    parser.add_argument('--semantic_mask_npy', type=str, default=None,
                        help='Optional semantic/SAM mask with shape (H,W) or labels with shape (N,). Edges across different labels are suppressed.')
    parser.add_argument('--thpg_edges_npy', type=str, default=None,
                        help='Optional THPG edge index array with shape (M,2). If not provided, image-domain 4-neighbor edges are built.')
    parser.add_argument('--anchor_weight', type=float, default=0.30,
                        help='Weak pseudo-anchor preserving strength when no explicit anchor mask is provided.')
    parser.add_argument('--strong_anchor_weight', type=float, default=0.85,
                        help='Anchor preserving strength for explicit anchor mask.')
    parser.add_argument('--topology_weight', type=float, default=0.20,
                        help='Strength of topology-preserving edge-length correction.')
    parser.add_argument('--topology_iters', type=int, default=5,
                        help='Number of topology-preserving correction iterations.')
    parser.add_argument('--boundary_depth_thresh', type=float, default=0.0,
                        help='If >0, suppress topology edges whose original depth difference exceeds this threshold.')
    parser.add_argument('--max_edges', type=int, default=300000,
                        help='Maximum number of topology edges used for refinement.')
    parser.add_argument('--preserve_depth_affine', action='store_true',
                        help='Fit weighted affine scale/shift on refined depth to match original anchor depth convention.')
    parser.add_argument('--keep_pixel_positions', action='store_true',
                        help='Keep original valid pixel positions in depth-map output and update XYZ along camera rays.')
    parser.add_argument('--output_report', type=str, default=None,
                        help='Optional JSON report for anchor drift/topology error before and after refinement.')
    return parser.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    # Model
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = DenoiseNet(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Load input data
    if args.input_ply:
        input_data = load_ply_to_xyz(args.input_ply)
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 3)
    elif args.input_npy:
        input_data = np.load(args.input_npy)
    else:
        raise ValueError('Either --input_npy or --input_ply must be provided')

    print(f'[INFO] Loaded input data with shape: {input_data.shape}')

    # Determine input format and get image resolution
    if input_data.ndim == 2 and input_data.shape[1] == 3:
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
            fx, fy, cx, cy = _load_intrinsics()
            x = input_data[:, 0].astype(np.float32)
            y = input_data[:, 1].astype(np.float32)
            depths = input_data[:, 2].astype(np.float32)
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

        # 重新从depth_map提取唯一有效像素，避免同一像素多个点导致索引和深度不一致。
        row_idx, col_idx = np.nonzero(depth_map > 0)
        print(f'[INFO] Converted {len(row_idx)} unique valid pixels to depth map of size {H}x{W}')

    elif input_data.ndim == 2:
        depth_map = input_data.astype(np.float32)
        H, W = depth_map.shape
        print(f'[INFO] Input is depth map format: {H}x{W}')
        mask = depth_map > 0
        if not mask.any():
            raise ValueError('Input depth map contains no positive depth values.')
        row_idx, col_idx = np.nonzero(mask)
    else:
        raise ValueError(
            f'Unsupported input format. Expected 2D array (H, W) or point cloud (N, 3), got shape {input_data.shape}'
        )

    # 使用相机内参将像素坐标转换到相机坐标系 (x, y, z)，再做去噪。
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

    print('[INFO] Start vanilla score-based denoising...')
    with torch.no_grad():
        pcl_denoised = patch_based_denoise(model, pcl).cpu()
    pcl_denoised = pcl_denoised * scale + center
    pcl_denoised_np = pcl_denoised.numpy().astype(np.float32)
    print('[INFO] Finished vanilla score-based denoising.')

    final_points = pcl_denoised_np
    report = {}
    if args.innovation_mode:
        print('[INFO] Start THPG-guided anchor-preserving refinement...')
        final_points, report = thpg_guided_anchor_preserving_refinement(
            args=args,
            original_points=pts_xyz,
            denoised_points=pcl_denoised_np,
            row_idx=row_idx,
            col_idx=col_idx,
            depth_values=depth_values,
            H=H,
            W=W,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
        print('[INFO] Finished THPG-guided anchor-preserving refinement.')
        print('[INFO] Refinement report:', json.dumps(report, indent=2))
        if args.output_report:
            with open(args.output_report, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            print(f'[INFO] Saved refinement report: {args.output_report}')

    final_z = final_points[:, 2]
    print(f'[INFO] Final depth range: [{final_z.min():.4f}, {final_z.max():.4f}]')
    print(f'[INFO] Valid final depths (>0): {(final_z > 0).sum()} / {len(final_z)}')

    # 保存深度图。
    if args.innovation_mode and args.keep_pixel_positions:
        denoised_depth = points_to_depth_keep_pixels(final_points, row_idx, col_idx, H, W)
    else:
        denoised_depth = points_to_depth_by_projection(final_points, H, W, fx, fy, cx, cy)

    print(f'[INFO] Saving denoised depth map ({H}x{W}) to: {args.output_npy}')
    np.save(args.output_npy, denoised_depth)
    print(f'[INFO] Output shape: {denoised_depth.shape}, valid pixels: {(denoised_depth > 0).sum()}')

    if args.output_xyz:
        save_xyz(final_points, args.output_xyz)


if __name__ == '__main__':
    main()
