import os
from typing import Optional, Tuple

import numpy as np


def load_point_cloud(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    读取点云文件，目前支持 .ply (ASCII) 与 .xyz/.pts/.txt。
    返回 (points, colors)；若无颜色信息则 colors 为 None。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.ply':
        return _load_ply(path)
    if ext in {'.xyz', '.pts', '.txt'}:
        return _load_xyz_like(path)
    raise ValueError(f'Unsupported point cloud format: {path}')


def save_point_cloud(path: str, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    """
    保存点云到指定路径，依据扩展名决定格式。
    points: (N, 3) float 数组；colors: (N, 3) uint8 或 None。
    """
    _ensure_parent_dir(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.ply':
        _save_ply(path, points, colors)
        return
    if ext in {'.xyz', '.pts', '.txt'}:
        np_points = _to_numpy(points, dtype=np.float32)
        np.savetxt(path, np_points, fmt='%.8f')
        return
    raise ValueError(f'Unsupported point cloud format: {path}')


def _load_xyz_like(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 3:
        raise ValueError(f'Point cloud file must have at least 3 columns: {path}')
    return data[:, :3], None


def _load_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with open(path, 'r') as f:
        header = _parse_ply_header(f)
        vertex_count = header['vertex_count']
        properties = header['vertex_props']

        points = np.zeros((vertex_count, 3), dtype=np.float32)
        has_color = {'red', 'green', 'blue'}.issubset(set(properties))
        colors = np.zeros((vertex_count, 3), dtype=np.uint8) if has_color else None

        for idx in range(vertex_count):
            line = f.readline()
            if not line:
                raise ValueError(f'Unexpected EOF while reading vertices in {path}')
            values = line.strip().split()
            if len(values) < len(properties):
                raise ValueError(f'Vertex line has insufficient properties in {path}: {line}')

            for prop_name, raw_val in zip(properties, values):
                if prop_name == 'x':
                    points[idx, 0] = float(raw_val)
                elif prop_name == 'y':
                    points[idx, 1] = float(raw_val)
                elif prop_name == 'z':
                    points[idx, 2] = float(raw_val)
                elif prop_name == 'red' and colors is not None:
                    colors[idx, 0] = _to_uint8(raw_val)
                elif prop_name == 'green' and colors is not None:
                    colors[idx, 1] = _to_uint8(raw_val)
                elif prop_name == 'blue' and colors is not None:
                    colors[idx, 2] = _to_uint8(raw_val)
                # 其他属性忽略

        return points, colors


def _save_ply(path: str, points: np.ndarray, colors: Optional[np.ndarray]) -> None:
    np_points = _to_numpy(points, dtype=np.float32)
    if np_points.ndim != 2 or np_points.shape[1] != 3:
        raise ValueError('points must have shape (N, 3)')

    if colors is None:
        np_colors = np.full((np_points.shape[0], 3), 255, dtype=np.uint8)
    else:
        np_colors = _to_numpy(colors, dtype=np.uint8)
        if np_colors.shape != (np_points.shape[0], 3):
            raise ValueError('colors must have shape (N, 3)')

    with open(path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {np_points.shape[0]}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for p, c in zip(np_points, np_colors):
            f.write(f'{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {int(c[0])} {int(c[1])} {int(c[2])}\n')


def _parse_ply_header(f) -> dict:
    first_line = f.readline().strip()
    if first_line != 'ply':
        raise ValueError('Invalid PLY file: missing header "ply".')
    format_line = f.readline().strip()
    if not format_line.startswith('format ascii'):
        raise ValueError('Only ASCII PLY format is supported.')

    vertex_count = None
    vertex_props = []
    current_element = None

    while True:
        line = f.readline()
        if not line:
            raise ValueError('Unexpected EOF while reading PLY header.')
        line = line.strip()
        if line == 'end_header':
            break
        if not line or line.startswith('comment'):
            continue
        tokens = line.split()
        if tokens[0] == 'element':
            current_element = tokens[1]
            if current_element == 'vertex':
                vertex_count = int(tokens[2])
        elif tokens[0] == 'property' and current_element == 'vertex':
            # property <type> <name>
            if len(tokens) < 3:
                raise ValueError(f'Malformed property line: {line}')
            vertex_props.append(tokens[-1])

    if vertex_count is None or not vertex_props:
        raise ValueError('PLY header missing vertex definition.')

    return {
        'vertex_count': vertex_count,
        'vertex_props': vertex_props,
    }


def _to_numpy(arr, dtype):
    if hasattr(arr, 'detach') and hasattr(arr, 'cpu'):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    return arr.astype(dtype, copy=False)


def _to_uint8(value) -> np.uint8:
    return np.uint8(max(0, min(255, int(float(value)))))


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

