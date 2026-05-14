#!/usr/bin/env python3
"""
使用官方 SAM 2 API 的内窥镜视频组织分割工具（优化版）：
1. 通过 SAM2VideoPredictor 对单目内窥镜视频执行多对象分割与跟踪
2. 支持 vos_optimized=True 和 torch.compile 优化，极大加速 VOS 推理
3. 使用 StableObjectTracker 确保每个组织/器官在整个视频中保持稳定的 ID/name
4. 导出与 NR-SLAM 兼容的 JSON 掩码格式，支持为每个组织设置不同参数
5. 通过 SAM2ImagePredictor 对单张图像进行分割
6. 自动生成 prompt（也可回退到中心或网格提示）以初始化视频跟踪状态
7. 可选保存可视化 PNG 方便调试

参考：https://github.com/facebookresearch/sam2
"""

import argparse
import contextlib
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

try:
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    # 新增导入以解决CUDAGraph问题
    from sam2.modeling.memory_attention import MemoryAttention
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    build_sam2 = None  # type: ignore
    build_sam2_video_predictor = None  # type: ignore
    SAM2ImagePredictor = None  # type: ignore
    MemoryAttention = None  # type: ignore
    print("[WARNING] sam2 not available. Install with: pip install git+https://github.com/facebookresearch/sam2.git")


COLOR_PALETTE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
    (255, 128, 0),
    (0, 128, 255),
    (128, 255, 0),
]


# 新增：修复MemoryAttention的forward方法以解决CUDAGraph问题
def patch_memory_attention():
    if MemoryAttention is None:
        return
        
    original_forward = MemoryAttention.forward
    
    def patched_forward(self, *args, **kwargs):
        # 在每次前向传播前标记CUDAGraph步骤开始
        # torch.compiler.cudagraph_mark_step_begin()
        # 调用原始forward方法
        result = original_forward(self, *args, **kwargs)
        # 处理transpose操作可能导致的张量覆盖问题
        if hasattr(result, 'transpose'):
            return result.transpose(0, 1).clone()
        return result
    
    # 应用补丁
    MemoryAttention.forward = patched_forward


class StableObjectTracker:
    """
    稳定的对象跟踪器，确保在整个视频中每个组织/器官的 ID/name 保持一致。
    将 SAM2 的 object_ids 映射到稳定的 organ_X 名称，用于 NR-SLAM 参数设置。
    """

    def __init__(self):
        self.sam2_id_to_stable_name: Dict[int, str] = {}
        self.next_organ_id = 0
        self.stable_names: List[str] = []
    
    def get_or_create_name(self, sam2_object_id: int) -> str:
        """
        获取或创建稳定的对象名称。
        如果 object_id 已存在，返回对应的稳定名称；否则创建新名称。
        确保同一组织/器官在整个视频中保持相同的名称。
        """
        if sam2_object_id not in self.sam2_id_to_stable_name:
            stable_name = f"organ_{self.next_organ_id}"
            self.sam2_id_to_stable_name[sam2_object_id] = stable_name
            self.stable_names.append(stable_name)
            self.next_organ_id += 1
            print(f"[INFO] Mapped SAM2 object_id {sam2_object_id} to stable name: {stable_name}")
        return self.sam2_id_to_stable_name[sam2_object_id]
    
    def get_all_tracked_names(self) -> List[str]:
        """获取所有已跟踪的稳定名称列表。"""
        return self.stable_names.copy()


class MaskTracker:
    """简易掩码跟踪器，仅用于单帧 auto 模式保持 organ_X 标签。"""

    def __init__(self, iou_thresh: float = 0.5, max_age: int = 20):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_label = 0
        self.tracks: Dict[str, Dict] = {}

    @staticmethod
    def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        return 0.0 if union == 0 else inter / union

    def assign(self, mask_entries: List[Dict], frame_idx: int) -> List[Dict]:
        assignments = []
        for entry in mask_entries:
            mask_bool = entry["mask_bool"]
            best_label = None
            best_iou = 0.0
            for label, track in self.tracks.items():
                if track["last_frame"] < frame_idx - self.max_age:
                    continue
                iou = self._mask_iou(mask_bool, track["mask_bool"])
                if iou > best_iou:
                    best_iou = iou
                    best_label = label

            if best_label is not None and best_iou >= self.iou_thresh:
                label = best_label
                color = self.tracks[label]["color"]
            else:
                label = f"organ_{self.next_label}"
                color = COLOR_PALETTE[self.next_label % len(COLOR_PALETTE)]
                self.next_label += 1

            self.tracks[label] = {
                "mask_bool": mask_bool,
                "color": color,
                "last_frame": frame_idx,
                "score": float(entry.get("score", 0.0)),
                "area": int(entry.get("area", 0)),
            }

            assignments.append({
                "name": label,
                "mask": mask_bool.astype(np.uint8).flatten().tolist(),
                "score": float(entry.get("score", 0.0)),
                "area": int(entry.get("area", 0)),
                "color": list(color),
            })

        stale = [label for label, track in self.tracks.items()
                 if track["last_frame"] < frame_idx - self.max_age]
        for label in stale:
            del self.tracks[label]

        return assignments


def _sanitize_device(device: str) -> str:
    device = device.lower()
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available, falling back to cpu.")
        return "cpu"
    return device


def _open_video_capture(video_path: str) -> cv2.VideoCapture:
    """
    打开视频，优先默认后端；若失败，尝试 FFMPEG（可兼容 mp4/avi 等）。
    """
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        return cap
    cap.release()

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if cap.isOpened():
        print("[INFO] Fallback to FFMPEG backend for video decoding (mp4/avi).")
        return cap

    raise ValueError(f"Failed to open video: {video_path}. Please check codec/format.")


def _resolve_ckpt_path(ckpt_path: str) -> str:
    """解析 checkpoint 路径，支持相对路径和绝对路径。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(script_dir, ckpt_path)
    return ckpt_path


@contextlib.contextmanager
def torch_runtime(device: str, amp_dtype: str = "bfloat16"):
    """PyTorch 推理上下文管理器，支持选择混合精度 dtype。"""
    stack = contextlib.ExitStack()
    stack.enter_context(torch.inference_mode())
    if device.startswith("cuda") and torch.cuda.is_available():
        dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        stack.enter_context(torch.autocast(device_type="cuda", dtype=dtype))
    try:
        yield
    finally:
        stack.close()


def load_sam2_image_predictor(ckpt_path: str,
                              device: str = "cuda",
                              config_path: Optional[str] = None) -> "SAM2ImagePredictor":
    """
    加载 SAM2 图像预测器。
    
    根据官方 API，支持多种方式：
    1. SAM2ImagePredictor.from_pretrained() (如果使用 HuggingFace)
    2. build_sam2(config_file, checkpoint, device=None)
    3. build_sam2(variant, checkpoint, config_path=None, device=None)
    """
    if not SAM2_AVAILABLE:
        raise ImportError("sam2 is not available. Please install it from https://github.com/facebookresearch/sam2")
    if build_sam2 is None:
        raise ImportError("sam2 build utilities missing.")

    ckpt_path = _resolve_ckpt_path(ckpt_path)
    
    # 尝试使用 from_pretrained (如果可用)
    try:
        if hasattr(SAM2ImagePredictor, 'from_pretrained'):
            predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
            if hasattr(predictor, 'to'):
                predictor.to(device)
            return predictor
    except Exception:
        pass
    
    # 使用 build_sam2 构建模型
    model = None
    
    if config_path:
        if config_path.startswith("configs/"):
            try:
                model = build_sam2(config_path, ckpt_path, device=device)
                print(f"[INFO] Using package config path for image predictor: {config_path}")
            except Exception as e:
                print(f"[WARNING] Failed to use package config path directly: {e}")
                config_name = os.path.splitext(os.path.basename(config_path))[0]
                try:
                    model = build_sam2(config_name, ckpt_path, device=device)
                    print(f"[INFO] Using config name: {config_name}")
                except Exception:
                    raise RuntimeError(f"Failed to build SAM2 image predictor with config: {config_path}")
        else:
            if not os.path.isabs(config_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                local_config_path = os.path.join(script_dir, config_path)
            else:
                local_config_path = config_path
            
            if os.path.isfile(local_config_path):
                try:
                    model = build_sam2(local_config_path, ckpt_path, device=device)
                except Exception:
                    config_dir = os.path.dirname(local_config_path)
                    config_name = os.path.splitext(os.path.basename(local_config_path))[0]
                    try:
                        model = build_sam2(config_name, ckpt_path, config_path=config_dir, device=device)
                    except Exception:
                        model = build_sam2(local_config_path, ckpt_path, device=device)
            else:
                try:
                    model = build_sam2("sam2.1_hiera_large", ckpt_path, config_path=local_config_path, device=device)
                except Exception:
                    model = build_sam2("sam2.1_hiera_l", ckpt_path, config_path=local_config_path, device=device)
    else:
        try:
            model = build_sam2("sam2.1_hiera_large", ckpt_path, device=device)
        except Exception:
            try:
                model = build_sam2("sam2.1_hiera_l", ckpt_path, device=device)
            except Exception:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                for candidate in [
                    "sam2.1_hiera_l.yaml",
                    "sam2.1_hiera_large.yaml",
                    "sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
                ]:
                    full_cfg = os.path.join(script_dir, candidate)
                    if os.path.exists(full_cfg):
                        try:
                            model = build_sam2(full_cfg, ckpt_path, device=device)
                            print(f"[INFO] Found config file: {full_cfg}")
                            break
                        except Exception:
                            config_dir = os.path.dirname(full_cfg)
                            config_name = os.path.splitext(os.path.basename(full_cfg))[0]
                            try:
                                model = build_sam2(config_name, ckpt_path, config_path=config_dir, device=device)
                                print(f"[INFO] Found config file: {full_cfg}")
                                break
                            except Exception:
                                continue
                
                if model is None:
                    raise RuntimeError("Could not find SAM2 config file. Please specify --config")

    if model is None:
        raise RuntimeError("Failed to build SAM2 model")
    
    predictor = SAM2ImagePredictor(model)
    if hasattr(predictor, 'to'):
        predictor.to(device)
    return predictor


def load_sam2_video_predictor(ckpt_path: str,
                              device: str = "cuda",
                              config_path: Optional[str] = None,
                              vos_optimized: bool = True):
    """
    加载 SAM2 视频预测器。
    
    根据官方 API，build_sam2_video_predictor 的签名是：
    build_sam2_video_predictor(config_file, checkpoint, device=None)
    
    config_file 应该是相对于 sam2 包内部的路径，例如 "configs/sam2.1/sam2.1_hiera_l.yaml"
    或者可以是完整的文件路径。
    """
    if not SAM2_AVAILABLE or build_sam2_video_predictor is None:
        raise ImportError("build_sam2_video_predictor is unavailable in the installed sam2 package.")

    ckpt_path = _resolve_ckpt_path(ckpt_path)

    # 处理配置文件路径
    if config_path is None:
        cfg_source = "configs/sam2.1/sam2.1_hiera_l.yaml"
    elif config_path.startswith("configs/"):
        cfg_source = config_path
        print(f"[INFO] Using provided package config path: {cfg_source}")
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if not os.path.isabs(config_path):
            local_config_path = os.path.join(script_dir, config_path)
        else:
            local_config_path = config_path
        
        config_name = os.path.splitext(os.path.basename(config_path))[0].lower()
        
        if "sam2.1_hiera_l" in config_name or "hiera_large" in config_name or "hiera_l" in config_name:
            cfg_source = "configs/sam2.1/sam2.1_hiera_l.yaml"
        elif "sam2.1_hiera_s" in config_name or "hiera_small" in config_name:
            cfg_source = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif "sam2.1_hiera_t" in config_name or "hiera_tiny" in config_name:
            cfg_source = "configs/sam2.1/sam2.1_hiera_t.yaml"
        elif "sam2.1_hiera_b" in config_name or "hiera_base" in config_name or "hiera_b+" in config_name:
            cfg_source = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        elif "sam2_hiera_l" in config_name or "sam2_hiera_large" in config_name:
            cfg_source = "configs/sam2/sam2_hiera_l.yaml"
        elif "sam2_hiera_s" in config_name or "sam2_hiera_small" in config_name:
            cfg_source = "configs/sam2/sam2_hiera_s.yaml"
        elif "sam2_hiera_t" in config_name or "sam2_hiera_tiny" in config_name:
            cfg_source = "configs/sam2/sam2_hiera_t.yaml"
        elif "sam2_hiera_b" in config_name or "sam2_hiera_base" in config_name:
            cfg_source = "configs/sam2/sam2_hiera_b+.yaml"
        else:
            if "configs" in config_path:
                idx = config_path.find("configs")
                cfg_source = config_path[idx:]
            else:
                cfg_source = f"configs/sam2.1/{os.path.basename(config_path)}"
        
        if os.path.isfile(local_config_path):
            print(f"[INFO] Found local config file: {local_config_path}")
            print(f"[INFO] Inferred package config path: {cfg_source}")

    # 构建预测器，支持 vos_optimized 优化
    try:
        import inspect
        sig = inspect.signature(build_sam2_video_predictor)
        kwargs = {}
        
        if "vos_optimized" in sig.parameters:
            kwargs["vos_optimized"] = vos_optimized
            if vos_optimized:
                print("[INFO] Enabling VOS optimization for faster inference")
        
        if "device" in sig.parameters:
            kwargs["device"] = device
        
        predictor = build_sam2_video_predictor(cfg_source, ckpt_path, **kwargs)
        print(f"[INFO] Using package config: {cfg_source}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to build SAM2 video predictor with config: {cfg_source}\n"
            f"Error: {e}\n"
            f"Please ensure the SAM2 package is properly installed and contains the config file."
        )
    
    return predictor


def _normalize_tensor_mask(mask) -> np.ndarray:
    """将各种格式的 mask 转换为 numpy 数组。"""
    if isinstance(mask, np.ndarray):
        return mask
    if hasattr(mask, "detach"):
        return mask.detach().cpu().numpy()
    if hasattr(mask, "cpu"):
        return mask.cpu().numpy()
    return np.asarray(mask)


def segment_frame_prompt_sam2(image_bgr: np.ndarray,
                              predictor: "SAM2ImagePredictor",
                              device: str = "cuda") -> Dict:
    """
    使用中心点 prompt 分割单帧图像。
    返回 soft_mask 和 hard_mask 格式，供 NR-SLAM 使用。
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    H, W = img_rgb.shape[:2]
    
    # 使用中心点作为 prompt
    input_point = np.array([[W // 2, H // 2]], dtype=np.float32)
    input_label = np.array([1], dtype=np.int32)
    
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    if scores is None or len(scores) == 0:
        raise RuntimeError("SAM2 predictor returned no masks.")
    
    # 选择得分最高的 mask
    best_idx = int(np.argmax(scores))
    mask = masks[best_idx] > 0.5
    
    # 转换为列表格式（NR-SLAM 期望的格式）
    soft_mask = mask.astype(np.uint8).flatten().tolist()
    hard_mask = (~mask).astype(np.uint8).flatten().tolist()
    
    return {"soft_mask": soft_mask, "hard_mask": hard_mask}


def segment_frame_auto_sam2(image_bgr: np.ndarray,
                            predictor: "SAM2ImagePredictor",
                            top_k: int = 5,
                            min_area: int = 0,
                            grid_size: int = 3) -> List[Dict]:
    """
    使用网格采样自动分割单帧图像。
    返回 mask_entries 列表，每个包含 mask_bool, score, area。
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    H, W = img_rgb.shape[:2]
    
    # 生成网格采样点
    grid_size = max(2, grid_size)
    ys = np.linspace(H * 0.2, H * 0.8, grid_size, dtype=np.float32)
    xs = np.linspace(W * 0.2, W * 0.8, grid_size, dtype=np.float32)
    points = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    labels = np.ones(points.shape[0], dtype=np.int32)

    all_masks = []
    all_scores = []
    
    for p, l in zip(points, labels):
        masks, scores, _ = predictor.predict(
            point_coords=p[None, :],
            point_labels=np.array([l], dtype=np.int32),
            multimask_output=True,
        )
        if scores is None or len(scores) == 0:
            continue
        for m, s in zip(masks, scores):
            all_masks.append(m)
            all_scores.append(float(s))

    if not all_masks:
        return []

    # 若 top_k <= 0，则自动选择：先按得分筛阈值，再最多保留 20 个
    all_masks = np.stack(all_masks, axis=0)
    all_scores = np.array(all_scores, dtype=np.float32)
    order = np.argsort(-all_scores)
    
    selected = []
    if top_k <= 0:
        # 自动：先保留分数 >= 0.5 的掩码，最多 20 个
        order = [i for i in order if all_scores[i] >= 0.5][:20]
    else:
        order = order[:top_k]
    
    for idx in order:
        mask = all_masks[idx] > 0.5
        area = int(mask.sum())
        if min_area > 0 and area < min_area:
            continue
        selected.append({
            "mask_bool": mask,
            "score": float(all_scores[idx]),
            "area": area,
        })
    
    return selected


def save_visualization(image_bgr: np.ndarray,
                       frame_data: Dict,
                       vis_path: str,
                       mode: str = "prompt"):
    """保存可视化图像，并生成索引图与图例 (legend JSON) 供交互/映射使用。"""
    H, W = image_bgr.shape[:2]
    overlay = image_bgr.copy()

    # 单通道索引图，每个像素存储 label index（0 表示背景）
    label_img = np.zeros((H, W), dtype=np.uint16)
    legend = []

    def _save_outputs(base_vis_path, overlay_img, label_img_arr, legend_list):
        os.makedirs(os.path.dirname(base_vis_path), exist_ok=True)
        # 主可视化
        try:
            cv2.imwrite(base_vis_path, overlay_img)
        except Exception as e:
            print(f"[WARNING] Failed to save visualization to {base_vis_path}: {e}")
        # 索引图（保存为 PNG 的低位表示，亦可保存为 .npy）
        label_path = os.path.splitext(base_vis_path)[0] + "_labels.png"
        try:
            # 将索引压缩到 0-255 范围作为快速查看（若索引过大，用户应用 np.save 保存原始 label_img）
            cv2.imwrite(label_path, (label_img_arr % 256).astype(np.uint8))
        except Exception as e:
            print(f"[WARNING] Failed to save label image to {label_path}: {e}")
        # 原始索引数据保存为 .npy（精确映射）
        npy_path = os.path.splitext(base_vis_path)[0] + "_labels.npy"
        try:
            np.save(npy_path, label_img_arr)
        except Exception as e:
            print(f"[WARNING] Failed to save label npy to {npy_path}: {e}")
        # 图例 JSON
        legend_path = os.path.splitext(base_vis_path)[0] + "_legend.json"
        try:
            with open(legend_path, "w") as f:
                json.dump(legend_list, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to save legend to {legend_path}: {e}")

    if mode == "auto" and "classes" in frame_data:
        for idx, cls in enumerate(frame_data["classes"], start=1):
            mask_flat = cls.get("mask", [])
            if len(mask_flat) != H * W:
                # 跳过不匹配的掩码
                continue
            mask = np.array(mask_flat, dtype=np.uint8).reshape(H, W)
            color = tuple(int(c) for c in cls.get("color", [0, 0, 255]))
            name = cls.get("name", f"obj_{idx}")

            # 半透明叠加
            colored = np.zeros_like(image_bgr)
            colored[mask > 0] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored, 0.6, 0)

            # 轮廓描边
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, thickness=2)

            # 在质心处写 name（白底黑字描边增强可读性）
            ys, xs = np.nonzero(mask)
            if xs.size:
                cx, cy = int(xs.mean()), int(ys.mean())
                # 白色外描，黑色细字
                cv2.putText(overlay, name, (max(5, cx - 10), max(15, cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(overlay, name, (max(5, cx - 10), max(15, cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # 写入索引图
            label_img[mask > 0] = idx

            legend.append({"index": idx, "name": name, "color": list(color), "sam2_object_id": int(cls.get("sam2_object_id", -1))})
    else:
        # 处理 prompt 模式（soft/hard masks）
        soft_flat = frame_data.get("soft_mask", [])
        hard_flat = frame_data.get("hard_mask", [])
        total = H * W
        if len(soft_flat) == total and len(hard_flat) == total:
            soft = np.array(soft_flat, dtype=np.uint8).reshape(H, W)
            hard = np.array(hard_flat, dtype=np.uint8).reshape(H, W)
            # 软体：红，硬体：绿（保持与原实现兼容）
            color_soft = (0, 0, 255)
            color_hard = (0, 255, 0)
            colored = np.zeros_like(image_bgr)
            colored[soft > 0] = color_soft
            colored[(hard > 0) & (soft == 0)] = color_hard
            overlay = cv2.addWeighted(overlay, 1.0, colored, 0.6, 0)
            # 简单标注
            legend.append({"index": 1, "name": "soft_mask", "color": list(color_soft)})
            legend.append({"index": 2, "name": "hard_mask", "color": list(color_hard)})
            # label_img: soft->1, hard-only->2
            label_img[soft > 0] = 1
            label_img[(hard > 0) & (soft == 0)] = 2

    # 保存所有输出（主可视化、索引图、npy、legend）
    _save_outputs(vis_path, overlay, label_img, legend)
    print(f"[INFO] Saved visualization, label image (.png/.npy) and legend for {vis_path}")


def _load_video_frames(video_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """加载视频帧。"""
    cap = _open_video_capture(video_path)
    
    frames_bgr: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
    cap.release()
    
    if not frames_bgr:
        raise ValueError("No frames found in video.")
    
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    return frames_bgr, frames_rgb


def _centroids_from_masks(mask_entries: Sequence[Dict],
                          H: int,
                          W: int) -> np.ndarray:
    """从 mask_entries 中提取质心作为 prompt 点。"""
    pts = []
    for entry in mask_entries:
        m = entry["mask_bool"]
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        pts.append([float(xs.mean()), float(ys.mean())])
    
    if not pts:
        pts = [[W / 2.0, H / 2.0]]
    
    return np.array(pts, dtype=np.float32)


def _build_prompts(first_frame: np.ndarray,
                   prompt_strategy: str,
                   auto_predictor: Optional["SAM2ImagePredictor"],
                   auto_topk: int,
                   min_mask_area: int,
                   grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建视频初始化的 prompts。
    返回 (point_coords, point_labels) 格式，符合 SAM2 API 要求
    """
    H, W = first_frame.shape[:2]
    prompt_strategy = prompt_strategy.lower()
    points: Optional[np.ndarray] = None
    
    if prompt_strategy == "auto" and auto_predictor is not None:
        mask_entries = segment_frame_auto_sam2(first_frame,
                                               auto_predictor,
                                               top_k=auto_topk,
                                               min_area=min_mask_area,
                                               grid_size=grid_size)
        if mask_entries:
            points = _centroids_from_masks(mask_entries, H, W)
    
    if points is None:
        if prompt_strategy == "grid":
            grid_size = max(2, grid_size)
            ys = np.linspace(H * 0.2, H * 0.8, grid_size, dtype=np.float32)
            xs = np.linspace(W * 0.2, W * 0.8, grid_size, dtype=np.float32)
            points = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        else:
            # 默认使用中心点
            points = np.array([[W / 2.0, H / 2.0]], dtype=np.float32)
    
    labels = np.ones(points.shape[0], dtype=np.int32)
    return points, labels


def _classes_from_masks_with_tracker(obj_ids: Sequence[int],
                                     masks: Sequence[np.ndarray],
                                     tracker: StableObjectTracker,
                                     min_area: int) -> List[Dict]:
    """
    将 SAM2 视频预测器返回的 object_ids 和 masks 转换为 classes 格式。
    使用 StableObjectTracker 确保每个对象的名称在整个视频中保持一致。
    """
    classes = []
    for oid, mask in zip(obj_ids, masks):
        # 将 oid 规范化为标量
        if isinstance(oid, (list, tuple, np.ndarray)):
            if len(oid) == 0:
                continue
            oid = oid[0]
        if hasattr(oid, "item"):
            oid = oid.item()

        # 规范化 mask
        if isinstance(mask, (list, tuple)) and len(mask) > 0:
            mask = mask[0]

        mask_np = _normalize_tensor_mask(mask)
        mask_bool = mask_np > 0.5
        area = int(mask_bool.sum())
        
        if min_area > 0 and area < min_area:
            continue
        
        stable_name = tracker.get_or_create_name(int(oid))
        name_idx = tracker.stable_names.index(stable_name) if stable_name in tracker.stable_names else int(oid)
        color = COLOR_PALETTE[name_idx % len(COLOR_PALETTE)]
        
        classes.append({
            "name": stable_name,
            "mask": mask_bool.astype(np.uint8).flatten().tolist(),
            "score": 1.0,
            "area": area,
            "color": list(color),
            "sam2_object_id": int(oid),
        })
    
    return classes


def process_video(video_path: str,
                  output_json: str,
                  sam2_ckpt: str,
                  device: str = "cuda",
                  vis_dir: Optional[str] = None,
                  auto_topk: int = 5,
                  min_mask_area: int = 0,
                  config_path: Optional[str] = None,
                  prompt_strategy: str = "auto",
                  grid_size: int = 3,
                  vos_optimized: bool = True,
                  use_compile: bool = False,
                  amp_dtype: str = "bfloat16",
                  empty_cache_interval: int = 0):
    """
    处理内窥镜视频，使用 SAM2 视频预测器进行稳定的多对象跟踪。
    """
    if not SAM2_AVAILABLE:
        raise ImportError("sam2 not available. Install with: pip install git+https://github.com/facebookresearch/sam2.git")

    print(f"[INFO] Processing video: {video_path}")
    print(f"[INFO] VOS optimized: {vos_optimized}, torch.compile: {use_compile}, amp_dtype: {amp_dtype}")
    # 避免 CUDAGraph 覆盖问题，若开启 vos_optimized 则禁用 torch.compile
    if vos_optimized and use_compile:
        print("[WARNING] Detected vos_optimized + torch.compile may trigger CUDAGraph overwrite; disabling torch.compile for stability.")
        use_compile = False

    # 加载第一帧用于生成 prompts
    cap = _open_video_capture(video_path)
    ret, first_frame_bgr = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError("No frames found in video.")
    
    H, W = first_frame_bgr.shape[:2]
    print(f"[INFO] Video resolution: {W}x{H}")
    
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    device = _sanitize_device(device)
    
    # 加载视频预测器
    video_predictor = load_sam2_video_predictor(
        sam2_ckpt, 
        device=device, 
        config_path=config_path,
        vos_optimized=vos_optimized
    )
    
    # 应用 torch.compile 优化
    if use_compile and hasattr(torch, 'compile'):
        print("[INFO] Applying torch.compile optimization...")
        if hasattr(video_predictor, 'model'):
            video_predictor.model = torch.compile(video_predictor.model)
        elif hasattr(video_predictor, '_model'):
            video_predictor._model = torch.compile(video_predictor._model)
    
    # 加载图像预测器用于自动生成 prompts
    auto_predictor = None
    if prompt_strategy == "auto":
        auto_predictor = load_sam2_image_predictor(sam2_ckpt, device=device, config_path=config_path)
        if use_compile and hasattr(torch, 'compile') and hasattr(auto_predictor, 'model'):
            auto_predictor.model = torch.compile(auto_predictor.model)

    # 构建初始 prompts（返回点坐标和标签）
    points, labels = _build_prompts(first_frame_bgr,
                                   prompt_strategy,
                                   auto_predictor,
                                   auto_topk,
                                   min_mask_area,
                                   grid_size)
    
    print(f"[INFO] Initial prompts: {len(points)} points")
    
    # 创建稳定对象跟踪器
    object_tracker = StableObjectTracker()
    
    # 按帧保存模式：创建输出目录
    output_dir = os.path.dirname(output_json)
    output_basename = os.path.basename(output_json)
    if output_basename.endswith('.json'):
        output_basename = output_basename[:-5]  # 移除 .json 后缀
    frames_dir = os.path.join(output_dir, output_basename + "_frames")
    os.makedirs(frames_dir, exist_ok=True)
    print(f"[INFO] 按帧保存模式：每帧数据将保存到 {frames_dir}/frame_*.json")
    
    frame_masks: Dict[str, Dict] = {}
    total_frames_processed = 0

    # 视频跟踪主流程
    with torch_runtime(device, amp_dtype=amp_dtype):
        print("[INFO] Initializing video state...")
        state = video_predictor.init_state(video_path)
        print("[INFO] Video state initialized successfully!")
        
        # 添加初始 prompts（匹配官方 API 参数顺序）
        print("[INFO] Adding initial prompts...")
        frame_idx = 0  # 从第一帧开始
        # 为每个 prompt 创建独立的初始对象，避免全部合并为同一个 obj_id
        obj_ids = []
        masks = []
        for i in range(len(points)):
            try:
                _, oid, mask = video_predictor.add_new_points_or_box(
                    state,
                    frame_idx=frame_idx,
                    obj_id=i,  # 每个点对应一个新 object
                    points=points[i:i+1],
                    labels=labels[i:i+1],
                    normalize_coords=True
                )
                # 将返回的 oid 规范化为标量
                oid_scalar = oid
                if isinstance(oid_scalar, (list, tuple, np.ndarray)) and len(oid_scalar) > 0:
                    oid_scalar = oid_scalar[0]
                if hasattr(oid_scalar, "item"):
                    oid_scalar = oid_scalar.item()
                obj_ids.append(oid_scalar)
                
                # 将返回的 mask 规范化为单个掩码
                mask_entry = mask
                if isinstance(mask_entry, (list, tuple)) and len(mask_entry) > 0:
                    mask_entry = mask_entry[0]
                masks.append(mask_entry)
            except Exception as e_point:
                print(f"[WARNING] 点 {i} 添加失败: {e_point}")
                continue
        
        if not obj_ids:
            raise RuntimeError("所有 prompts 添加失败，无法初始化跟踪")
        
        # 处理第一帧结果
        initial_classes = _classes_from_masks_with_tracker(obj_ids, masks, object_tracker, min_mask_area)
        frame_data = {
            "classes": initial_classes,
            "image_width": W,
            "image_height": H,
        }
        frame_key = f"frame_{frame_idx}"
        frame_masks[frame_key] = frame_data
        
        # 立即保存第一帧数据
        frame_file = os.path.join(frames_dir, f"{frame_key}.json")
        with open(frame_file, "w") as f:
            json.dump({frame_key: frame_data}, f, indent=2)
        total_frames_processed += 1
        
        if vis_dir:
            vis_path = os.path.join(vis_dir, f"{frame_key}.png")
            save_visualization(first_frame_bgr, frame_data, vis_path, mode="auto")
        
        print(f"[INFO] Frame {frame_idx}: 初始化 {len(initial_classes)} 个跟踪对象")
        print(f"[INFO] 跟踪对象名称: {[c['name'] for c in initial_classes]}")
        print(f"[INFO] 已保存到 {frame_file}")

        # 传播到后续帧
        print("[INFO] 开始在视频中传播掩码...")
        # 获取传播生成器
        propagation_generator = video_predictor.propagate_in_video(state)
        mark_step_warned = False
        
        while True:
            try:
                # 每次迭代前标记 CUDAGraph 步骤开始，避免缓存被覆盖
                if vos_optimized and hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    try:
                        torch.compiler.cudagraph_mark_step_begin()
                    except Exception as e_mark:
                        # 不影响主流程，仅提示一次
                        if not mark_step_warned:
                            print(f"[DEBUG] cudagraph_mark_step_begin skipped: {e_mark}")
                            mark_step_warned = True
                # 获取下一帧结果
                fidx, obj_ids_step, masks_step = next(propagation_generator)
                
                classes = _classes_from_masks_with_tracker(obj_ids_step, masks_step, object_tracker, min_mask_area)
                frame_data = {
                    "classes": classes,
                    "image_width": W,
                    "image_height": H,
                }
                frame_key = f"frame_{fidx}"
                frame_masks[frame_key] = frame_data
                
                # 立即保存当前帧数据到单独文件
                frame_file = os.path.join(frames_dir, f"{frame_key}.json")
                with open(frame_file, "w") as f:
                    json.dump({frame_key: frame_data}, f, indent=2)
                total_frames_processed += 1
                
                if vis_dir:
                    vis_path = os.path.join(vis_dir, f"{frame_key}.png")
                    cap_vis = cv2.VideoCapture(video_path)
                    cap_vis.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                    ret_vis, frame_vis = cap_vis.read()
                    cap_vis.release()
                    if ret_vis:
                        save_visualization(frame_vis, frame_data, vis_path, mode="auto")
                
                if empty_cache_interval > 0 and torch.cuda.is_available() and (fidx + 1) % empty_cache_interval == 0:
                    torch.cuda.empty_cache()
                
                if (fidx + 1) % 10 == 0:
                    print(f"[INFO] Frame {fidx}: {len(classes)} 个跟踪对象，已保存 {total_frames_processed} 帧")
                    
            except StopIteration:
                # 视频处理完成
                break
            except Exception as e:
                print(f"[ERROR] 处理帧时出错: {e}")
                break

    # 保存汇总信息（可选，用于兼容性）
    summary_file = os.path.join(frames_dir, "summary.json")
    summary_data = {
        "total_frames": total_frames_processed,
        "tracked_names": object_tracker.get_all_tracked_names(),
        "frames_dir": frames_dir,
        "frame_files": [f"frame_{i}.json" for i in range(total_frames_processed)]
    }
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    
    # 也保存一个完整的 JSON 文件（可选，用于向后兼容）
    if output_json:
        with open(output_json, "w") as f:
            json.dump(frame_masks, f, indent=2)
        print(f"[INFO] 完整 JSON 已保存至 {output_json} (用于向后兼容)")
    
    tracked_names = object_tracker.get_all_tracked_names()
    print(f"[INFO] 视频处理完成!")
    print(f"[INFO] 处理总帧数: {total_frames_processed}")
    print(f"[INFO] 稳定跟踪对象数: {len(tracked_names)}")
    print(f"[INFO] 对象名称: {tracked_names}")
    print(f"[INFO] 按帧保存目录: {frames_dir}")
    print(f"[INFO] 汇总信息: {summary_file}")


def process_image(image_path: str,
                  output_json: str,
                  sam2_ckpt: str,
                  device: str = "cuda",
                  vis_dir: Optional[str] = None,
                  mode: str = "prompt",
                  auto_topk: int = 5,
                  min_mask_area: int = 0,
                  config_path: Optional[str] = None,
                  grid_size: int = 3,
                  amp_dtype: str = "bfloat16"):
    """处理单张图像"""
    if not SAM2_AVAILABLE:
        raise ImportError("sam2 not available. Install with: pip install git+https://github.com/facebookresearch/sam2.git")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")

    device = _sanitize_device(device)
    predictor = load_sam2_image_predictor(sam2_ckpt, device=device, config_path=config_path)
    tracker = MaskTracker() if mode == "auto" else None

    try:
        if mode == "auto":
            mask_entries = segment_frame_auto_sam2(img,
                                                   predictor,
                                                   top_k=auto_topk,
                                                   min_area=min_mask_area,
                                                   grid_size=grid_size)
            classes = tracker.assign(mask_entries, frame_idx=0)
            frame_data = {"classes": classes}
        else:
            frame_data = segment_frame_prompt_sam2(img, predictor, device=device)
    except Exception as e:
        print(f"[ERROR] SAM2 图像分割失败: {e}")
        H, W = img.shape[:2]
        default_mask = np.ones((H * W,), dtype=np.uint8).tolist()
        frame_data = {"soft_mask": default_mask, "hard_mask": [0] * len(default_mask)}

    frame_data["image_width"] = img.shape[1]
    frame_data["image_height"] = img.shape[0]

    frame_masks = {"frame_0": frame_data}
    with open(output_json, "w") as f:
        json.dump(frame_masks, f, indent=2)
    
    print(f"[INFO] SAM2 掩码已保存至 {output_json}")

    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, os.path.basename(image_path))
        save_visualization(img, frame_data, vis_path, mode=mode)


def main():
    # 应用MemoryAttention补丁解决CUDAGraph问题
    patch_memory_attention()
    
    parser = argparse.ArgumentParser(description="基于官方 SAM 2 API 的组织分割脚本")
    parser.add_argument("--video", type=str, default=None, help="输入视频路径")
    parser.add_argument("--image", type=str, default=None, help="输入图像路径（与--video二选一）")
    parser.add_argument("--out", type=str, required=True, help="输出 JSON 路径")
    parser.add_argument("--ckpt", type=str, required=True, help="SAM2 模型权重路径（如 sam2.1_hiera_large.pt）")
    parser.add_argument("--config", type=str, default=None,
                        help="SAM2 配置文件路径，默认使用 configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备（cuda/cpu）")
    parser.add_argument("--vis_dir", type=str, default=None,
                        help="可视化图像保存目录（可选）")
    parser.add_argument("--mode", type=str, choices=["prompt", "auto"], default="prompt",
                        help="单图像处理模式（视频处理始终使用 auto 模式）")
    parser.add_argument("--auto_topk", type=int, default=0,
                        help="auto 模式下保留的 top-K 掩码数量，<=0 表示自动根据得分筛选（最多20个）")
    parser.add_argument("--min_mask_area", type=int, default=0,
                        help="掩码过滤的最小面积")
    parser.add_argument("--prompt_strategy", type=str, choices=["auto", "center", "grid"], default="auto",
                        help="视频初始化的 prompt 生成策略")
    parser.add_argument("--grid_size", type=int, default=3,
                        help="网格采样的大小（>=2）")
    parser.add_argument("--vos_optimized", action="store_true", default=True,
                        help="启用 VOS 优化加速推理（默认启用）")
    parser.add_argument("--no_vos_optimized", dest="vos_optimized", action="store_false",
                        help="禁用 VOS 优化")
    parser.add_argument("--use_compile", action="store_true", default=False,
                        help="使用 torch.compile 进一步优化（首次运行可能较慢）")
    parser.add_argument("--amp_dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16",
                        help="混合精度 dtype（bfloat16 更稳定，float16 更省显存）")
    parser.add_argument("--empty_cache_interval", type=int, default=0,
                        help="每隔 N 帧调用 torch.cuda.empty_cache()，0 表示不清理")
    args = parser.parse_args()

    if not SAM2_AVAILABLE:
        print("[ERROR] SAM2 未安装")
        print("安装方法: pip install 'git+https://github.com/facebookresearch/sam2.git'")
        return 1

    args.device = _sanitize_device(args.device)

    if args.video:
        process_video(args.video,
                      args.out,
                      args.ckpt,
                      device=args.device,
                      vis_dir=args.vis_dir,
                      auto_topk=args.auto_topk,
                      min_mask_area=args.min_mask_area,
                      config_path=args.config,
                      prompt_strategy=args.prompt_strategy,
                      grid_size=args.grid_size,
                      vos_optimized=args.vos_optimized,
                      use_compile=args.use_compile,
                      amp_dtype=args.amp_dtype,
                      empty_cache_interval=args.empty_cache_interval)
    elif args.image:
        process_image(args.image,
                      args.out,
                      args.ckpt,
                      device=args.device,
                      vis_dir=args.vis_dir,
                      mode=args.mode,
                      auto_topk=args.auto_topk,
                      min_mask_area=args.min_mask_area,
                      config_path=args.config,
                      grid_size=args.grid_size,
                      amp_dtype=args.amp_dtype)
    else:
        parser.error("必须提供 --video 或 --image 参数")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())