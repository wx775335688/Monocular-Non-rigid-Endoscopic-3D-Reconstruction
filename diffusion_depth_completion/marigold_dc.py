# Copyright 2024 Massimiliano Viola, Kevin Qu, Nando Metzger, Anton Obukhov ETH Zurich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold-DC#-citation
# More information can be found at https://marigolddepthcompletion.github.io
# ---------------------------------------------------------------------------------
# MODIFIED FOR NR-SLAM FUSION by GitHub Copilot
# Integrated with real data loader based on user-provided file structure.
# FIX: Correctly parse elastic_coefficient.txt which contains a single float.
# ---------------------------------------------------------------------------------
import logging
import os
import warnings
import argparse
import json

import cv2
import diffusers
import numpy as np
import torch
from diffusers import DDIMScheduler, MarigoldDepthPipeline
from PIL import Image
from typing import Tuple

warnings.simplefilter(action="ignore", category=FutureWarning)
diffusers.utils.logging.disable_progress_bar()


def load_nr_slam_frame_data(data_dir: str, frame_id_str: str) -> dict:
    """
    Loads all necessary NR-SLAM data for a single frame from the specified directory structure.
    """
    logging.info(f"Loading NR-SLAM data for frame {frame_id_str} from {data_dir}")

    def read_txt_with_count(path, dtype=np.float32, is_connections=False):
        """Helper for files with a count on the first line."""
        if not os.path.exists(path):
            logging.warning(f"File not found: {path}. Returning empty array.")
            return np.empty((0, 2) if is_connections else 0, dtype=dtype)
        with open(path, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0 or int(lines[0].strip()) == 0:
                return np.empty((0, 2) if is_connections else 0, dtype=dtype)
            if is_connections:
                return np.array([list(map(int, line.strip().split())) for line in lines[1:]], dtype=dtype)
            return np.array([line.strip() for line in lines[1:]], dtype=dtype)

    base_path = os.path.join(data_dir)

    # Paths
    cam_params_path = os.path.join(base_path, "camera_params", f"frame_{frame_id_str}_camera_matrix.xml")
    connections_path = os.path.join(base_path, "ddg_constraints", f"frame_{frame_id_str}_connections.txt")
    distances_path = os.path.join(base_path, "ddg_constraints", f"frame_{frame_id_str}_distances.txt")
    weights_path = os.path.join(base_path, "ddg_constraints", f"frame_{frame_id_str}_weights.txt")
    mappoint_pixel_path = os.path.join(base_path, "ddg_constraints", f"frame_{frame_id_str}_mappoint_to_pixel.txt")
    viscous_weights_path = os.path.join(base_path, "viscous_constraints", f"frame_{frame_id_str}_viscous_weights.txt")
    ssim_conf_path = os.path.join(base_path, "viscous_constraints", f"frame_{frame_id_str}_ssim_confidence.txt")
    elastic_coeff_path = os.path.join(base_path, "viscous_constraints", f"frame_{frame_id_str}_elastic_coefficient.txt")
    metadata_path = os.path.join(base_path, "metadata", f"frame_{frame_id_str}_metadata.json")

    # Load data
    fs = cv2.FileStorage(cam_params_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Can't open file: '{cam_params_path}' in read mode")
    camera_matrix = fs.getNode("camera_matrix").mat()
    fs.release()

    # --- FIX: Special handling for elastic_coefficient.txt ---
    elastic_coefficient = 1.0
    if os.path.exists(elastic_coeff_path):
        with open(elastic_coeff_path, 'r') as f:
            try:
                elastic_coefficient = float(f.readline().strip())
            except (ValueError, IndexError):
                logging.warning(f"Could not parse float from {elastic_coeff_path}. Using default 1.0.")
    else:
        logging.warning(f"File not found: {elastic_coeff_path}. Using default 1.0.")
    # ---------------------------------------------------------

    nr_slam_data = {
        'camera_matrix': camera_matrix,
        'connections': read_txt_with_count(connections_path, dtype=np.int32, is_connections=True),
        'initial_distances': read_txt_with_count(distances_path),
        'connection_weights': read_txt_with_count(weights_path),
        'mappoint_to_pixel': read_txt_with_count(mappoint_pixel_path, dtype=np.int32, is_connections=True),
        'viscous_weights': read_txt_with_count(viscous_weights_path),
        'ssim_confidence': read_txt_with_count(ssim_conf_path),
        'elastic_coefficient': elastic_coefficient,
    }

    # Load metadata for tracking stats
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            num_tracked = metadata.get("tracking_stats", {}).get("num_tracked_features", 0)
            nr_slam_data['tracking_success_rate'] = min(1.0, num_tracked / 200.0)
    else:
        nr_slam_data['tracking_success_rate'] = 0.9

    return nr_slam_data


def densify_sparse_depth_map(sparse_depth: np.ndarray, kernel_size: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalized convolution to inpaint sparse measurements. Returns a pseudo-dense seed map
    and a confidence map in [0, 1].
    """
    if sparse_depth.ndim != 2:
        raise ValueError("Sparse depth must be a 2D array")

    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    valid_mask = (sparse_depth > 1e-6).astype(np.float32)
    weighted_depth = sparse_depth.astype(np.float32) * valid_mask

    blurred_depth = cv2.GaussianBlur(weighted_depth, (kernel_size, kernel_size), 0)
    blurred_mask = cv2.GaussianBlur(valid_mask, (kernel_size, kernel_size), 0)

    densified = blurred_depth / (blurred_mask + 1e-6)
    densified[valid_mask > 0] = sparse_depth[valid_mask > 0]

    confidence = np.clip(blurred_mask / (blurred_mask.max() + 1e-6), 0.0, 1.0)
    return densified.astype(np.float32), confidence.astype(np.float32)


class MarigoldDepthCompletionPipeline(MarigoldDepthPipeline):
    """
    Pipeline for Marigold Depth Completion.
    Extends the MarigoldDepthPipeline to include depth completion functionality.
    MODIFIED to incorporate geometric and temporal constraints from NR-SLAM.
    """

    def _get_dynamic_weights(self, tracking_success_rate: float):
        if tracking_success_rate >= 0.8:
            return {"lambda_geo": 0.35, "lambda_visc": 0.2, "lambda_elas": 0.35, "lambda_seed": 0.15,
                    "lambda_prior": 0.08}
        if 0.5 <= tracking_success_rate < 0.8:
            return {"lambda_geo": 0.15, "lambda_visc": 0.1, "lambda_elas": 0.25, "lambda_seed": 0.1,
                    "lambda_prior": 0.05}
        return {"lambda_geo": 0.05, "lambda_visc": 0.05, "lambda_elas": 0.15, "lambda_seed": 0.05,
                "lambda_prior": 0.03}

    def _get_depth_range(self, sparse_depth: np.ndarray) -> Tuple[float, float]:
        valid_depths = sparse_depth[sparse_depth > 1e-6]
        if valid_depths.size == 0:
            return 15.0, 120.0
        lower = float(np.percentile(valid_depths, 5))
        upper = float(np.percentile(valid_depths, 95))
        lower = max(10.0, lower * 0.8)
        upper = min(150.0, upper * 1.2)
        if upper - lower < 5.0:
            upper = lower + 5.0
        return lower, upper

    def _pixels_to_3d(self, pixels_uv, depth, camera_matrix):
        fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
        u, v = pixels_uv[:, 0], pixels_uv[:, 1]
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        return torch.stack([x, y, depth], dim=-1)

    def __call__(
            self,
            image: Image.Image,
            sparse_depth: np.ndarray,
            nr_slam_data: dict,
            num_inference_steps: int = 50,
            processing_resolution: int = 768,
            seed: int = 2024
    ) -> np.ndarray:
        device = self._execution_device
        generator = torch.Generator(device=device).manual_seed(seed)

        if not isinstance(sparse_depth, np.ndarray) or sparse_depth.ndim != 2:
            raise ValueError("Sparse depth should be a 2D numpy ndarray")

        with torch.no_grad():
            if self.empty_text_embedding is None:
                text_inputs = self.tokenizer("", padding="do_not_pad", max_length=self.tokenizer.model_max_length,
                                             truncation=True, return_tensors="pt")
                self.empty_text_embedding = self.text_encoder(text_inputs.input_ids.to(device))[0]

        image, padding, original_resolution = self.image_processor.preprocess(image,
                                                                              processing_resolution=processing_resolution,
                                                                              device=device, dtype=self.dtype)
        if sparse_depth.shape != original_resolution:
            raise ValueError(
                f"Sparse depth dimensions ({sparse_depth.shape}) must match image dimensions ({original_resolution})")

        with torch.no_grad():
            image_latent, pred_latent = self.prepare_latents(image, None, generator, 1, 1)
        del image

        sparse_depth_torch = torch.from_numpy(sparse_depth)[None, None].float().to(device)
        sparse_mask = sparse_depth_torch > 0
        logging.info(f"Using {sparse_mask.int().sum().item()} guidance points")

        H, W = original_resolution
        tracking_success_rate = nr_slam_data.get('tracking_success_rate', 0.9)
        lambdas = self._get_dynamic_weights(tracking_success_rate)
        cam_mat = torch.from_numpy(nr_slam_data['camera_matrix']).float().to(device)

        dense_seed_np, dense_conf_np = densify_sparse_depth_map(sparse_depth)
        dense_seed = torch.from_numpy(dense_seed_np)[None, None].float().to(device)
        dense_confidence = torch.from_numpy(dense_conf_np)[None, None].float().to(device)
        dense_seed_mask = dense_confidence > 0.1

        depth_min_val, depth_max_val = self._get_depth_range(sparse_depth)
        depth_min = torch.tensor(depth_min_val, device=device)
        depth_max = torch.tensor(depth_max_val, device=device)

        if nr_slam_data['mappoint_to_pixel'].size > 0:
            mappoint_tensor = torch.from_numpy(nr_slam_data['mappoint_to_pixel']).long().to(device)
            linear_idx = mappoint_tensor[:, 1]
            valid_linear_mask = (linear_idx >= 0) & (linear_idx < H * W)
            mappoint_tensor = mappoint_tensor[valid_linear_mask]
            linear_idx = linear_idx[valid_linear_mask]
            pixel_uv = torch.stack((linear_idx % W, linear_idx // W), dim=-1)
        else:
            mappoint_tensor = torch.empty((0, 2), dtype=torch.long, device=device)
            pixel_uv = torch.empty((0, 2), dtype=torch.long, device=device)

        valid_slam_points_mask = sparse_mask[
            0, 0, pixel_uv[:, 1], pixel_uv[:, 0]] if pixel_uv.numel() > 0 else torch.empty(0, dtype=torch.bool,
                                                                                           device=device)
        mp_ids = mappoint_tensor[:, 0][valid_slam_points_mask] if mappoint_tensor.numel() > 0 else torch.empty(0, dtype=torch.long, device=device)
        pixel_uv = pixel_uv[valid_slam_points_mask]
        mp_id_to_tensor_idx = {mid.item(): i for i, mid in enumerate(mp_ids)}

        connections = torch.from_numpy(nr_slam_data['connections']).to(device)
        conn_mask = torch.tensor(
            [(c[0].item() in mp_id_to_tensor_idx and c[1].item() in mp_id_to_tensor_idx) for c in connections],
            dtype=torch.bool, device=device) if connections.numel() > 0 else torch.empty(0, dtype=torch.bool,
                                                                                         device=device)
        connections_valid = connections[conn_mask]
        conn_idx_i = torch.tensor([mp_id_to_tensor_idx[c[0].item()] for c in connections_valid], device=device,
                                  dtype=torch.long) if connections_valid.numel() > 0 else torch.empty(0,
                                                                                                      dtype=torch.long,
                                                                                                      device=device)
        conn_idx_j = torch.tensor([mp_id_to_tensor_idx[c[1].item()] for c in connections_valid], device=device,
                                  dtype=torch.long) if connections_valid.numel() > 0 else torch.empty(0,
                                                                                                      dtype=torch.long,
                                                                                                      device=device)

        l0_ij = torch.from_numpy(nr_slam_data['initial_distances']).float().to(device)[conn_mask]
        w_ij = torch.from_numpy(nr_slam_data['connection_weights']).float().to(device)[conn_mask]

        # viscous_weights 可能按边或按点存储，按长度自适应
        visc_np = nr_slam_data['viscous_weights']
        visc_t = torch.from_numpy(visc_np).float().to(device)
        if visc_t.numel() == connections.shape[0]:
            b_ij_t = visc_t[conn_mask]
        elif visc_t.numel() == mappoint_tensor.shape[0]:
            visc_valid_pts = visc_t[valid_slam_points_mask]
            if conn_idx_i.numel() > 0:
                b_ij_t = 0.5 * (visc_valid_pts[conn_idx_i] + visc_valid_pts[conn_idx_j])
            else:
                b_ij_t = torch.empty(0, device=device)
        else:
            logging.warning(f"viscous_weights size {visc_t.numel()} mismatches connections {connections.shape[0]} and points {mappoint_tensor.shape[0]}; using zeros.")
            b_ij_t = torch.zeros(conn_idx_i.numel(), device=device)

        # ssim_confidence 可能是按点或全图，按长度处理
        ssim_np = nr_slam_data['ssim_confidence']
        ssim_t = torch.from_numpy(ssim_np).float().to(device)
        if ssim_t.numel() == mappoint_tensor.shape[0]:
            ssim_confidence = ssim_t[valid_slam_points_mask]
        elif ssim_t.numel() == sparse_mask.numel():
            ssim_confidence = ssim_t.view_as(sparse_mask)[sparse_mask]
        else:
            logging.warning(f"ssim_confidence size {ssim_t.numel()} mismatches expected; using ones.")
            ssim_confidence = torch.ones_like(d_i)
        k = nr_slam_data['elastic_coefficient']
        huber_loss = torch.nn.HuberLoss(reduction='none', delta=1.0)

        # In the original Marigold-DC, scale and shift are optimized.
        # Let's stick to their affine transformation logic.
        scale, shift = torch.nn.Parameter(torch.ones(1, device=device)), torch.nn.Parameter(
            torch.zeros(1, device=device))
        pred_latent = torch.nn.Parameter(pred_latent)

        if sparse_mask.any():
            sparse_valid = sparse_depth_torch[sparse_mask]
            sparse_lower = sparse_valid.min().item()
            sparse_upper = sparse_valid.max().item()
            sparse_range = max(sparse_upper - sparse_lower, 1e-6)  # 避免范围为0
        else:  # Handle case with no sparse depth
            sparse_range, sparse_lower = 1.0, 0.0

        optimizer = torch.optim.Adam([{"params": [scale, shift], "lr": 0.005}, {"params": [pred_latent], "lr": 0.05}])

        def affine_to_metric(depth: torch.Tensor) -> torch.Tensor:
            # Using the original paper's affine transform: a*d + b
            return scale * depth + shift

        def latent_to_metric(latent: torch.Tensor) -> torch.Tensor:
            affine_inv = self.decode_prediction(latent)
            # 使用更稳健的归一化，避免极端值的影响
            if affine_inv.numel() > 1:
                # 使用百分位数进行归一化，更稳健
                lower_bound = torch.quantile(affine_inv, 0.01)  # 1%分位数
                upper_bound = torch.quantile(affine_inv, 0.99)  # 99%分位数
                range_val = upper_bound - lower_bound
                if range_val > 1e-6:
                    affine_inv = (affine_inv - lower_bound) / range_val
                else:
                    affine_inv = torch.clamp(affine_inv, 0.0, 1.0)
            else:
                affine_inv = torch.clamp(affine_inv, 0.0, 1.0)

            # Map to the sparse depth range and then apply the learned affine transform 'a*d+b'
            prediction = affine_inv * sparse_range + sparse_lower
            prediction = affine_to_metric(prediction)

            prediction = self.image_processor.unpad_image(prediction, padding)
            return self.image_processor.resize_antialias(prediction, original_resolution, "bilinear", is_aa=False)

        def loss_l1l2(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            if input.numel() == 0: return torch.tensor(0.0, device=device)
            return torch.nn.functional.l1_loss(input, target) + torch.nn.functional.mse_loss(input, target)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        for _, t in enumerate(
                self.progress_bar(self.scheduler.timesteps, desc=f"Marigold-DC steps ({str(device)})...")):
            optimizer.zero_grad()
            batch_latent = torch.cat([image_latent, pred_latent], dim=1)
            noise = self.unet(batch_latent, t, encoder_hidden_states=self.empty_text_embedding, return_dict=False)[0]
            step_output = self.scheduler.step(noise, t, pred_latent, generator=generator)
            pred_original_sample = step_output.pred_original_sample
            current_metric_estimate = latent_to_metric(pred_original_sample)

            loss = loss_l1l2(current_metric_estimate[sparse_mask], sparse_depth_torch[sparse_mask])
            d_i = current_metric_estimate[
                0, 0, pixel_uv[:, 1], pixel_uv[:, 0]] if pixel_uv.numel() > 0 else torch.empty(0, device=device)

            if lambdas["lambda_geo"] > 0 and conn_idx_i.numel() > 0:
                X_i, X_j = self._pixels_to_3d(pixel_uv[conn_idx_i], d_i[conn_idx_i], cam_mat), self._pixels_to_3d(
                    pixel_uv[conn_idx_j], d_i[conn_idx_j], cam_mat)
                loss += lambdas["lambda_geo"] * (w_ij * huber_loss(torch.linalg.norm(X_i - X_j, dim=1), l0_ij)).mean()

            if lambdas["lambda_elas"] > 0 and d_i.numel() > 0:
                z_sparse_i = sparse_depth_torch[0, 0, pixel_uv[:, 1], pixel_uv[:, 0]]
                loss += lambdas["lambda_elas"] * (
                            ssim_confidence * k * torch.pow(d_i - z_sparse_i, 2) / (z_sparse_i + 1e-8)).mean()

            if lambdas["lambda_visc"] > 0 and conn_idx_i.numel() > 0:
                d_i_grads_x = current_metric_estimate[0, 0, pixel_uv[:, 1], pixel_uv[:, 0]] - current_metric_estimate[
                    0, 0, pixel_uv[:, 1], torch.clamp(pixel_uv[:, 0] - 1, 0, W - 1)]
                d_i_grads_y = current_metric_estimate[0, 0, pixel_uv[:, 1], pixel_uv[:, 0]] - current_metric_estimate[
                    0, 0, torch.clamp(pixel_uv[:, 1] - 1, 0, H - 1), pixel_uv[:, 0]]
                grad_d = torch.stack([d_i_grads_x, d_i_grads_y], dim=-1)
                grad_di, grad_dj = grad_d[conn_idx_i], grad_d[conn_idx_j]
                loss += lambdas["lambda_visc"] * (
                            b_ij_t * torch.pow(torch.linalg.norm(grad_di - grad_dj, dim=1), 2)).mean()

            if lambdas["lambda_seed"] > 0 and dense_seed_mask.any().item():
                seed_residual = torch.abs(current_metric_estimate[dense_seed_mask] - dense_seed[dense_seed_mask])
                loss += lambdas["lambda_seed"] * (dense_confidence[dense_seed_mask] * seed_residual).mean()

            if lambdas["lambda_prior"] > 0:
                below = torch.nn.functional.relu(depth_min - current_metric_estimate)
                above = torch.nn.functional.relu(current_metric_estimate - depth_max)
                loss += lambdas["lambda_prior"] * (below.mean() + above.mean())

            loss.backward()
            with torch.no_grad():
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                pred_epsilon = (alpha_prod_t ** 0.5) * noise + ((1 - alpha_prod_t) ** 0.5) * pred_latent
                pred_latent.grad *= torch.linalg.norm(pred_epsilon).item() / max(
                    torch.linalg.norm(pred_latent.grad).item(), 1e-8)
            optimizer.step()
            with torch.no_grad():
                pred_latent.data = self.scheduler.step(noise, t, pred_latent, generator=generator).prev_sample
            torch.cuda.empty_cache()

        with torch.no_grad():
            prediction = latent_to_metric(pred_latent.detach())
        return self.image_processor.pt_to_numpy(prediction).squeeze()


def main():
    parser = argparse.ArgumentParser(description="Marigold-DC Pipeline with NR-SLAM Fusion")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to the 'marigold_data' directory (legacy format).")
    parser.add_argument("--json_path", type=str, default=None, help="Path to JSON file with DDG/visc data (new format).")
    parser.add_argument("--in_image", type=str, required=True, help="Path to input RGB image.")
    parser.add_argument("--in_depth", type=str, required=True, help="Path to input sparse depth (.npy).")
    parser.add_argument("--K", type=float, nargs=4, default=None, help="Camera intrinsics: fx fy cx cy (if not in JSON).")
    parser.add_argument("--frame_id", type=int, default=0, help="Frame ID to process (e.g., 0, 1, 2).")
    parser.add_argument("--out_dir", type=str, default="output", help="Directory to save the dense depth output.")
    parser.add_argument("--out_depth", type=str, default=None, help="Output dense depth path (.npy).")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--processing_resolution", type=int, default=512, help="Denoising resolution")
    parser.add_argument("--checkpoint", type=str, default="prs-eth/marigold-depth-v1-0", help="Depth checkpoint")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Use frame_id directly as a string (e.g., 30 -> "30")
    frame_id_str = str(args.frame_id)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device == torch.device("cpu"):
        args.processing_resolution = min(args.processing_resolution, 512)
        args.num_inference_steps = min(args.num_inference_steps, 10)
        logging.warning(
            f"CUDA/MPS not found: Reducing resolution to {args.processing_resolution} and steps to {args.num_inference_steps}")

    pipe = MarigoldDepthCompletionPipeline.from_pretrained(args.checkpoint, prediction_type="depth").to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    if device == torch.device("cpu"):
        logging.warning("Using a lightweight VAE")
        del pipe.vae
        pipe.vae = diffusers.AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device)

    # 加载NR-SLAM数据（支持新旧两种格式）
    nr_slam_data = None
    if args.json_path:
        # 新格式：从JSON文件加载
        try:
            with open(args.json_path, 'r') as f:
                json_data = json.load(f)
            
            # 解析JSON数据
            ddg_edges = np.array(json_data.get('ddg', {}).get('edges', []))
            visc_points = np.array(json_data.get('visc', {}).get('points', []))
            camera_K = json_data.get('camera_K', [])
            track_rate = json_data.get('track_rate', 0.9)
            
            # 构建相机矩阵
            if len(camera_K) == 4:
                fx, fy, cx, cy = camera_K
            elif args.K:
                fx, fy, cx, cy = args.K
            else:
                raise ValueError("Camera intrinsics not found in JSON and --K not provided")
            
            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            
            # 转换为marigold_dc期望的格式
            connections = ddg_edges[:, :2].astype(np.int32) if ddg_edges.size > 0 else np.empty((0, 2), dtype=np.int32)
            initial_distances = ddg_edges[:, 2] if ddg_edges.size > 0 else np.empty(0, dtype=np.float32)
            connection_weights = ddg_edges[:, 3] if ddg_edges.size > 0 else np.empty(0, dtype=np.float32)
            
            # 构建mappoint_to_pixel映射（简化：假设按顺序）
            mappoint_to_pixel = np.array([[i, i] for i in range(len(visc_points))], dtype=np.int32) if visc_points.size > 0 else np.empty((0, 2), dtype=np.int32)
            
            viscous_weights = visc_points[:, 5] if visc_points.size > 0 else np.empty(0, dtype=np.float32)
            ssim_confidence = visc_points[:, 3] if visc_points.size > 0 else np.empty(0, dtype=np.float32)
            elastic_coefficient = float(visc_points[0, 4]) if visc_points.size > 0 else 1.0
            
            nr_slam_data = {
                'camera_matrix': camera_matrix,
                'connections': connections,
                'initial_distances': initial_distances,
                'connection_weights': connection_weights,
                'mappoint_to_pixel': mappoint_to_pixel,
                'viscous_weights': viscous_weights,
                'ssim_confidence': ssim_confidence,
                'elastic_coefficient': elastic_coefficient,
                'tracking_success_rate': track_rate
            }
            logging.info(f"Loaded JSON data from {args.json_path}: {len(connections)} connections, {len(visc_points)} points")
        except Exception as e:
            logging.error(f"Error loading JSON data: {e}")
            return
    elif args.data_dir:
        # 旧格式：从目录加载
        try:
            nr_slam_data = load_nr_slam_frame_data(args.data_dir, frame_id_str)
        except FileNotFoundError as e:
            logging.error(f"Error loading data: {e}. Make sure --data_dir and --frame_id are correct.")
            return
    else:
        logging.error("Either --json_path or --data_dir must be provided")
        return

    # 加载输入图像和深度
    in_image_path = args.in_image
    in_depth_path = args.in_depth
    
    if not os.path.exists(in_image_path) or not os.path.exists(in_depth_path):
        logging.error(f"Input image or sparse depth not found: {in_image_path} or {in_depth_path}")
        return

    pred = pipe(
        image=Image.open(in_image_path),
        sparse_depth=np.load(in_depth_path),
        nr_slam_data=nr_slam_data,
        num_inference_steps=args.num_inference_steps,
        processing_resolution=args.processing_resolution,
    )

    if args.out_depth:
        out_depth_path = args.out_depth
    else:
        out_depth_path = os.path.join(args.out_dir, f"frame_{frame_id_str}_dense.npy")
    out_vis_path = os.path.join(args.out_dir, f"frame_{frame_id_str}_dense_vis.jpg")

    np.save(out_depth_path, pred)
    vis = pipe.image_processor.visualize_depth(pred, val_min=pred.min(), val_max=pred.max())[0]
    vis.save(out_vis_path)
    print(f"Saved dense depth to {out_depth_path} and visualization to {out_vis_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()