/*
 * This file is part of NR-SLAM
 *
 * Copyright (C) 2022-2023 Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * NR-SLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "tracking.h"

#include "features/shi_tomasi.h"
#include "features/orb_feature.h"
#include "optimization/g2o_optimization.h"
#include "mapping/mapping.h"
#include "utilities/dbscan.h"
#include "utilities/geometry_toolbox.h"
#include "utilities/statistics_toolbox.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <numeric>
#include "absl/log/log.h"
#include "absl/log/check.h"
#include "absl/container/flat_hash_map.h"

namespace fs = std::filesystem;
 
 
 using namespace std;
 
Tracking::Tracking(const Tracking::Options options, std::shared_ptr<Map> map,
                   std::shared_ptr<CameraModel> calibration,
                   std::shared_ptr<StereoLucasKanade> stereo_matcher,
                   std::shared_ptr<ImageVisualizer> image_visualizer,
                   TimeProfiler* time_profiler, Mapping* mapper) :
        options_(options), map_(map), calibration_(calibration), stereo_matcher_(stereo_matcher),
        image_visualizer_(image_visualizer), tracking_status_(NOT_INITIALIZED), time_profiler_(time_profiler),
        mapper_(mapper), internal_frame_counter_(0), last_export_frame_id_(-1)
 {
    // 根据配置选择特征提取方法（Shi-Tomasi 或 ORB）
    if (options_.feature_type == "orb3" || options_.feature_type == "orb") {
        ORBFeature::Options orb_opts;
        orb_opts.n_features = options_.orb_n_features;
        orb_opts.scale_factor = options_.orb_scale_factor;
        orb_opts.n_levels = options_.orb_n_levels;
        feature_extractor_ = std::make_shared<ORBFeature>(orb_opts);
        LOG(INFO) << "Using ORB feature extractor with n_features=" << orb_opts.n_features;
    } else {
        ShiTomasi::Options shi_tomasi_options;
        shi_tomasi_options.non_max_suprresion_window_size = 7;
        feature_extractor_ = std::make_shared<ShiTomasi>(shi_tomasi_options);
        LOG(INFO) << "Using Shi-Tomasi feature extractor";
    }
 
     klt_tracker_ = LucasKanadeTracker(cv::Size(options_.klt_window_size, options_.klt_window_size),
                                       options_.klt_max_level, options_.klt_max_iters,
                                       options_.klt_epsilon, options_.klt_min_eig_th);
 
     current_frame_ = make_shared<Frame>();
 
     current_frame_->SetCalibration(calibration);
 
     MonocularMapInitializer::Options monocular_map_initializer_options;
     monocular_map_initializer_options.klt_window_size = 21;
     monocular_map_initializer_options.klt_max_level = 4;
     monocular_map_initializer_options.klt_max_iters = 10;
     monocular_map_initializer_options.klt_epsilon = 0.0001;
     monocular_map_initializer_options.klt_min_eig_th = 0.0001;
     monocular_map_initializer_options.klt_min_SSIM = 0.3;
 
     monocular_map_initializer_options.rigid_initializer_max_features = 4000;
    monocular_map_initializer_options.rigid_initializer_min_sample_set_size = 8;
    // 放宽单目初始化视差阈值与极线误差：原始 0.999 / 0.005 在内窥镜低基线场景过于严格
    // 将 min_parallax 从 0.999 降到 0.05（可在 0.01-0.1 之间调参）
    monocular_map_initializer_options.rigid_initializer_min_parallax = 0.05f;
    monocular_map_initializer_options.rigid_initializer_radians_per_pixel = options_.radians_per_pixel;
    // 将 epipolar_threshold 从 0.005 放宽到 0.01，允许更宽松的极线一致性
    monocular_map_initializer_options.rigid_initializer_epipolar_threshold = 0.01f;
 
     monocular_map_initializer_ = make_unique<MonocularMapInitializer>(
             monocular_map_initializer_options, feature_extractor_, calibration_, image_visualizer_);
     
     // 初始化路径（统一保存RGB图像到output/data/images/）
     point_cloud_and_pose_path_ = "output/data/images/";
     try {
         fs::create_directories(point_cloud_and_pose_path_);
     } catch (const fs::filesystem_error& e) {
         LOG(WARNING) << "Failed to create images directory: " << e.what();
     }

     // 初始化跟踪统计文件路径
     tracking_stats_path_ = "output/data/tracking_stats.csv";
     try {
         // 如果文件已存在，先备份
         if (fs::exists(tracking_stats_path_)) {
             std::string backup_path = tracking_stats_path_ + ".backup";
             fs::copy(tracking_stats_path_, backup_path, fs::copy_options::overwrite_existing);
         }
         // 创建新的统计文件并写入表头
         std::ofstream stats_file(tracking_stats_path_);
         if (stats_file.is_open()) {
             stats_file << "frame_idx,total_keypoints,tracked_3d,tracked,just_triangulated,candidate_3d,bad,total_3d,point_change,3d_ratio\n";
             stats_file.close();
             LOG(INFO) << "Initialized tracking statistics file: " << tracking_stats_path_;
         }
     } catch (const std::exception& e) {
         LOG(WARNING) << "Failed to initialize tracking statistics file: " << e.what();
     }
     
     
     // 初始化Marigold数据导出器
     enable_marigold_export_ = options_.enable_marigold_export;
     if (enable_marigold_export_) {
         nrslam::MarigoldDataExporter::Options marigold_options;
         marigold_options.output_dir = options_.marigold_output_dir;
         marigold_exporter_ = std::make_unique<nrslam::MarigoldDataExporter>(marigold_options);
         LOG(INFO) << "Marigold data exporter initialized. Output directory: " << marigold_options.output_dir;
     }
 }
 
 
 void Tracking::TrackImage(const cv::Mat &im, const absl::flat_hash_map<std::string, cv::Mat>& masks,
                           const cv::Mat &additional_im, const cv::Mat& im_clahe,int frame_idx) {
     map_->SetAllMappointsToNonActive();
 
    // 安全获取全局mask
    cv::Mat global_mask = masks.count("Global") ? masks.at("Global") : cv::Mat();

     if (map_->IsEmpty()) {
        // 地图未初始化时执行初始化
        // 如果提供了右目图像（additional_im 非空），优先采用双目初始化以获得更可靠的度量深度；
        // 否则退化为单目初始化（只能恢复到尺度不定的深度，再通过 target_median_depth 估尺度）。
        if (!additional_im.empty()) {
            StereoMapInitialization(im, additional_im, global_mask, im_clahe);
        } else {
            MonocularMapInitialization(im, global_mask, im_clahe);
        }
     } else {
         // 将当前帧视野内的地图点添加到当前帧
         AddVisibleMapPointsToCurrentFrame(im);

         // 更新上一帧三角化的点
         UpdateTriangulatedPoints();

         // 执行正常跟踪流程
         absl::flat_hash_set<ID> lost_mappoint_ids = TrackCameraAndDeformation(im, global_mask, frame_idx);
 
        // 点重用
        PointReuse(im, global_mask, lost_mappoint_ids);

        // 检查是否需要补充新的特征点进行三角化
        int tracked_3d_count = current_frame_->GetKeypointsWithStatus({TRACKED_WITH_3D}).size();
        int tracked_count = current_frame_->GetKeypointsWithStatus({TRACKED}).size();

        // 如果3D点太少，或者TRACKED点太少（用于三角化），就补充新特征点
        if (tracked_3d_count < 20 || tracked_count < 30) {
            LOG(INFO) << "[特征补充] 3D点不足，正在补充新特征...";

            // 补充新的特征点用于三角化
            SupplementFeaturesForTriangulation(im, global_mask);
        }

       if (tracked_3d_count < 10) {
           LOG(WARNING) << "[警告] 3D点过少 (" << tracked_3d_count << ")，正在重检测特征...";

           // 尝试重提取特征点
           RedetectFeatures(im, global_mask);

           // 主动进行三角化以生成新的3D点
           PerformActiveTriangulation();
       }
 
         // 关键帧插入
         KeyFrameInsertion(im, masks);
 
        // 将当前帧插入临时缓冲区
    // pass current KLT SSIM values if available
    {
        auto ssim_values = klt_tracker_.GetLastSSIMValues();
        map_->SetLastFrame(current_frame_, ssim_values);
    }
        
        // 更新所有活跃MapPoint的生命周期
        for (auto& [mappoint_id, mappoint] : map_->GetMapPoints()) {
            if (mappoint->IsActive()) {
                mappoint->IncrementLifetime();
            }
        }

           // 绘制当前帧
        image_visualizer_->DrawCurrentFrame(*current_frame_);
         image_visualizer_->DrawRegularizationGraph(*current_frame_, *(map_->GetRegularizationGraph()));
         image_visualizer_->DrawFeatures(current_frame_->Keypoints());
         
        // 每帧指标统计（简洁中文版）
        int total_keypoints = current_frame_->Keypoints().size();
        int tracked_3d_keypoints = current_frame_->GetKeypointsWithStatus({TRACKED_WITH_3D}).size();
        int tracked_keypoints = current_frame_->GetKeypointsWithStatus({TRACKED}).size();
        int just_triangulated = current_frame_->GetKeypointsWithStatus({JUST_TRIANGULATED}).size();
        int candidate_3d = current_frame_->GetKeypointsWithStatus({CANDIDATE_3D}).size();
        int bad_keypoints = current_frame_->GetKeypointsWithStatus({BAD}).size();

        // 计算点数变化趋势
        int total_3d_points = tracked_3d_keypoints + just_triangulated + candidate_3d;
        int point_change = (prev_frame_idx_ >= 0) ? total_3d_points - prev_total_3d_ : 0;

        // 简洁的中文输出
        std::string change_str = (point_change >= 0) ? ("+" + std::to_string(point_change)) : std::to_string(point_change);
        LOG(INFO) << "【第" << frame_idx << "帧】总特征:" << total_keypoints 
                  << " | 3D点:" << total_3d_points 
                  << " (已跟踪:" << tracked_3d_keypoints 
                  << " | 新三角化:" << just_triangulated 
                  << " | 候选:" << candidate_3d 
                  << " | 失败:" << bad_keypoints << ")"
                  << " | 变化:" << change_str;

        // 将统计数据写入CSV文件
        try {
            std::ofstream stats_file(tracking_stats_path_, std::ios::app);
            if (stats_file.is_open()) {
                stats_file << frame_idx << ","
                           << total_keypoints << ","
                           << tracked_3d_keypoints << ","
                           << tracked_keypoints << ","
                           << just_triangulated << ","
                           << candidate_3d << ","
                           << bad_keypoints << ","
                           << total_3d_points << ","
                           << point_change << ","
                           << std::fixed << std::setprecision(4)
                           << (total_keypoints > 0 ? static_cast<float>(tracked_3d_keypoints) / total_keypoints : 0.0f) << "\n";
                stats_file.close();
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to write tracking statistics: " << e.what();
        }

        prev_total_3d_ = total_3d_points;
        prev_frame_idx_ = frame_idx;
     }
     
     // 保存当前帧的点云、相机位姿和RGB图像（使用相同帧ID命名）
     SaveCurrentFrameData(im, frame_idx);
     
    // 导出Marigold-DC联合优化数据
    if (enable_marigold_export_ && tracking_status_ == TRACKING) {
        ExportMarigoldData(im, frame_idx);
    }
    
    // 导出JSON/PLY格式数据（创新点1：联合优化融合）
    if (options_.enable_json_ply_export && tracking_status_ == TRACKING) {
        ExportJSONAndPLY(frame_idx, options_.json_path_template, options_.ply_path_template);
    }
    
    // 加载组织分割mask（创新点2：首次加载）
    if (options_.enable_heterogeneous_viscoelastic && !masks_loaded_) {
        LoadTissueMasks(options_.mask_json_path);
    }
    
    // 更新上一帧指针（用于下一次变形计算）
    if (!map_->IsEmpty()) {
        previous_frame_ = std::make_shared<Frame>(*current_frame_);
    }
 }
 
 Tracking::TrackingStatus Tracking::GetTrackingStatus() const {
     return tracking_status_;
 }
 
 void Tracking::ExtractFeatures(const cv::Mat& im, const cv::Mat& mask,
                                std::vector<cv::KeyPoint>& keypoints) {
     // 提取特征
     feature_extractor_->Extract(im, keypoints);
 
    // 如果 mask 为空，保留所有特征点（无掩码模式）
    if (mask.empty()) return;

    // 应用掩码过滤点
    vector<cv::KeyPoint> masked_keypoints;
    cv::Rect mask_rect(0, 0, mask.cols, mask.rows);
    for(size_t i = 0; i < keypoints.size(); i++){
        cv::Point pt(cv::Point2f(keypoints[i].pt));
        if(mask_rect.contains(pt) && mask.at<uchar>(pt)){
            masked_keypoints.push_back(keypoints[i]);
        }
    }
 
     keypoints = masked_keypoints;
 }
 
 void Tracking::MonocularMapInitialization(const cv::Mat& im_left,
                                        const cv::Mat& mask, const cv::Mat& im_clahe) {
     auto initialization_status = monocular_map_initializer_->ProcessNewImage(im_left, im_clahe, mask);
 
     if(!initialization_status.ok()) {
         LOG(INFO) << initialization_status.status().message();
         return;
     }
 
     auto initialization_results = *initialization_status;
 
     vector<float> depths;
     for (int idx = 0; idx < initialization_results.current_keypoints.size(); idx++) {
         Eigen::Vector3f current_landmark_position = initialization_results.current_landmark_positions[idx];
         depths.push_back(current_landmark_position.z());
     }
 
    if (depths.empty()) {
        LOG(WARNING) << "No valid depths found for scale estimation";
        return;
    }

    const int median_idx = depths.size() / 2;
    nth_element(depths.begin(), depths.begin() + median_idx, depths.end());
    const float median_depth = depths[median_idx];

    if (median_depth <= 0.0f) {
        LOG(WARNING) << "Invalid median depth for scale estimation: " << median_depth;
        return;
    }

    // 使用可配置的"目标中值深度"来估计全局尺度，避免不同数据集下硬编码 3.f 带来的整体压扁/拉长
    const float target_median_depth = options_.mono_init_target_median_depth;
    const float scale = target_median_depth / median_depth;
     map_->SetMapScale(scale);
 
    float sigma = Sigma(depths);
    float sigma_scaled = sigma * scale;
 
    // 计算初始化深度的合理范围，用于剔除极端偏近/偏远的离群点
    const float depth_min_allowed = std::max(0.0f, target_median_depth - 3.0f * sigma_scaled);
    const float depth_max_allowed = target_median_depth + 3.0f * sigma_scaled;

    Frame reference_frame;
     for (int idx = 0; idx < initialization_results.current_keypoints.size(); idx++) {
         cv::KeyPoint reference_keypoint = initialization_results.reference_keypoints[idx];
         cv::KeyPoint current_keypoint = initialization_results.current_keypoints[idx];
 
        Eigen::Vector3f reference_landmark_position = initialization_results.reference_landmark_positions[idx] * scale;
        Eigen::Vector3f current_landmark_position = initialization_results.current_landmark_positions[idx] * scale;

        // 以当前帧为参考，过滤掉深度明显超出整体范围的异常点
        const float current_depth_scaled = current_landmark_position.z();
        if (current_depth_scaled < depth_min_allowed || current_depth_scaled > depth_max_allowed) {
            continue;
        }
 
         ID mappoint_id = map_->CreateAndInsertMapPoint(reference_landmark_position,
                                                        reference_keypoint.class_id)->GetId();
 
         reference_frame.InsertObservation(reference_keypoint,
                                           reference_landmark_position,
                                           mappoint_id,
                                           TRACKED_WITH_3D);
 
         current_frame_->InsertObservation(current_keypoint,
                                           current_landmark_position,
                                           mappoint_id,
                                           TRACKED_WITH_3D);
     }
     reference_frame.SetCalibration(calibration_);
 
     reference_frame.MutableCameraTransformationWorld() = Sophus::SE3f();
     initialization_results.camera_transform_world.translation() = initialization_results.camera_transform_world.translation() * scale;
     current_frame_->MutableCameraTransformationWorld() = initialization_results.camera_transform_world;
 
     // 从帧创建关键帧
     auto first_keyframe = make_shared<KeyFrame>(reference_frame);
     auto current_keyframe = make_shared<KeyFrame>(*current_frame_);
 
     // 将关键帧插入地图
     map_->InsertKeyFrame(first_keyframe);
     map_->InsertKeyFrame(current_keyframe);
 
        {
            auto ssim_values = klt_tracker_.GetLastSSIMValues();
            map_->SetLastFrame(current_frame_, ssim_values);
        }
 
     // 初始化正则化图
    // 使用可配置的比例系数控制正则化图半径，减轻“过度平滑导致点云略扁”的风险
    const float reg_radius = sigma_scaled * options_.mono_init_reg_sigma_scale;
    map_->InitializeRegularizationGraph(reg_radius);
 
     // 为KLT跟踪器设置参考图像
     klt_tracker_.SetReferenceImage(im_left, current_frame_->Keypoints());
 
     // 保存地图点的光度信息
     for (const auto& [mappoint_id, idx] : current_frame_->MapPointIdToIndex()) {
         LucasKanadeTracker::PhotometricInformation photometric_information =
                 klt_tracker_.GetPhotometricInformationOfPoint(idx);
 
         map_->GetMapPoint(mappoint_id)->SetPhotometricInformation(photometric_information);
     }
 
     tracking_status_ = TRACKING;
 }
 
 void Tracking::StereoMapInitialization(const cv::Mat& im_left, const cv::Mat& im_right,
                                        const cv::Mat& mask, const cv::Mat& im_clahe) {
     current_frame_->Clear();
 
     vector<cv::KeyPoint> keypoints;
     ExtractFeatures(im_clahe, mask, keypoints);
 
     std::vector<Eigen::Vector3f> filtered_landmarks;
     std::vector<cv::KeyPoint> filtered_keypoints;
 
    // 使用可配置的立体匹配焦距参数，避免对特定数据集的硬编码
    auto stereo_matcher = StereoPatternMatching(calibration_, options_.stereo_pattern_focal);
     for (int idx = 0; idx < keypoints.size(); idx++) {
         auto landmark = stereo_matcher.computeStereo3D(keypoints[idx], im_left, im_right);
        if (landmark.ok()) {
            const float z = (*landmark).z();
            // 使用可配置的深度范围做过滤，默认基本不过滤，便于针对不同场景调节
            if (z > options_.stereo_init_min_depth && z < options_.stereo_init_max_depth) {
             filtered_landmarks.push_back(*landmark);
             filtered_keypoints.push_back(keypoints[idx]);
            }
         }
     }
 
     // 应用dbscan进一步去除离群点
     vector<int> labels = Dbscan3D(filtered_landmarks);
 
     std::vector<float> depths;
 
     for (int idx = 0; idx < labels.size(); idx++) {
         if (labels[idx] == 0) {
             depths.push_back(filtered_landmarks[idx].z());
         }
     }
 
    if (depths.empty()) {
        LOG(WARNING) << "No valid depths found for stereo initialization";
        return;
    }
    
    const int median_idx = depths.size() / 2;
    nth_element(depths.begin(), depths.begin() + median_idx, depths.end());
    float median_depth = depths[median_idx];
    
    // 初始化相机位姿为单位矩阵（第一帧作为世界坐标系原点）
    current_frame_->MutableCameraTransformationWorld() = Sophus::SE3f();

     for(int idx = 0; idx < labels.size(); idx++){
         if (labels[idx] == 0) {
             // Stereo计算返回的是相机坐标系中的点，需要转换为世界坐标系
             // 由于初始化时相机位姿是单位矩阵，camera_transform_world.inverse()也是单位矩阵
             // 但为了代码的正确性和可维护性，显式进行坐标转换
             Eigen::Vector3f camera_position = filtered_landmarks[idx];
             Eigen::Vector3f world_position = current_frame_->CameraTransformationWorld().inverse() * camera_position;
             
             auto mappoint = map_->CreateAndInsertMapPoint(world_position,
                                                           filtered_keypoints[idx].class_id);
             current_frame_->InsertObservation(filtered_keypoints[idx],
                                              world_position,
                                              mappoint->GetId(),
                                              TRACKED_WITH_3D);
         }
     }
 
    {
        auto ssim_values = klt_tracker_.GetLastSSIMValues();
        map_->SetLastFrame(current_frame_, ssim_values);
    }
 
    // 初始化正则化图与地图尺度（可通过 Tracking::Options 配置）
    map_->InitializeRegularizationGraph(options_.stereo_reg_radius);
    map_->SetMapScale(options_.stereo_map_scale);
 
     // 为KLT跟踪器设置参考图像
     klt_tracker_.SetReferenceImage(im_left, current_frame_->Keypoints());
 
     // 保存地图点的光度信息
     for (const auto& [mappoint_id, idx] : current_frame_->MapPointIdToIndex()) {
         LucasKanadeTracker::PhotometricInformation photometric_information =
                 klt_tracker_.GetPhotometricInformationOfPoint(idx);
 
         map_->GetMapPoint(mappoint_id)->SetPhotometricInformation(photometric_information);
     }
 
     // 从当前帧创建关键帧
     auto keyframe = make_shared<KeyFrame>(*current_frame_);
 
     // 将关键帧插入地图
     map_->InsertKeyFrame(keyframe);
 
     tracking_status_ = TRACKING;
 }
 
absl::flat_hash_set<ID> Tracking::TrackCameraAndDeformation(const cv::Mat &im, const cv::Mat& mask, int frame_idx) {
    // 执行数据关联
    DataAssociation(im, mask);

    // 粗略相机位姿估计
    CameraPoseEstimation();

    // 变形+相机位姿估计
    auto lost_ids = CameraPoseAndDeformationEstimation(im, frame_idx);
    return lost_ids;
}
 
 void Tracking::DataAssociation(const cv::Mat &im, const cv::Mat &mask) {
     // 修复：在进行KLT跟踪前，将临时不良状态重置为TRACKED
     // 这样之前因SSIM低或图像边界外而失败的点有机会在当前帧重新被跟踪
     auto& statuses = current_frame_->LandmarkStatuses();
     for (auto& status : statuses) {
         if (status == BAD_FEATURE || status == OUT_IMAGE_BOUNDARIES) {
             status = TRACKED;
         }
     }
     
     klt_tracker_.Track(im, current_frame_->Keypoints(), current_frame_->LandmarkStatuses(),
                        true, options_.klt_min_SSIM, mask);
 }
 
 void Tracking::CameraPoseEstimation() {
     // 应用运动模型获取当前相机位姿的初始值
     current_frame_->MutableCameraTransformationWorld() = motion_model_ *
             current_frame_->CameraTransformationWorld();
 
     previous_camera_transform_world_ = current_frame_->CameraTransformationWorld();
 
     // 执行优化
     CameraPoseOptimization(*current_frame_, previous_camera_transform_world_);
 }
 
absl::flat_hash_set<ID> Tracking::CameraPoseAndDeformationEstimation(const cv::Mat &im, int frame_idx) {
    // 构建异质粘弹性参数映射（创新点2）
    absl::flat_hash_map<ID, std::pair<float, float>> hetero_params;
    absl::flat_hash_map<std::string, std::pair<int, std::pair<float, float>>> class_param_stats; // class_name -> {count, {avg_k, avg_b}}
    
    int current_frame_id = frame_idx;
    
    if (options_.enable_heterogeneous_viscoelastic && masks_loaded_) {
        auto keypoints_3d = current_frame_->GetKeypointsWithStatus({TRACKED_WITH_3D});
        auto mappoint_ids = current_frame_->GetMapPointsIdsWithStatus({TRACKED_WITH_3D});
        const auto& index_to_id = current_frame_->IndexToMapPointId();
        
        // 获取真实的图像尺寸
        int H = im.rows;
        int W = im.cols;
        
        // 用于统计每个组织类别的参数
        absl::flat_hash_map<std::string, std::vector<std::pair<float, float>>> class_params_map;
        
        // 获取最后一次跟踪的SSIM值
        std::vector<float> ssim_values = klt_tracker_.GetLastSSIMValues();
        
        // 获取 temporal buffer 用于检查跟踪长度
        auto temporal_buffer = map_->GetTemporalBuffer();
        
        for (size_t i = 0; i < keypoints_3d.size(); ++i) {
            auto it = index_to_id.find(static_cast<int>(i));
            if (it != index_to_id.end()) {
                ID mp_id = it->second;
                cv::Point2f pt = keypoints_3d[i].pt;
                // 对坐标进行clamp，确保在图像范围内
                int x = std::clamp(static_cast<int>(pt.x), 0, W - 1);
                int y = std::clamp(static_cast<int>(pt.y), 0, H - 1);
                int pixel_idx = y * W + x;
                
                // 获取置信度相关信息
                float ssim = (i < ssim_values.size()) ? ssim_values[i] : 0.0f;
                int track_length = temporal_buffer->TrackLength(keypoints_3d[i].class_id);
                auto mappoint = map_->GetMapPoint(mp_id);
                int lifetime = mappoint ? mappoint->GetLifetime() : 0;
                
                // 检查置信度条件
                bool use_hetero_params = false;
                if (ssim > 0.6f && track_length >= 3 && lifetime >= 1) {
                    // 检查深度变化稳定性（简化版：至少有2个历史深度值）
                    if (mappoint) {
                        auto landmark_flow = mappoint->GetLandmarkFlow(3);
                        if (landmark_flow.size() >= 2) {
                            use_hetero_params = true;
                        }
                    }
                }
                
                float k, b;
                std::string class_name;
                if (use_hetero_params && GetHeterogeneousViscoelasticParams(mp_id, pixel_idx, k, b, &class_name)) {
                    hetero_params[mp_id] = std::make_pair(k, b);
                    if (!class_name.empty()) {
                        class_params_map[class_name].push_back({k, b});
                    }
                }
            }
        }
        
        // 计算每个组织类别的平均参数
        for (const auto& entry : class_params_map) {
            const std::string& cls_name = entry.first;
            const auto& params_list = entry.second;
            if (!params_list.empty()) {
                float sum_k = 0.0f, sum_b = 0.0f;
                for (const auto& p : params_list) {
                    sum_k += p.first;
                    sum_b += p.second;
                }
                float avg_k = sum_k / params_list.size();
                float avg_b = sum_b / params_list.size();
                class_param_stats[cls_name] = {static_cast<int>(params_list.size()), {avg_k, avg_b}};
            }
        }
        
        // 按帧保存模式：确保数据已加载
        if (use_per_frame_files_) {
            if (frame_loaded_cache_.find(current_frame_id) == frame_loaded_cache_.end() ||
                !frame_loaded_cache_.at(current_frame_id)) {
                LoadFrameMaskDataFromFile(current_frame_id);
            }
        }
        
        // 输出当前帧的 mask 使用情况
        auto frame_it = frame_mask_data_.find(current_frame_id);
        if (frame_it != frame_mask_data_.end()) {
            LOG(INFO) << "[Frame " << current_frame_id << "] ✅ Using mask data from: " 
                      << std::filesystem::absolute(options_.mask_json_path).string();
            
            // 显示该帧 mask 数据中的所有组织
            const auto& frame_data = frame_it->second;
            LOG(INFO) << "[Frame " << current_frame_id << "] Mask data contains " 
                      << frame_data.class_masks.size() << " tissue class(es):";
            
            // RGB 转中文颜色描述的函数
            auto rgbToChineseColor = [](const std::vector<int>& rgb) -> std::string {
                if (rgb.size() < 3) return "未知颜色";
                int r = rgb[0], g = rgb[1], b = rgb[2];
                if (r > 200 && g < 80 && b < 80) return "红色";
                if (g > 200 && r < 80 && b < 80) return "绿色";
                if (b > 200 && r < 80 && g < 80) return "蓝色";
                if (r > 200 && g > 200 && b < 80) return "黄色";
                if (r > 200 && b > 200 && g < 80) return "品红/紫红";
                if (g > 200 && b > 200 && r < 80) return "青色";
                if (r > 200 && g > 120 && b < 80) return "橙色";
                if (r > 120 && b > 200 && g < 120) return "紫色/蓝紫";
                if (g > 120 && b > 200 && r < 120) return "青蓝/湖蓝";
                if (r > 120 && g > 200 && b < 120) return "黄绿";
                return "其他混合色";
            };
            
            for (const auto& cls_entry : frame_data.class_masks) {
                const std::string& cls_name = cls_entry.first;
                const auto& cls_mask = cls_entry.second;
                auto param_it = options_.tissue_params.find(cls_name);
                
                // 构建颜色信息字符串
                std::string color_info = "";
                if (!cls_mask.color.empty() && cls_mask.color.size() >= 3) {
                    std::string chinese_color = rgbToChineseColor(cls_mask.color);
                    color_info = " [RGB: (" + std::to_string(cls_mask.color[0]) + "," 
                                + std::to_string(cls_mask.color[1]) + "," 
                                + std::to_string(cls_mask.color[2]) + ") -> " + chinese_color + "]";
                }
                
                if (param_it != options_.tissue_params.end()) {
                    LOG(INFO) << "    - " << cls_name << color_info << " (area: " << cls_mask.area 
                              << " pixels, score: " << std::fixed << std::setprecision(3) << cls_mask.score
                              << ") -> k=" << std::fixed << std::setprecision(2) << param_it->second.k
                              << ", b=" << std::fixed << std::setprecision(2) << param_it->second.b;
                } else {
                    LOG(INFO) << "    - " << cls_name << color_info << " (area: " << cls_mask.area 
                              << " pixels, score: " << std::fixed << std::setprecision(3) << cls_mask.score
                              << ") -> (auto-guessed params)";
                }
            }
            
            // 显示实际匹配到特征点的组织统计
            LOG(INFO) << "[Frame " << current_frame_id << "] Feature points matched to tissue classes:";
            if (!class_param_stats.empty()) {
                int total_matched_points = 0;
                for (const auto& stat : class_param_stats) {
                    total_matched_points += stat.second.first;
                    LOG(INFO) << "    - " << stat.first << ": " << stat.second.first 
                              << " points, avg k=" << std::fixed << std::setprecision(2) << stat.second.second.first
                              << ", avg b=" << std::fixed << std::setprecision(2) << stat.second.second.second;
                }
                LOG(INFO) << "[Frame " << current_frame_id << "] Total matched points: " 
                          << total_matched_points << " / " << keypoints_3d.size() 
                          << " tracked 3D points";
            } else {
                LOG(INFO) << "    - No points matched to any tissue class (using default k=1.0, b=0.5)";
                LOG(INFO) << "[Frame " << current_frame_id << "] Total tracked 3D points: " << keypoints_3d.size();
            }
        } else {
            LOG(WARNING) << "[Frame " << current_frame_id << "] ⚠️  No mask data found for this frame, using default parameters (k=1.0, b=0.5)";
        }
    } else if (options_.enable_heterogeneous_viscoelastic && !masks_loaded_) {
        LOG(WARNING) << "[Frame " << current_frame_id << "] ⚠️  Heterogeneous viscoelastic enabled but mask file not loaded, using default parameters";
    }
    
    // 分阶段优化
    // Step 1: Camera-only BA (冻结 deformation)
    CameraPoseOptimization(*current_frame_, previous_camera_transform_world_);
    
    // Step 2: Joint BA (解锁 deformation)
    auto lost_mappoint_ids = CameraPoseAndDeformationOptimization(*current_frame_,
                                         map_,previous_camera_transform_world_,
                                         map_->GetMapScale(),
                                         hetero_params.empty() ? nullptr : &hetero_params);

     // 更新运动模型
     motion_model_ = current_frame_->CameraTransformationWorld() *
                     map_->GetLastFrame().CameraTransformationWorld().inverse();

     return lost_mappoint_ids;
 }
 
 void Tracking::KeyFrameInsertion(const cv::Mat& im,
                                  const absl::flat_hash_map<std::string, cv::Mat>& masks) {
     if (NeedNewKeyFrame()) {
         CreateNewKeyFrame(im, masks);
     }
 }
 
 bool Tracking::NeedNewKeyFrame() {
     if(n_images_from_last_keyframe_ >= options_.images_to_insert_keyframe){
         n_images_from_last_keyframe_ = 0;
         return true;
     }
     else{
         n_images_from_last_keyframe_++;
         return false;
     }
 }
 
 void Tracking::CreateNewKeyFrame(const cv::Mat& im,
                                  const absl::flat_hash_map<std::string, cv::Mat>& masks) {
    // 安全获取全局mask
    cv::Mat global_mask = masks.count("Global") ? masks.at("Global") : cv::Mat();

    // 提取新特征
    ExtractFeaturesInFrame(im, global_mask, *current_frame_);
 
     // 从当前帧创建关键帧
     auto keyframe = make_shared<KeyFrame>(*current_frame_);
 
     // 将关键帧插入地图
     map_->InsertKeyFrame(keyframe);
 
     // 更新当前帧
     current_frame_->SetFromKeyFrame(keyframe);
 
    // 设置新的KLT参考
    SetKLTReference(im, *current_frame_, global_mask);
 }
 
void Tracking::ExtractFeaturesInFrame(const cv::Mat& im, const cv::Mat& mask, Frame &frame) {
    // 获取现有特征点的位置，用于距离过滤
    vector<cv::KeyPoint> existing_keypoints = frame.Keypoints();

    // 提取新的特征点
    vector<cv::KeyPoint> new_keypoints;
    ExtractFeatures(im, mask, new_keypoints);

    // 过滤与现有特征点距离太近的新特征点
    vector<cv::KeyPoint> filtered_new_keypoints;
    for (const auto& new_kp : new_keypoints) {
        bool too_close = false;
        for (const auto& existing_kp : existing_keypoints) {
            float dist = cv::norm(new_kp.pt - existing_kp.pt);
            if (dist < 10.0f) {  // 最小距离10像素
                too_close = true;
                break;
            }
        }

        if (!too_close) {
            filtered_new_keypoints.push_back(new_kp);
        }
    }

    // 只插入新增的特征点，标记为TRACKED状态
    for (const auto& kp : filtered_new_keypoints) {
        frame.InsertObservation(kp, Eigen::Vector3f::Zero(), -1, TRACKED);
    }

    LOG(INFO) << "[关键帧] 提取了新特征点数: " << filtered_new_keypoints.size();
}
 
 void Tracking::SetKLTReference(const cv::Mat& im, Frame& frame, const cv::Mat& mask) {
     klt_tracker_.SetReferenceImage(im, frame.Keypoints(), mask);
 
     // 更新地图点的光度信息
     for (const auto &[mappoint_id, idx] : frame.MapPointIdToIndex()) {
         LucasKanadeTracker::PhotometricInformation photometric_information =
                 klt_tracker_.GetPhotometricInformationOfPoint(idx);
 
         map_->GetMapPoint(mappoint_id)->SetPhotometricInformation(photometric_information);
     }
 }
 
 void Tracking::PointReuse(const cv::Mat& im, const cv::Mat& mask,
                           absl::flat_hash_set<ID> lost_mappoint_ids) {
     auto all_mappoints = map_->GetMapPoints();
     for (const auto &[mappoint_id, mappoint] : all_mappoints) {
         if (!current_frame_->LandmarkPosition(mappoint_id).ok()) {
             // 历史信誉检查
             // 1. 检查点是否活跃
             if (!mappoint->IsActive()) {
                 continue;
             }
             
             // 2. 检查生命周期，避免重用太新或不稳定的点
             if (mappoint->GetLifetime() < 2) {
                 continue;
             }
             
             // 3. 检查是否有足够的历史深度值
             auto landmark_flow = mappoint->GetLandmarkFlow(2);
             if (landmark_flow.size() < 2) {
                 continue;
             }
             
             // 将地图点投影到相机并检查是否在图像内
             Eigen::Vector3f landmark_position_seed = mappoint->GetLastWorldPosition();
             Eigen::Vector3f landmark_camera_position = current_frame_->CameraTransformationWorld() * landmark_position_seed;

             if (landmark_camera_position.z() < 0) {
                 continue;
             }

             cv::Point2f projected_landmark = calibration_->Project(landmark_camera_position);

             if (projected_landmark.x >= 0 && projected_landmark.x < im.cols &&
                 projected_landmark.y >= 0 && projected_landmark.y < im.rows) {
                 lost_mappoint_ids.insert(mappoint_id);
             }
         }
     }
 
     if (lost_mappoint_ids.empty()) {
         return;
     }
 
     // 将候选点投影到图像
     Frame frame_with_only_candidates;
 
     LucasKanadeTracker klt(cv::Size(options_.klt_window_size, options_.klt_window_size),
                            1, options_.klt_max_iters,
                            options_.klt_epsilon, options_.klt_min_eig_th);
 
     int candidates_in_image = 0;
     vector<cv::KeyPoint> keypoint_seeds;
     for (const auto& mappoint_id : lost_mappoint_ids) {
         auto mappoint = map_->GetMapPoint(mappoint_id);
         Eigen::Vector3f landmark_position_seed = mappoint->GetLastWorldPosition();
         Eigen::Vector3f landmark_camera_position = current_frame_->CameraTransformationWorld() * landmark_position_seed;
        cv::Point2f projected_landmark = calibration_->Project(landmark_camera_position);

        if (std::isnan(projected_landmark.x) || std::isnan(projected_landmark.y)) {
            LOG(WARNING) << "NaN found in projected landmark, skipping this point";
            continue;
        }
 
         if (projected_landmark.x >= 0 && projected_landmark.x < im.cols &&
             projected_landmark.y >= 0 && projected_landmark.y < im.rows) {
             cv::KeyPoint keypoint(projected_landmark, 1);
             frame_with_only_candidates.InsertObservation(keypoint, landmark_position_seed, mappoint_id, TRACKED_WITH_3D);
 
             // 在KLT中设置光度信息
             LucasKanadeTracker::PhotometricInformation photometric_information =
                     map_->GetMapPoint(mappoint_id)->GetPhotometricInformation();
             klt.InsertPhotometricInformation(keypoint, photometric_information);
 
             keypoint_seeds.push_back(keypoint);
 
             candidates_in_image++;
         }
     }
 
     if (candidates_in_image == 0) {
         return;
     }
 
     // 使用KLT跟踪候选点
     klt.Track(im, frame_with_only_candidates.Keypoints(), frame_with_only_candidates.LandmarkStatuses(),
               true, 0.75, mask);
 
     // 将跟踪到的候选点插入当前帧
     vector<cv::KeyPoint> tracked_candidate_keypoints =
             frame_with_only_candidates.GetKeypointsWithStatus({TRACKED_WITH_3D});
     vector<Eigen::Vector3f> tracked_candidate_landmarks =
             frame_with_only_candidates.GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
     vector<ID> tracked_candidate_mappoint_ids =
             frame_with_only_candidates.GetMapPointsIdsWithStatus({TRACKED_WITH_3D});
 
     int reused_landmarks = 0;
 
     for (int idx = 0; idx < tracked_candidate_keypoints.size(); idx++) {
         cv::KeyPoint keypoint = tracked_candidate_keypoints[idx];
         Eigen::Vector3f landmark_position = tracked_candidate_landmarks[idx];
         ID mappoint_id = tracked_candidate_mappoint_ids[idx];
 
         keypoint.class_id = map_->GetMapPoint(mappoint_id)->GetKeyPointId();
 
        Eigen::Vector3f landmark_camera_position = current_frame_->CameraTransformationWorld() * landmark_position;
        cv::Point2f projected_landmark = calibration_->Project(landmark_camera_position);

        // 与三角化阶段保持一致，放宽重投影误差阈值以便在真实噪声条件下复用更多历史地图点。
        if (SquaredReprojectionError(projected_landmark, keypoint.pt) > 9.21f) {
             continue;
         }
 
         if (current_frame_->MapPointIdToIndex().contains(mappoint_id)) {
             const int idx_in_frame = current_frame_->MapPointIdToIndex().at(mappoint_id);
 
             current_frame_->Keypoints()[idx_in_frame] = keypoint;
             current_frame_->LandmarkPositions()[idx_in_frame] = landmark_position;
             current_frame_->LandmarkStatuses()[idx_in_frame] = TRACKED_WITH_3D;
 
         } else {
             current_frame_->InsertObservation(keypoint, landmark_position, mappoint_id, TRACKED_WITH_3D);
 
             LucasKanadeTracker::PhotometricInformation photometric_information =
                     map_->GetMapPoint(mappoint_id)->GetPhotometricInformation();
             klt_tracker_.InsertPhotometricInformation(keypoint, photometric_information);
         }
 
         reused_landmarks++;
     }

     if (reused_landmarks > 0) {
         LOG(INFO) << "[点重用] 成功复用历史地图点: " << reused_landmarks;
     }
 }
 
 void Tracking::UpdateTriangulatedPoints() {
     auto indices = current_frame_->GetIndexWithStatus({JUST_TRIANGULATED});
 
    for (auto index : indices) {
        LucasKanadeTracker::PhotometricInformation photometric_information =
                klt_tracker_.GetPhotometricInformationOfPoint(index);

        auto it = current_frame_->IndexToMapPointId().find(index);
        if (it == current_frame_->IndexToMapPointId().end()) {
            LOG(WARNING) << "Index " << index << " not found in IndexToMapPointId map, skipping";
            continue;
        }
        ID mappoint_id = it->second;
        auto mappoint = map_->GetMapPoint(mappoint_id);
        mappoint->SetPhotometricInformation(photometric_information);

        // 重置不良记录，因为刚刚成功三角化
        mappoint->ResetBadCount();

        // 设置保护期：新三角化的点在5帧内更 resistant to 被标记为BAD
        mappoint->SetProtectionFrames(5);

        // 更新地标状态
        current_frame_->LandmarkStatuses()[index] = TRACKED_WITH_3D;
     }
 }
 
// 简化：只保存RGB图像（点云和位姿由map_visualizer负责保存）
void Tracking::SaveCurrentFrameData(const cv::Mat& rgb_image, int frame_idx) {
    if (map_->IsEmpty()) return;

    // 如果frame_idx为-1，使用内部计数器
    int actual_frame_idx;
    if (frame_idx == -1) {
        actual_frame_idx = internal_frame_counter_++;
    } else {
        actual_frame_idx = frame_idx;
    }
    last_export_frame_id_ = actual_frame_idx;

    // 生成图像文件名
    std::string image_filename = point_cloud_and_pose_path_ + std::to_string(actual_frame_idx) + ".png";
     
    // 只保存RGB图像
    SaveImage(rgb_image, image_filename);
} // 新增：保存RGB图像到文件
 void Tracking::SaveImage(const cv::Mat& image, const std::string& filename) {
     if (image.empty()) {
         LOG(WARNING) << "Cannot save empty image: " << filename;
         return;
     }
 
     // 如果是单通道图像，转换为三通道以保存为彩色图像
     cv::Mat save_img;
     if (image.channels() == 1) {
         cv::cvtColor(image, save_img, cv::COLOR_GRAY2BGR);
     } else {
         save_img = image;
     }
 
     if (!cv::imwrite(filename, save_img)) {
         LOG(WARNING) << "Failed to save image: " << filename;
     } else {
         LOG(INFO) << "Saved RGB image: " << filename;
     }
}

// 导出Marigold-DC联合优化所需数据
void Tracking::ExportMarigoldData(const cv::Mat& rgb_image, int frame_idx) {
    if (!marigold_exporter_ || map_->IsEmpty()) {
        return;
    }
    
    try {
        // 如果frame_idx为-1，使用内部计数器
        int actual_frame_idx = (frame_idx == -1) ? last_export_frame_id_ : frame_idx;
        if (actual_frame_idx < 0) {
            LOG(WARNING) << "No valid frame id for Marigold export.";
            return;
        }
        
        // 从当前帧和地图导出数据包
        auto data_package = marigold_exporter_->ExportFrameData(
            rgb_image, 
            *current_frame_, 
            map_, 
            actual_frame_idx, 
            previous_frame_.get(),  // 上一帧指针，用于计算变形偏移
            &klt_tracker_          // KLT跟踪器引用，用于获取SSIM值
        );
        
        // 保存数据包到文件
        marigold_exporter_->SaveDataPackage(data_package, actual_frame_idx);
        
        LOG(INFO) << "Exported Marigold data for frame " << actual_frame_idx 
                  << " with " << data_package.ddg_constraints.connections.size() << " DDG connections";
                  
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to export Marigold data for frame " << frame_idx 
                   << ": " << e.what();
    }
}

// 导出JSON/PLY格式数据（创新点1：联合优化融合）
void Tracking::ExportJSONAndPLY(int frame_idx, const std::string& json_path_template, const std::string& ply_path_template) {
    if (map_->IsEmpty() || tracking_status_ != TRACKING) {
        return;
    }
    
    try {
        int actual_frame_idx = (frame_idx == -1) ? last_export_frame_id_ : frame_idx;
        if (actual_frame_idx < 0) {
            LOG(WARNING) << "No valid frame id for JSON/PLY export.";
            return;
        }
        
        // 格式化路径
        char json_path[512], ply_path[512];
        snprintf(json_path, sizeof(json_path), json_path_template.c_str(), actual_frame_idx);
        snprintf(ply_path, sizeof(ply_path), ply_path_template.c_str(), actual_frame_idx);
        
        // 确保输出目录存在
        std::filesystem::path json_file_path(json_path);
        std::filesystem::path ply_file_path(ply_path);
        std::filesystem::create_directories(json_file_path.parent_path());
        std::filesystem::create_directories(ply_file_path.parent_path());
        
        // 1. 提取稀疏点云
        auto& all_mappoints = map_->GetMapPoints();
        std::vector<Eigen::Vector3f> sparse_points;
        std::vector<ID> mappoint_ids;
        absl::flat_hash_map<ID, int> id_to_index;
        
        int point_idx = 0;
        for (const auto& [id, mappoint] : all_mappoints) {
            if (mappoint->IsActive()) {
                sparse_points.push_back(mappoint->GetLastWorldPosition());
                mappoint_ids.push_back(id);
                id_to_index[id] = point_idx++;
            }
        }
        
        // 2. 保存PLY文件
        std::ofstream ply_file(ply_path);
        if (!ply_file.is_open()) {
            LOG(ERROR) << "Failed to open PLY file: " << ply_path;
            return;
        }
        // 过滤掉非法点（NaN/Inf），并统计范围信息，保存调试统计文件
        size_t valid_count = 0;
        double xmin = std::numeric_limits<double>::infinity();
        double xmax = -std::numeric_limits<double>::infinity();
        double ymin = std::numeric_limits<double>::infinity();
        double ymax = -std::numeric_limits<double>::infinity();
        double zmin = std::numeric_limits<double>::infinity();
        double zmax = -std::numeric_limits<double>::infinity();
        for (const auto& pt : sparse_points) {
            if (std::isfinite(pt.x()) && std::isfinite(pt.y()) && std::isfinite(pt.z())) {
                ++valid_count;
                xmin = std::min<double>(xmin, pt.x());
                xmax = std::max<double>(xmax, pt.x());
                ymin = std::min<double>(ymin, pt.y());
                ymax = std::max<double>(ymax, pt.y());
                zmin = std::min<double>(zmin, pt.z());
                zmax = std::max<double>(zmax, pt.z());
            }
        }

        // 保存统计信息
        try {
            std::string stats_path = std::filesystem::path(ply_path).parent_path().string() + "/"
                                     + std::filesystem::path(ply_path).stem().string() + "_stats.txt";
            std::ofstream stats_out(stats_path);
            if (stats_out.is_open()) {
                stats_out << "total_points: " << sparse_points.size() << "\n";
                stats_out << "valid_points: " << valid_count << "\n";
                stats_out << "x_min: " << (valid_count ? xmin : 0.0) << "\n";
                stats_out << "x_max: " << (valid_count ? xmax : 0.0) << "\n";
                stats_out << "y_min: " << (valid_count ? ymin : 0.0) << "\n";
                stats_out << "y_max: " << (valid_count ? ymax : 0.0) << "\n";
                stats_out << "z_min: " << (valid_count ? zmin : 0.0) << "\n";
                stats_out << "z_max: " << (valid_count ? zmax : 0.0) << "\n";
                stats_out.close();
            }
        } catch (...) {
            LOG(WARNING) << "Failed to write PLY stats file";
        }

        ply_file << "ply\n";
        ply_file << "format ascii 1.0\n";
        ply_file << "element vertex " << valid_count << "\n";
        ply_file << "property float x\n";
        ply_file << "property float y\n";
        ply_file << "property float z\n";
        ply_file << "end_header\n";

        for (const auto& pt : sparse_points) {
            if (!(std::isfinite(pt.x()) && std::isfinite(pt.y()) && std::isfinite(pt.z()))) {
                continue;
            }
            ply_file << std::fixed << std::setprecision(6)
                     << pt.x() << " " << pt.y() << " " << pt.z() << "\n";
        }
        ply_file.close();
        LOG(INFO) << "Saved PLY file: " << ply_path << " with " << valid_count << " / " << sparse_points.size() << " valid points";
        
        // 3. 提取DDG信息
        auto reg_graph = map_->GetRegularizationGraph();
        std::vector<std::vector<float>> ddg_edges; // [i, j, l0, w]
        
        for (const auto& id : mappoint_ids) {
            auto edges = reg_graph->GetEdges(id);
            for (const auto& [other_id, edge] : edges) {
                if (id_to_index.contains(other_id) && id < other_id) {
                    float stretch = (edge->max_distance - edge->min_distance) / std::max(edge->min_distance, 1e-6f);
                    if (stretch < 0.3f) {  // th_stretching=0.3筛选E_rel
                        int i = id_to_index[id];
                        int j = id_to_index[other_id];
                        float l0 = edge->first_distance;
                        float w = edge->weight;
                        float sigma = reg_graph->GetMinWeightAllowed() > 0 ? 0.5f : 0.5f; // 简化
                        float w_ij = w; // 使用edge的weight
                        ddg_edges.push_back({static_cast<float>(i), static_cast<float>(j), l0, w_ij});
                    }
                }
            }
        }
        
        // 4. 提取粘弹性参数
        std::vector<std::vector<float>> visc_points; // [δ_x, δ_y, δ_z, s_i, k, b]
        
        // 获取当前帧的SSIM置信度（如果有）
        auto current_keypoints = current_frame_->GetKeypointsWithStatus({TRACKED_WITH_3D});
        auto current_positions = current_frame_->GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
        absl::flat_hash_map<ID, float> ssim_map;
        
        if (previous_frame_) {
            auto prev_positions = previous_frame_->GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
            const auto& prev_index_to_id = previous_frame_->IndexToMapPointId();
            const auto& curr_index_to_id = current_frame_->IndexToMapPointId();
            
            for (size_t i = 0; i < current_positions.size(); ++i) {
                auto it = curr_index_to_id.find(static_cast<int>(i));
                if (it != curr_index_to_id.end()) {
                    ID mp_id = it->second;
                    // 查找上一帧中的位置
                    for (size_t j = 0; j < prev_positions.size(); ++j) {
                        auto prev_it = prev_index_to_id.find(static_cast<int>(j));
                        if (prev_it != prev_index_to_id.end() && prev_it->second == mp_id) {
                            Eigen::Vector3f delta = current_positions[i] - prev_positions[j];
                            ssim_map[mp_id] = 0.8f; // 简化：使用固定值，实际应从KLT获取
                            break;
                        }
                    }
                }
            }
        }
        
        for (const auto& id : mappoint_ids) {
            Eigen::Vector3f delta(0, 0, 0);
            float s_i = ssim_map.contains(id) ? ssim_map[id] : 0.5f;
            float k = 1.0f; // 默认弹性系数
            float b = 0.5f; // 默认粘性系数

            if (previous_frame_) {
                // 计算当前帧相对上一帧的变形偏移
                auto current_pos_status = current_frame_->LandmarkPosition(id);
                auto prev_pos_status = previous_frame_->LandmarkPosition(id);
                if (current_pos_status.ok() && prev_pos_status.ok()) {
                    delta = *current_pos_status - *prev_pos_status;
                }
            }
            
            visc_points.push_back({delta.x(), delta.y(), delta.z(), s_i, k, b});
        }
        
        // 5. 获取相机内参
        auto calib = calibration_;
        auto K = calib->ToIntrinsicsMatrix();
        float fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
        
        // 6. 计算跟踪成功率
        float track_rate = current_keypoints.size() > 0 ? 
            static_cast<float>(current_keypoints.size()) / 200.0f : 0.5f;
        track_rate = std::min(1.0f, track_rate);
        
        // 7. 生成JSON（简单字符串拼接）
        std::ofstream json_file(json_path);
        if (!json_file.is_open()) {
            LOG(ERROR) << "Failed to open JSON file: " << json_path;
            return;
        }
        
        json_file << "{\n";
        json_file << "  \"ddg\": {\n";
        json_file << "    \"edges\": [\n";
        for (size_t i = 0; i < ddg_edges.size(); ++i) {
            json_file << "      [" << static_cast<int>(ddg_edges[i][0]) << ", "
                      << static_cast<int>(ddg_edges[i][1]) << ", "
                      << std::fixed << std::setprecision(6) << ddg_edges[i][2] << ", "
                      << ddg_edges[i][3] << "]";
            if (i < ddg_edges.size() - 1) json_file << ",";
            json_file << "\n";
        }
        json_file << "    ]\n";
        json_file << "  },\n";
        json_file << "  \"visc\": {\n";
        json_file << "    \"points\": [\n";
        for (size_t i = 0; i < visc_points.size(); ++i) {
            json_file << "      [" << visc_points[i][0] << ", " << visc_points[i][1] << ", "
                      << visc_points[i][2] << ", " << visc_points[i][3] << ", "
                      << visc_points[i][4] << ", " << visc_points[i][5] << "]";
            if (i < visc_points.size() - 1) json_file << ",";
            json_file << "\n";
        }
        json_file << "    ]\n";
        json_file << "  },\n";
        json_file << "  \"camera_K\": [" << fx << ", " << fy << ", " << cx << ", " << cy << "],\n";
        json_file << "  \"track_rate\": " << track_rate << "\n";
        json_file << "}\n";
        json_file.close();
        
        LOG(INFO) << "Exported JSON: " << json_path << " with " << ddg_edges.size() 
                  << " DDG edges and " << visc_points.size() << " viscoelastic points";
                  
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to export JSON/PLY for frame " << frame_idx << ": " << e.what();
    }
}

// 加载组织分割mask（创新点2）- 支持按帧保存模式
void Tracking::LoadTissueMasks(const std::string& mask_json_path) {
    frame_mask_data_.clear();
    available_frame_ids_.clear();
    frame_loaded_cache_.clear();
    masks_loaded_ = false;
    use_per_frame_files_ = false;
    mask_json_path_ = mask_json_path;

    std::string abs_path = std::filesystem::absolute(mask_json_path).string();
    
    LOG(INFO) << "========================================";
    LOG(INFO) << "[Mask Loading] Attempting to load mask data...";
    LOG(INFO) << "[Mask Loading] Path: " << abs_path;

    // 检查是文件还是目录
    if (std::filesystem::is_directory(mask_json_path)) {
        // 按帧保存模式：目录中包含 frame_*.json 文件
        use_per_frame_files_ = true;
        frames_dir_ = mask_json_path;
        LOG(INFO) << "[Mask Loading] ✅ Detected per-frame file mode (directory)";
        
        // 扫描目录，找出所有可用的帧
        int frame_count = 0;
        for (const auto& entry : std::filesystem::directory_iterator(mask_json_path)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                // 检查是否以 "frame_" 开头且以 ".json" 结尾（兼容 C++17）
                if (filename.rfind("frame_", 0) == 0 && 
                    filename.length() > 5 && 
                    filename.substr(filename.length() - 5) == ".json") {
                    try {
                        std::string frame_id_str = filename.substr(6, filename.length() - 11); // "frame_" + id + ".json"
                        int frame_id = std::stoi(frame_id_str);
                        available_frame_ids_.insert(frame_id);
                        frame_count++;
                    } catch (...) {
                        continue;
                    }
                }
            }
        }
        
        if (frame_count > 0) {
            masks_loaded_ = true;
            LOG(INFO) << "[Mask Loading] ✅ Found " << frame_count << " frame mask files";
            LOG(INFO) << "[Mask Loading] Mask data will be loaded on-demand per frame";
        } else {
            LOG(WARNING) << "[Mask Loading] ⚠️  No frame mask files found in directory";
        }
        
        // 显示配置的组织参数
        if (!options_.tissue_params.empty()) {
            LOG(INFO) << "[Mask Loading] Configured tissue parameters:";
            for (const auto& param_entry : options_.tissue_params) {
                LOG(INFO) << "    - " << param_entry.first 
                          << ": k=" << std::fixed << std::setprecision(2) << param_entry.second.k
                          << ", b=" << std::fixed << std::setprecision(2) << param_entry.second.b;
            }
        }
        LOG(INFO) << "========================================";
        return;
    }
    
    // 单文件模式：检查文件是否存在
    if (!std::filesystem::exists(mask_json_path)) {
        LOG(WARNING) << "[Mask Loading] ❌ File not found: " << abs_path;
        LOG(WARNING) << "[Mask Loading] Skipping heterogeneous viscoelastic modeling.";
        LOG(INFO) << "========================================";
        return;
    }

    // 单文件模式：打开 JSON 文件
    cv::FileStorage fs(mask_json_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_AUTO);
    if (!fs.isOpened()) {
        LOG(ERROR) << "[Mask Loading] ❌ Failed to open mask JSON: " << abs_path;
        LOG(INFO) << "========================================";
        return;
    }
    
    LOG(INFO) << "[Mask Loading] ✅ Successfully opened mask file (single file mode)";
    LOG(INFO) << "[Mask Loading] ⚠️  Single file mode: will load all frames at once (may be slow for large files)";
    std::cout << "[Mask Loading] Scanning file structure..." << std::endl;
    std::cout.flush();

    auto convertMask = [](const std::vector<int>& src, int& area_out) {
        std::vector<uint8_t> dst;
        dst.reserve(src.size());
        area_out = 0;
        for (int v : src) {
            uint8_t val = static_cast<uint8_t>(v != 0);
            dst.push_back(val);
            area_out += val;
        }
        return dst;
    };

    // 先统计总帧数（用于进度显示）- 添加进度提示
    cv::FileNode root = fs.root();
    int total_frames = 0;
    int scan_count = 0;
    std::cout << "[Mask Loading] Counting frames..." << std::endl;
    std::cout.flush();
    
    for (auto frame_node = root.begin(); frame_node != root.end(); ++frame_node) {
        scan_count++;
        if (scan_count % 200 == 0) {
            std::cout << "[Mask Loading] Scanning: " << scan_count << " nodes, found " 
                      << total_frames << " frames so far..." << std::endl;
            std::cout.flush();
        }
        std::string node_name = (*frame_node).name();
        if (node_name.rfind("frame_", 0) == 0) {
            total_frames++;
        }
    }
    
    std::cout << "[Mask Loading] Found " << total_frames << " frames with mask data" << std::endl;
    std::cout << "[Mask Loading] Starting to load mask data (this may take a while for large files)..." << std::endl;
    std::cout << "[Mask Loading] Progress: [" << std::flush;
    
    int loaded_frames = 0;
    int last_progress = -1;
    int last_logged_frame = -1;
    
    // 重新遍历加载数据
    for (auto frame_node = root.begin(); frame_node != root.end(); ++frame_node) {
        std::string node_name = (*frame_node).name();
        if (node_name.rfind("frame_", 0) != 0) {
            continue;
        }

        int frame_id = 0;
        try {
            frame_id = std::stoi(node_name.substr(6));
        } catch (...) {
            continue;
        }

        FrameMaskData data;
        (*frame_node)["image_width"] >> data.image_width;
        (*frame_node)["image_height"] >> data.image_height;

        cv::FileNode classes_node = (*frame_node)["classes"];
        if (!classes_node.empty()) {
            for (auto cls_node : classes_node) {
                std::string class_name;
                cls_node["name"] >> class_name;
                if (class_name.empty()) {
                    continue;
                }
                FrameMaskData::ClassMask class_mask;
                std::vector<int> mask_values;
                cv::FileNode mask_node = cls_node["mask"];
                if (mask_node.empty()) {
                    continue;
                }
                mask_node >> mask_values;
                if (mask_values.empty()) {
                    continue;
                }
                int computed_area = 0;
                class_mask.mask = convertMask(mask_values, computed_area);
                class_mask.area = computed_area;
                if (!cls_node["score"].empty()) {
                    cls_node["score"] >> class_mask.score;
                }
                if (!cls_node["area"].empty()) {
                    cls_node["area"] >> class_mask.area;
                }
                // 读取颜色信息
                cv::FileNode color_node = cls_node["color"];
                if (!color_node.empty()) {
                    std::vector<int> color_vec;
                    color_node >> color_vec;
                    if (color_vec.size() >= 3) {
                        class_mask.color = {color_vec[0], color_vec[1], color_vec[2]};
                    }
                }
                data.class_masks[class_name] = std::move(class_mask);
            }
        } else {
            std::vector<int> soft_mask_values;
            std::vector<int> hard_mask_values;
            (*frame_node)["soft_mask"] >> soft_mask_values;
            (*frame_node)["hard_mask"] >> hard_mask_values;

            if (!soft_mask_values.empty()) {
                FrameMaskData::ClassMask cls;
                cls.mask = convertMask(soft_mask_values, cls.area);
                cls.score = 1.0f;
                data.class_masks["soft"] = cls;
            }
            if (!hard_mask_values.empty()) {
                FrameMaskData::ClassMask cls;
                cls.mask = convertMask(hard_mask_values, cls.area);
                cls.score = 0.8f;
                data.class_masks["hard"] = cls;
            }
        }

        if (data.class_masks.empty()) {
            continue;
        }

        if (data.image_width <= 0 || data.image_height <= 0) {
            size_t mask_size = data.class_masks.begin()->second.mask.size();
            if (mask_size > 0) {
                data.image_width = static_cast<int>(mask_size);
                data.image_height = 1;
            }
        }

        frame_mask_data_[frame_id] = std::move(data);
        loaded_frames++;
        
        // 每1%显示一次进度，每50帧输出一次日志
        int progress = total_frames > 0 ? (loaded_frames * 100) / total_frames : 0;
        if (progress != last_progress) {
            if (progress % 10 == 0) {
                std::cout << "=" << std::flush;
            } else if (progress % 5 == 0) {
                std::cout << "." << std::flush;
            }
            last_progress = progress;
        }
        
        // 每50帧输出一次详细日志（更频繁的反馈）
        if (loaded_frames % 50 == 0 || loaded_frames == total_frames) {
            std::cout << "\n[Mask Loading] Loaded " << loaded_frames << "/" << total_frames 
                      << " frames (" << progress << "%) - Frame " << frame_id << std::flush;
            if (loaded_frames < total_frames) {
                std::cout << "\n[Mask Loading] Progress: [" << std::flush;
                // 重新绘制进度条
                for (int i = 0; i < progress / 10; i++) {
                    std::cout << "=" << std::flush;
                }
                for (int i = 0; i < (progress % 10) / 5; i++) {
                    std::cout << "." << std::flush;
                }
            }
            last_logged_frame = frame_id;
        }
    }
    
    std::cout << "] 100%" << std::endl;
    std::cout << "[Mask Loading] Completed loading " << loaded_frames << " frames" << std::endl;
    std::cout.flush();

    masks_loaded_ = !frame_mask_data_.empty();
    if (masks_loaded_) {
        LOG(INFO) << "[Mask Loading] ✅ Successfully loaded " << frame_mask_data_.size()
                  << " frame mask(s) from: " << abs_path;
        
        // 显示第一帧的组织信息作为示例
        if (!frame_mask_data_.empty()) {
            auto first_frame_it = frame_mask_data_.begin();
            int first_frame_id = first_frame_it->first;
            LOG(INFO) << "[Mask Loading] Sample frame " << first_frame_id << " has " 
                      << first_frame_it->second.class_masks.size() << " tissue class(es):";
            for (const auto& cls_entry : first_frame_it->second.class_masks) {
                const std::string& cls_name = cls_entry.first;
                const auto& cls_mask = cls_entry.second;
                auto param_it = options_.tissue_params.find(cls_name);
                if (param_it != options_.tissue_params.end()) {
                    LOG(INFO) << "    - " << cls_name << " (area: " << cls_mask.area 
                              << " pixels) -> k=" << std::fixed << std::setprecision(2) << param_it->second.k
                              << ", b=" << std::fixed << std::setprecision(2) << param_it->second.b;
                } else {
                    LOG(INFO) << "    - " << cls_name << " (area: " << cls_mask.area 
                              << " pixels) -> (auto-guessed params)";
                }
            }
        }
        
        // 显示所有配置的组织参数
        if (!options_.tissue_params.empty()) {
            LOG(INFO) << "[Mask Loading] All configured tissue parameters:";
            for (const auto& param_entry : options_.tissue_params) {
                LOG(INFO) << "    - " << param_entry.first 
                          << ": k=" << std::fixed << std::setprecision(2) << param_entry.second.k
                          << ", b=" << std::fixed << std::setprecision(2) << param_entry.second.b;
            }
        } else {
            LOG(INFO) << "[Mask Loading] ⚠️  No tissue parameters configured in settings.yaml, will use auto-guessed values.";
        }
    } else {
        LOG(WARNING) << "[Mask Loading] ⚠️  No valid masks found in: " << abs_path;
    }
    LOG(INFO) << "========================================";
}

// 按需加载指定帧的 mask 数据（按帧保存模式）
bool Tracking::LoadFrameMaskDataFromFile(int frame_id) const {
    // 检查是否已加载
    if (frame_loaded_cache_.find(frame_id) != frame_loaded_cache_.end() && 
        frame_loaded_cache_.at(frame_id)) {
        return true;  // 已加载
    }
    
    // 检查该帧是否有数据
    if (available_frame_ids_.find(frame_id) == available_frame_ids_.end()) {
        return false;  // 该帧没有 mask 数据
    }
    
    // 构建文件路径
    std::string frame_file = frames_dir_ + "/frame_" + std::to_string(frame_id) + ".json";
    
    if (!std::filesystem::exists(frame_file)) {
        return false;
    }
    
    // 打开并解析 JSON 文件
    cv::FileStorage fs(frame_file, cv::FileStorage::READ | cv::FileStorage::FORMAT_AUTO);
    if (!fs.isOpened()) {
        return false;
    }
    
    std::string frame_key = "frame_" + std::to_string(frame_id);
    cv::FileNode frame_node = fs[frame_key];
    if (frame_node.empty()) {
        fs.release();
        return false;
    }
    
    auto convertMask = [](const std::vector<int>& src, int& area_out) {
        std::vector<uint8_t> dst;
        dst.reserve(src.size());
        area_out = 0;
        for (int v : src) {
            uint8_t val = static_cast<uint8_t>(v != 0);
            dst.push_back(val);
            area_out += val;
        }
        return dst;
    };
    
    FrameMaskData data;
    frame_node["image_width"] >> data.image_width;
    frame_node["image_height"] >> data.image_height;
    
    cv::FileNode classes_node = frame_node["classes"];
    if (!classes_node.empty()) {
        for (auto cls_node : classes_node) {
            std::string class_name;
            cls_node["name"] >> class_name;
            if (class_name.empty()) {
                continue;
            }
            FrameMaskData::ClassMask class_mask;
            std::vector<int> mask_values;
            cv::FileNode mask_node = cls_node["mask"];
            if (mask_node.empty()) {
                continue;
            }
            mask_node >> mask_values;
            if (mask_values.empty()) {
                continue;
            }
            int computed_area = 0;
            class_mask.mask = convertMask(mask_values, computed_area);
            class_mask.area = computed_area;
            if (!cls_node["score"].empty()) {
                cls_node["score"] >> class_mask.score;
            }
            if (!cls_node["area"].empty()) {
                cls_node["area"] >> class_mask.area;
            }
            // 读取颜色信息
            cv::FileNode color_node = cls_node["color"];
            if (!color_node.empty()) {
                std::vector<int> color_vec;
                color_node >> color_vec;
                if (color_vec.size() >= 3) {
                    class_mask.color = {color_vec[0], color_vec[1], color_vec[2]};
                }
            }
            data.class_masks[class_name] = std::move(class_mask);
        }
    } else {
        std::vector<int> soft_mask_values;
        std::vector<int> hard_mask_values;
        frame_node["soft_mask"] >> soft_mask_values;
        frame_node["hard_mask"] >> hard_mask_values;
        
        if (!soft_mask_values.empty()) {
            FrameMaskData::ClassMask cls;
            cls.mask = convertMask(soft_mask_values, cls.area);
            cls.score = 1.0f;
            data.class_masks["soft"] = cls;
        }
        if (!hard_mask_values.empty()) {
            FrameMaskData::ClassMask cls;
            cls.mask = convertMask(hard_mask_values, cls.area);
            cls.score = 0.8f;
            data.class_masks["hard"] = cls;
        }
    }
    
    if (data.class_masks.empty()) {
        fs.release();
        return false;
    }
    
    if (data.image_width <= 0 || data.image_height <= 0) {
        size_t mask_size = data.class_masks.begin()->second.mask.size();
        if (mask_size > 0) {
            data.image_width = static_cast<int>(mask_size);
            data.image_height = 1;
        }
    }
    
    // 构建 O(1) 查表
    data.BuildClassIndexMap();
    
    // 存储到缓存（frame_mask_data_ 和 frame_loaded_cache_ 已声明为 mutable）
    frame_mask_data_[frame_id] = std::move(data);
    frame_loaded_cache_[frame_id] = true;
    
    fs.release();
    return true;
}

// 获取异质粘弹性参数（创新点2）
bool Tracking::GetHeterogeneousViscoelasticParams(ID mappoint_id, int pixel_idx, float& k, float& b, std::string* class_name_out) const {
    auto set_default_kb = [&](float& k_out, float& b_out) {
        // 默认的粘弹性参数优先从配置文件中的 TissueParams["default"] 读取，
        // 若未配置该条目，则退化为硬编码的 1.0 / 0.5。
        auto it = options_.tissue_params.find("default");
        if (it != options_.tissue_params.end()) {
            k_out = it->second.k;
            b_out = it->second.b;
        } else {
            k_out = 1.0f;
            b_out = 0.5f;
        }
    };

    if (!masks_loaded_ || !options_.enable_heterogeneous_viscoelastic) {
        // 未启用异质粘弹性或尚未加载 mask：使用配置文件中的默认 k,b
        set_default_kb(k, b);
        return false;
    }
    
    if (last_export_frame_id_ < 0) {
        set_default_kb(k, b);
        return false;
    }

    // 按帧保存模式：按需加载
    if (use_per_frame_files_) {
        if (frame_loaded_cache_.find(last_export_frame_id_) == frame_loaded_cache_.end() ||
            !frame_loaded_cache_.at(last_export_frame_id_)) {
            if (!LoadFrameMaskDataFromFile(last_export_frame_id_)) {
                set_default_kb(k, b);
                return false;
            }
        }
    }
    
    auto frame_it = frame_mask_data_.find(last_export_frame_id_);
    if (frame_it == frame_mask_data_.end()) {
        set_default_kb(k, b);
        return false;
    }

    const FrameMaskData& frame_data = frame_it->second;
    if (frame_data.image_width <= 0 || frame_data.image_height <= 0) {
        k = 1.0f;
        b = 0.5f;
        return false;
    }

    if (pixel_idx < 0 || pixel_idx >= frame_data.image_width * frame_data.image_height) {
        k = 1.0f;
        b = 0.5f;
        return false;
    }

    const int total_pixels = frame_data.image_width * frame_data.image_height;

    // 使用 O(1) class_index_map 查表（如果已构建）
    if (!frame_data.class_index_map.empty() && 
        pixel_idx < static_cast<int>(frame_data.class_index_map.size())) {
        uint8_t cidx = frame_data.class_index_map[pixel_idx];
        if (cidx > 0 && cidx < frame_data.class_names_by_index.size()) {
            const std::string& class_name = frame_data.class_names_by_index[cidx];
            if (class_name_out) {
                *class_name_out = class_name;
            }
            auto param_it = options_.tissue_params.find(class_name);
            if (param_it != options_.tissue_params.end()) {
                k = param_it->second.k;
                b = param_it->second.b;
            } else {
                auto mask_it = frame_data.class_masks.find(class_name);
                float area_ratio = 0.f;
                float score = 0.f;
                if (mask_it != frame_data.class_masks.end()) {
                    if (total_pixels > 0 && mask_it->second.area > 0) {
                        area_ratio = static_cast<float>(mask_it->second.area) / static_cast<float>(total_pixels);
                    }
                    score = mask_it->second.score;
                }
                GuessTissueParams(area_ratio, score, k, b);
            }
            return true;
        }
    }
    
    if (class_name_out) {
        class_name_out->clear();
    }

    // 未能根据 mask 或组织类别找到参数时，回退到配置文件中的默认 k,b
    set_default_kb(k, b);
    return false;
} 

void Tracking::GuessTissueParams(float area_ratio, float score, float& k, float& b) const {
    // 简单启发式：面积大 -> 软，面积中等 -> 中等，面积小 -> 硬
    if (area_ratio > 0.4f) {
        k = 0.8f;
        b = 1.2f;
    } else if (area_ratio > 0.15f) {
        k = 1.1f;
        b = 0.9f;
    } else {
        k = 1.6f;
        b = 0.5f;
    }

    // 根据score微调（越高越软）
    if (score > 0.9f) {
        k *= 0.9f;
        b *= 1.1f;
    } else if (score < 0.5f) {
        k *= 1.1f;
        b *= 0.9f;
    }

    // 保持合理范围
    k = std::clamp(k, 0.5f, 2.5f);
    b = std::clamp(b, 0.2f, 1.5f);
}

void Tracking::RedetectFeatures(const cv::Mat &im, const cv::Mat &mask) {
    // 提取新的特征点
    vector<cv::KeyPoint> new_keypoints;
    ExtractFeatures(im, mask, new_keypoints);

    // 只保留前50个最好的特征点，避免过多
    if (new_keypoints.size() > 50) {
        // 按响应值排序
        std::sort(new_keypoints.begin(), new_keypoints.end(),
                  [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                      return a.response > b.response;
                  });
        new_keypoints.resize(50);
    }

    LOG(INFO) << "[特征重检测] 重新检测到新特征点: " << new_keypoints.size();

    // 将新特征点添加到当前帧，但标记为TRACKED状态（没有3D信息）
    for (const auto& kp : new_keypoints) {
        // 检查是否与现有特征点太近
        bool too_close = false;
        for (const auto& existing_kp : current_frame_->Keypoints()) {
            float dist = cv::norm(kp.pt - existing_kp.pt);
            if (dist < 10.0f) {  // 最小距离10像素
                too_close = true;
                break;
            }
        }

        if (!too_close) {
            current_frame_->InsertObservation(kp, Eigen::Vector3f::Zero(), -1, TRACKED);
        }
    }

    // 为KLT跟踪器设置新的参考图像
    if (!current_frame_->Keypoints().empty()) {
        klt_tracker_.SetReferenceImage(im, current_frame_->Keypoints());
    }

    LOG(INFO) << "[特征重检测] 当前总特征点数: " << current_frame_->Keypoints().size();
}

void Tracking::AddVisibleMapPointsToCurrentFrame(const cv::Mat& im) {
    // 将所有在当前帧视野内的地图点添加到当前帧中进行跟踪
    auto all_mappoints = map_->GetMapPoints();

    for (const auto &[mappoint_id, mappoint] : all_mappoints) {
        // 检查这个点是否已经在当前帧中
        if (current_frame_->LandmarkPosition(mappoint_id).ok()) {
            continue; // 已经在当前帧中，跳过
        }

        // 检查点是否活跃
        if (!mappoint->IsActive()) {
            continue;
        }

        // 检查生命周期，避免使用太新的点
        if (mappoint->GetLifetime() < 1) {
            continue;
        }

        // 将地图点投影到当前相机坐标系
        Eigen::Vector3f landmark_world_position = mappoint->GetLastWorldPosition();
        Eigen::Vector3f landmark_camera_position = current_frame_->CameraTransformationWorld() * landmark_world_position;

        // 检查点是否在相机前方
        if (landmark_camera_position.z() <= 0) {
            continue;
        }

        // 投影到图像平面
        cv::Point2f projected_point = calibration_->Project(landmark_camera_position);

        // 检查投影点是否为NaN或无效值
        if (std::isnan(projected_point.x) || std::isnan(projected_point.y) ||
            std::isinf(projected_point.x) || std::isinf(projected_point.y)) {
            continue;  // 跳过无效投影点
        }

        // 检查投影点是否在图像范围内
        if (projected_point.x < 0 || projected_point.x >= im.cols ||
            projected_point.y < 0 || projected_point.y >= im.rows) {
            continue;
        }

        // 检查世界坐标是否有效（不能为0或NaN）
        if (landmark_world_position.norm() < 1e-6 ||
            std::isnan(landmark_world_position.x()) ||
            std::isnan(landmark_world_position.y()) ||
            std::isnan(landmark_world_position.z())) {
            continue;  // 跳过无效的地图点
        }

        // 检查是否与现有特征点太近（避免重复）
        bool too_close = false;
        for (const auto& existing_kp : current_frame_->Keypoints()) {
            float dist = cv::norm(projected_point - existing_kp.pt);
            if (dist < 15.0f) {  // 最小距离15像素
                too_close = true;
                break;
            }
        }

        if (!too_close) {
            // 创建特征点并添加到当前帧
            cv::KeyPoint keypoint(projected_point, 1.0f);
            keypoint.class_id = mappoint->GetKeyPointId();  // 设置正确的keypoint ID
            current_frame_->InsertObservation(keypoint, landmark_world_position, mappoint_id, TRACKED_WITH_3D);
        }
    }

    LOG(INFO) << "[可见点添加] 添加可见地图点到当前帧，当前总特征点: " << current_frame_->Keypoints().size();
}

void Tracking::SupplementFeaturesForTriangulation(const cv::Mat &im, const cv::Mat &mask) {
    // 提取新的特征点用于三角化
    vector<cv::KeyPoint> new_keypoints;
    ExtractFeatures(im, mask, new_keypoints);

    // 按响应值排序，保留最好的特征点
    std::sort(new_keypoints.begin(), new_keypoints.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                  return a.response > b.response;
              });

    // 只保留高质量的特征点：响应值大于中位数
    if (new_keypoints.size() > 10) {
        float median_response = new_keypoints[new_keypoints.size() / 2].response;
        vector<cv::KeyPoint> high_quality_keypoints;

        for (const auto& kp : new_keypoints) {
            if (kp.response >= median_response) {
                high_quality_keypoints.push_back(kp);
            }
        }

        // 限制数量，避免过多
        const int max_new_features = 30;
        if (high_quality_keypoints.size() > max_new_features) {
            high_quality_keypoints.resize(max_new_features);
        }

        new_keypoints = high_quality_keypoints;
    }

    if (new_keypoints.size() > 0) {
        LOG(INFO) << "[特征补充] 补充特征点用于三角化，数量: " << new_keypoints.size();
    }

    // 将新特征点添加到当前帧，标记为TRACKED状态（没有3D信息）
    for (const auto& kp : new_keypoints) {
        // 检查是否与现有特征点太近
        bool too_close = false;
        for (const auto& existing_kp : current_frame_->Keypoints()) {
            float dist = cv::norm(kp.pt - existing_kp.pt);
            if (dist < 15.0f) {  // 最小距离15像素，比关键帧中的更宽松
                too_close = true;
                break;
            }
        }

        if (!too_close) {
            current_frame_->InsertObservation(kp, Eigen::Vector3f::Zero(), -1, TRACKED);
        }
    }

    // 为KLT跟踪器设置新的参考图像
    if (!current_frame_->Keypoints().empty()) {
        klt_tracker_.SetReferenceImage(im, current_frame_->Keypoints());
    }
}

void Tracking::PerformActiveTriangulation() {
    if (!mapper_) {
        LOG(WARNING) << "No mapper available for active triangulation";
        return;
    }

    LOG(INFO) << "Performing active triangulation to generate new 3D points...";

    // 调用映射模块的三角化功能
    mapper_->DoMapping();

    // 更新刚刚三角化的点状态
    UpdateTriangulatedPoints();

    int new_3d_points = current_frame_->GetKeypointsWithStatus({TRACKED_WITH_3D}).size();
    LOG(INFO) << "Active triangulation completed. Current 3D points: " << new_3d_points;
}