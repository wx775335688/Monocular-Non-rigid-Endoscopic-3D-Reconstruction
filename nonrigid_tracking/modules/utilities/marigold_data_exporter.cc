#include "marigold_data_exporter.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace nrslam {

/**
 * @brief MarigoldDataExporter类的构造函数
 * @param options 包含数据导出配置选项的常量引用
 * 
 * 该构造函数负责初始化Marigold数据导出器，创建必要的输出目录结构，
 * 并根据配置选项初始化日志文件。
 */
MarigoldDataExporter::MarigoldDataExporter(const Options& options) : options_(options) {
    // 创建输出目录结构
    EnsureDirectoryExists(options_.output_dir);
    EnsureDirectoryExists(options_.output_dir + "rgb_images/");
    EnsureDirectoryExists(options_.output_dir + "sparse_depth/");
    EnsureDirectoryExists(options_.output_dir + "ddg_constraints/");
    EnsureDirectoryExists(options_.output_dir + "viscous_constraints/");
    EnsureDirectoryExists(options_.output_dir + "tracking_stats/");
    EnsureDirectoryExists(options_.output_dir + "camera_params/");
    EnsureDirectoryExists(options_.output_dir + "metadata/");
    
    // 初始化日志
    if (!options_.log_file.empty()) {
        log_file_.open(options_.log_file, std::ios::app);
        log_file_ << "Marigold Data Exporter initialized." << std::endl;
        log_file_ << "Output directory: " << options_.output_dir << std::endl;
    }
}

/**
 * @brief 导出当前帧的数据包，包含图像、相机内参以及根据配置选项导出的其他数据（如稀疏深度、约束信息等）。
 *
 * 该函数将当前帧的相关数据打包成 FrameDataPackage 结构，用于后续处理或保存。包括 RGB 图像、
 * 相机内参矩阵、图像尺寸等基础信息，并根据配置选项决定是否导出稀疏深度图、DDG 约束、
 * 粘性弹性约束以及跟踪统计信息等。
 *
 * @param rgb_image 当前帧的彩色图像（BGR 或 RGB 格式）
 * @param current_frame 当前帧对象，包含关键点、位姿等信息
 * @param map 地图对象的智能指针，用于提取地图相关约束
 * @param frame_id 当前帧的唯一标识符
 * @param previous_frame 上一帧的指针，用于计算帧间约束（可为空）
 * @param klt_tracker LK 光流跟踪器指针，用于获取跟踪质量统计数据（可为空）
 * @return 返回封装好的 FrameDataPackage 数据结构
 */
MarigoldDataExporter::FrameDataPackage MarigoldDataExporter::ExportFrameData(
    const cv::Mat& rgb_image, Frame& current_frame, std::shared_ptr<Map> map,
    int frame_id, Frame* previous_frame, 
    const LucasKanadeTracker* klt_tracker) {
    
    FrameDataPackage data_package;
    data_package.frame_id = frame_id;
    data_package.rgb_image = rgb_image.clone();
    
    // 获取相机内参并转换为 OpenCV 的 3x3 矩阵格式
    try {
        auto calibration = current_frame.GetCalibration();
        auto intrinsics_matrix = calibration->ToIntrinsicsMatrix();
        data_package.camera_intrinsics = (cv::Mat_<double>(3,3) <<
            intrinsics_matrix(0,0), intrinsics_matrix(0,1), intrinsics_matrix(0,2),
            intrinsics_matrix(1,0), intrinsics_matrix(1,1), intrinsics_matrix(1,2),
            intrinsics_matrix(2,0), intrinsics_matrix(2,1), intrinsics_matrix(2,2)
        );
        
        // 存储图像尺寸
        data_package.image_height = rgb_image.rows;
        data_package.image_width = rgb_image.cols;
        
        // 缓存图像尺寸用于后续处理
        current_image_height_ = rgb_image.rows;
        current_image_width_ = rgb_image.cols;
        
        std::cout << "Exporting frame " << frame_id 
                  << " with " << current_frame.GetKeypointsWithStatus({TRACKED_WITH_3D}).size() 
                  << " tracked features" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error getting camera calibration: " << e.what() << std::endl;
        return data_package;  // 出错时返回空数据包
    }
    
    // 根据配置选项导出附加数据组件
    if (options_.save_sparse_depth) {
        data_package.sparse_depth = GenerateSparseDepthMap(current_frame);
    }
    
    DDGConstraints ddg_cache;
    if (options_.save_ddg_constraints) {
        ddg_cache = ExtractDDGConstraints(current_frame, map);
        data_package.ddg_constraints = ddg_cache;
    } else if (options_.save_viscous_constraints) {
        // 即使不保存DDG文件，也计算一次以确保与粘性权重一致
        ddg_cache = ExtractDDGConstraints(current_frame, map);
    }
    
    if (options_.save_viscous_constraints && previous_frame) {
        data_package.viscous_constraints = ComputeViscousElasticConstraints(current_frame, *previous_frame, &ddg_cache);
    }
    
    if (options_.save_tracking_confidence && klt_tracker) {
        data_package.tracking_stats = ComputeTrackingStatistics(current_frame, klt_tracker);
    }
    
    return data_package;
}

void MarigoldDataExporter::SaveDataPackage(const FrameDataPackage& data_package, int frame_id) {
    std::string frame_str = std::to_string(frame_id);
    
    // 保存RGB图像
    if (options_.save_rgb_images && !data_package.rgb_image.empty()) {
        std::string rgb_path = options_.output_dir + "rgb_images/frame_" + frame_str + ".png";
        cv::imwrite(rgb_path, data_package.rgb_image);
    }
    
    // 保存稀疏深度图
    if (options_.save_sparse_depth && !data_package.sparse_depth.empty()) {
        std::string depth_path = options_.output_dir + "sparse_depth/frame_" + frame_str + "_sparse_depth.npy";
        
        // 获取深度图数据
        cv::Mat depth_map = data_package.sparse_depth;
        if (depth_map.type() == CV_32F) {
            SaveFloatArrayAsNpy((float*)depth_map.data, depth_map.rows, depth_map.cols, depth_path);
        } else {
            std::cerr << "Warning: Depth map is not CV_32F format" << std::endl;
        }
        // 额外保存稀疏深度统计信息，便于调试（非破坏性）
        try {
            std::string stats_path = options_.output_dir + "sparse_depth/frame_" + frame_str + "_sparse_stats.txt";
            std::ofstream stats_file(stats_path);
            if (stats_file.is_open()) {
                int H = depth_map.rows, W = depth_map.cols;
                double min_val = std::numeric_limits<double>::infinity();
                double max_val = -std::numeric_limits<double>::infinity();
                double sum = 0.0;
                double sumsq = 0.0;
                long count = 0;
                for (int r = 0; r < H; ++r) {
                    const float* row_ptr = depth_map.ptr<float>(r);
                    for (int c = 0; c < W; ++c) {
                        float v = row_ptr[c];
                        if (std::isfinite(v) && v > 0.0f) {
                            if (v < min_val) min_val = v;
                            if (v > max_val) max_val = v;
                            sum += v;
                            sumsq += static_cast<double>(v) * static_cast<double>(v);
                            ++count;
                        }
                    }
                }
                double mean = count > 0 ? sum / count : 0.0;
                double var = (count > 1) ? (sumsq / count - mean * mean) : 0.0;
                double stddev = var > 0.0 ? std::sqrt(var) : 0.0;
                stats_file << "nonzero_count: " << count << "\n";
                stats_file << "min: " << (count>0 ? min_val : 0.0) << "\n";
                stats_file << "max: " << (count>0 ? max_val : 0.0) << "\n";
                stats_file << "mean: " << mean << "\n";
                stats_file << "stddev: " << stddev << "\n";
                stats_file.close();
            }
        } catch (...) {
            // 忽略统计保存错误，不影响主流程
        }
    }
    
    // 保存DDG约束
    if (options_.save_ddg_constraints) {
        SaveIDPairsToText(data_package.ddg_constraints.connections,
                         options_.output_dir + "ddg_constraints/frame_" + frame_str + "_connections.txt");
        SaveFloatArrayToText(data_package.ddg_constraints.initial_distances,
                              options_.output_dir + "ddg_constraints/frame_" + frame_str + "_distances.txt");
        SaveFloatArrayToText(data_package.ddg_constraints.connection_weights,
                              options_.output_dir + "ddg_constraints/frame_" + frame_str + "_weights.txt");
        SaveFloatArrayToText(data_package.ddg_constraints.max_distances,
                              options_.output_dir + "ddg_constraints/frame_" + frame_str + "_max_distances.txt");
        SaveHashMapToText(data_package.ddg_constraints.mappoint_to_pixel,
                         options_.output_dir + "ddg_constraints/frame_" + frame_str + "_mappoint_to_pixel.txt");
    }
    
    // 保存粘弹性约束
    if (options_.save_viscous_constraints) {
        SaveEigenVectorArrayToText(data_package.viscous_constraints.deformation_offsets,
                                    options_.output_dir + "viscous_constraints/frame_" + frame_str + "_offsets.txt");
        SaveFloatArrayToText(data_package.viscous_constraints.viscous_weights,
                              options_.output_dir + "viscous_constraints/frame_" + frame_str + "_viscous_weights.txt");
        SaveFloatArrayToText(data_package.viscous_constraints.ssim_confidence,
                              options_.output_dir + "viscous_constraints/frame_" + frame_str + "_ssim_confidence.txt");
        
        // 保存弹性系数（单个值）
        std::ofstream elastic_file(options_.output_dir + "viscous_constraints/frame_" + frame_str + "_elastic_coefficient.txt");
        if (elastic_file.is_open()) {
            elastic_file << std::fixed << std::setprecision(6) << data_package.viscous_constraints.elastic_coefficient << "\n";
            elastic_file.close();
        }
    }
    
    // 保存跟踪统计
    if (options_.save_tracking_confidence) {
        SaveFloatArrayToText(data_package.tracking_stats.per_point_ssim,
                             options_.output_dir + "tracking_stats/frame_" + frame_str + "_per_point_ssim.txt");
    }
    
    // 保存相机参数
    if (options_.save_camera_params && !data_package.camera_intrinsics.empty()) {
        std::string cam_path = options_.output_dir + "camera_params/frame_" + frame_str + "_camera_matrix.xml";
        cv::FileStorage fs(cam_path, cv::FileStorage::WRITE);
        fs << "camera_matrix" << data_package.camera_intrinsics;
        fs.release();
    }
    
    // 保存元数据
    SaveMetadata(data_package, frame_id);
}

void MarigoldDataExporter::SaveMetadata(const FrameDataPackage& data_package, int frame_id) {
    std::string metadata_path = options_.output_dir + "metadata/frame_" + std::to_string(frame_id) + "_metadata.json";
    std::ofstream metadata_file(metadata_path);
    
    metadata_file << "{\n";
    metadata_file << "  \"frame_id\": " << frame_id << ",\n";
    metadata_file << "  \"image_height\": " << data_package.image_height << ",\n";
    metadata_file << "  \"image_width\": " << data_package.image_width << ",\n";
    
    // 相机内参矩阵
    if (!data_package.camera_intrinsics.empty()) {
        metadata_file << "  \"camera_matrix\": [\n";
        for (int i = 0; i < 3; ++i) {
            metadata_file << "    [";
            for (int j = 0; j < 3; ++j) {
                metadata_file << data_package.camera_intrinsics.at<double>(i, j);
                if (j < 2) metadata_file << ", ";
            }
            metadata_file << "]";
            if (i < 2) metadata_file << ",";
            metadata_file << "\n";
        }
        metadata_file << "  ],\n";
    }
    
    // DDG统计信息
    metadata_file << "  \"ddg_stats\": {\n";
    metadata_file << "    \"num_connections\": " << data_package.ddg_constraints.connections.size() << ",\n";
    metadata_file << "    \"num_map_points\": " << data_package.ddg_constraints.mappoint_to_pixel.size() << "\n";
    metadata_file << "  },\n";
    
    // 跟踪统计信息
    metadata_file << "  \"tracking_stats\": {\n";
    metadata_file << "    \"mean_ssim\": " << data_package.tracking_stats.mean_ssim << ",\n";
    metadata_file << "    \"std_ssim\": " << data_package.tracking_stats.std_ssim << ",\n";
    metadata_file << "    \"num_tracked_features\": " << data_package.tracking_stats.per_point_ssim.size() << "\n";
    metadata_file << "  }\n";
    
    metadata_file << "}\n";
    metadata_file.close();
}

MarigoldDataExporter::DDGConstraints MarigoldDataExporter::ExtractDDGConstraints(
    const Frame& frame, std::shared_ptr<Map> map) {
    
    DDGConstraints constraints;
    
    // 获取当前帧中有3D坐标的特征点和位置
    const auto& keypoints_3d = frame.GetKeypointsWithStatus({TRACKED_WITH_3D});
    const auto& landmark_positions = frame.GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
    const auto& index_to_mappoint_id = frame.IndexToMapPointId();
    
    if (keypoints_3d.empty() || landmark_positions.empty()) {
        return constraints;
    }
    
    // 收集有效的mappoint IDs和对应的索引
    std::vector<ID> valid_mappoint_ids;
    std::vector<int> valid_indices;
    std::vector<Eigen::Vector3f> valid_positions;
    std::vector<cv::KeyPoint> valid_keypoints;
    
    // 使用哈希映射以提高查找效率
    absl::flat_hash_map<int, ID> index_to_id_map(index_to_mappoint_id.begin(), index_to_mappoint_id.end());

    for (size_t i = 0; i < keypoints_3d.size(); ++i) {
        auto it = index_to_id_map.find(static_cast<int>(i));
        if (it != index_to_id_map.end()) {
            valid_mappoint_ids.push_back(it->second);
            valid_indices.push_back(static_cast<int>(i));
            valid_positions.push_back(landmark_positions[i]);
            valid_keypoints.push_back(keypoints_3d[i]);
            int linear_index = ComputeLinearPixelIndex(keypoints_3d[i]);
            if (linear_index >= 0) {
                constraints.mappoint_to_pixel[it->second] = linear_index;
            }
        }
    }
    
    // 改进的连接策略：使用基于空间近邻和图像平面近邻的双重过滤
    // 1. 首先计算所有点的图像坐标
    std::vector<cv::Point2f> image_points;
    for (const auto& kp : valid_keypoints) {
        image_points.push_back(kp.pt);
    }
    
    // 2. 为每个点找到最近的邻居，限制连接数量
    for (size_t i = 0; i < valid_mappoint_ids.size(); ++i) {
        // 为每个点找到最多10个最近邻
        std::vector<std::pair<float, size_t>> neighbors;
        
        for (size_t j = 0; j < valid_mappoint_ids.size(); ++j) {
            if (i == j) continue;
            
            const Eigen::Vector3f& pos1 = valid_positions[i];
            const Eigen::Vector3f& pos2 = valid_positions[j];
            
            // 空间距离
            float spatial_distance = (pos1 - pos2).norm();
            
            // 图像平面距离
            float image_distance = cv::norm(image_points[i] - image_points[j]);
            
            // 双重过滤：空间距离和图像距离都要满足条件
            if (spatial_distance > 1e-6 && spatial_distance <= options_.distance_threshold * 1.5 &&
                image_distance < 50.0) { // 图像平面上不超过50像素
                neighbors.emplace_back(spatial_distance, j);
            }
        }
        
        // 按距离排序，取最近的10个邻居
        std::sort(neighbors.begin(), neighbors.end());
        size_t max_neighbors = std::min(10ul, neighbors.size());
        
        for (size_t k = 0; k < max_neighbors; ++k) {
            size_t j = neighbors[k].second;
            float spatial_distance = neighbors[k].first;
            
            // 确保只添加一次连接（i < j）
            if (i < j) {
                constraints.connections.emplace_back(valid_mappoint_ids[i], valid_mappoint_ids[j]);
                constraints.initial_distances.push_back(spatial_distance);
                
                // 改进的权重计算：考虑空间距离和图像平面距离
                float image_distance = cv::norm(image_points[i] - image_points[j]);
                float spatial_weight = ComputeInterpolationWeight(spatial_distance, options_.sigma);
                float image_weight = std::exp(-image_distance / 20.0); // 图像距离权重，20像素为衰减因子
                float combined_weight = (spatial_weight + image_weight) / 2.0;
                
                constraints.connection_weights.push_back(combined_weight);
                
                // 改进的最大距离计算：基于权重动态调整
                float max_distance = spatial_distance * (1.0 + 0.5 * (1.0 - combined_weight));
                constraints.max_distances.push_back(max_distance);
            }
        }
    }
    
    return constraints;
}

/**
 * @brief 计算粘弹性约束参数，这些参数描述了连接（边）在连续帧之间的变化。
 *
 * 该函数的核心逻辑是：
 * 1. 找出在当前帧和上一帧中都存在的“持续连接”（边）。
 * 2. 对于每个持续连接 (i, j)，计算其长度在两帧之间的变化。
 * 3. 基于长度变化计算粘性权重 b_ij^t，变化越小，权重越高。
 * 4. 同时，计算连接中点 (i, j) 的平均变形偏移。
 *
 * @param current_frame 当前帧对象
 * @param previous_frame 上一帧对象
 * @return 返回一个 ViscousElasticConstraints 结构，其中所有数据都与“持续连接”一一对应。
 */
/**
 * @brief 计算粘弹性约束参数，这些参数描述了地图点在连续帧之间的变化。
 *
 * 该函数的核心逻辑是：
 * 1. 以当前帧所有可见的地图点为基准进行遍历。
 * 2. 对于每个地图点，查找其在上一帧的位置。
 * 3. 如果点在两帧中都存在，则计算其位移（变形偏移量）和基于位移的置信度。
 * 4. 如果点是新出现的，则偏移量为零，置信度为默认值。
 *
 * @param current_frame 当前帧对象
 * @param previous_frame 上一帧对象
 * @return 返回一个 ViscousElasticConstraints 结构，其中所有数据都与当前帧的可见地图点一一对应。
 */
MarigoldDataExporter::ViscousElasticConstraints MarigoldDataExporter::ComputeViscousElasticConstraints(
    const Frame& current_frame, const Frame& previous_frame, const DDGConstraints* ddg_constraints) {
    
    ViscousElasticConstraints constraints;

    // 1. 获取当前帧所有可见的地图点及其在数组中的索引
    const auto& current_keypoints_3d = current_frame.GetKeypointsWithStatus({TRACKED_WITH_3D});
    const auto& current_index_to_id = current_frame.IndexToMapPointId();
    const auto& current_landmark_positions = current_frame.GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});

    // 2. 为了快速查找，创建上一帧从地图点ID到其位置的映射
    const auto& previous_landmark_positions = previous_frame.GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
    const auto& previous_index_to_id = previous_frame.IndexToMapPointId();
    absl::flat_hash_map<ID, Eigen::Vector3f> previous_positions_map;
    for (size_t i = 0; i < previous_landmark_positions.size(); ++i) {
        auto it = previous_index_to_id.find(static_cast<int>(i));
        if (it != previous_index_to_id.end()) {
            previous_positions_map[it->second] = previous_landmark_positions[i];
        }
    }

    // 3. 以当前帧的可见点为基准进行遍历，确保数量和顺序一致
    for (size_t i = 0; i < current_keypoints_3d.size(); ++i) {
        auto it_curr = current_index_to_id.find(static_cast<int>(i));
        if (it_curr == current_index_to_id.end()) {
            continue; // 如果当前关键点没有关联的地图点，则跳过
        }
        
        ID current_id = it_curr->second;
        const Eigen::Vector3f& current_pos = current_landmark_positions[i];
        
        Eigen::Vector3f offset = Eigen::Vector3f::Zero();
        float ssim_conf = 1.0f; // 默认置信度为1.0（对于新点）

        // 查找该点在上一帧是否存在
        auto it_prev = previous_positions_map.find(current_id);
        if (it_prev != previous_positions_map.end()) {
            // 点在两帧都存在，计算偏移和置信度
            const Eigen::Vector3f& previous_pos = it_prev->second;
            offset = current_pos - previous_pos;
            ssim_conf = std::exp(-offset.norm() * 2.0f); // 偏移越小，置信度越高
        }
        
        constraints.deformation_offsets.push_back(offset);
        constraints.ssim_confidence.push_back(ssim_conf);
    }
    
    // 重新计算粘性权重，使其与导出的connections一一对应
    DDGConstraints local_ddg;
    const DDGConstraints* ddg_src = ddg_constraints;
    if (!ddg_src) {
        local_ddg = ExtractDDGConstraints(current_frame, nullptr);
        ddg_src = &local_ddg;
    }
    const auto& current_connections = ddg_src->connections;

    for (const auto& conn : current_connections) {
        ID id1 = conn.first;
        ID id2 = conn.second;

        auto it_curr1_pos = std::find_if(current_index_to_id.begin(), current_index_to_id.end(), 
                                         [id1](const auto& pair){ return pair.second == id1; });
        auto it_curr2_pos = std::find_if(current_index_to_id.begin(), current_index_to_id.end(),
                                         [id2](const auto& pair){ return pair.second == id2; });

        auto it_prev1 = previous_positions_map.find(id1);
        auto it_prev2 = previous_positions_map.find(id2);

        if (it_curr1_pos != current_index_to_id.end() && it_curr2_pos != current_index_to_id.end() &&
            it_prev1 != previous_positions_map.end() && it_prev2 != previous_positions_map.end()) {
            
            const Eigen::Vector3f& curr_pos1 = current_landmark_positions[it_curr1_pos->first];
            const Eigen::Vector3f& curr_pos2 = current_landmark_positions[it_curr2_pos->first];
            const Eigen::Vector3f& prev_pos1 = it_prev1->second;
            const Eigen::Vector3f& prev_pos2 = it_prev2->second;

            float dist_curr = (curr_pos1 - curr_pos2).norm();
            float dist_prev = (prev_pos1 - prev_pos2).norm();
            float dist_change = std::abs(dist_curr - dist_prev);

            float viscous_weight = std::exp(-dist_change / options_.sigma);
            constraints.viscous_weights.push_back(viscous_weight);
        } else {
            // 如果连接不是持续存在的，则添加一个默认权重
            constraints.viscous_weights.push_back(0.0f);
        }
    }

    // 设置弹性系数
    float avg_offset_norm = 0.0f;
    if (!constraints.deformation_offsets.empty()) {
        for (const auto& offset : constraints.deformation_offsets) {
            avg_offset_norm += offset.norm();
        }
        avg_offset_norm /= constraints.deformation_offsets.size();
        
        constraints.elastic_coefficient = std::exp(-avg_offset_norm * options_.sigma);
        constraints.elastic_coefficient = std::max(0.1f, std::min(2.0f, constraints.elastic_coefficient));
    }
    
    return constraints;
}

MarigoldDataExporter::TrackingStatistics MarigoldDataExporter::ComputeTrackingStatistics(
    const Frame& frame, const LucasKanadeTracker* klt_tracker) {
    
    TrackingStatistics stats;
    
    // 从KLT跟踪器获取SSIM值
    if (klt_tracker) {
        stats.per_point_ssim = klt_tracker->GetLastSSIMValues();
        
        // 计算统计指标
        if (!stats.per_point_ssim.empty()) {
            float sum = 0.0f;
            for (float ssim : stats.per_point_ssim) {
                sum += ssim;
            }
            stats.mean_ssim = sum / stats.per_point_ssim.size();
            
            // 计算标准差
            float variance = 0.0f;
            for (float ssim : stats.per_point_ssim) {
                variance += (ssim - stats.mean_ssim) * (ssim - stats.mean_ssim);
            }
            stats.std_ssim = std::sqrt(variance / stats.per_point_ssim.size());
        } else {
            stats.mean_ssim = 0.0f;
            stats.std_ssim = 0.0f;
        }
    }
    
    return stats;
}

cv::Mat MarigoldDataExporter::GenerateSparseDepthMap(const Frame& frame) {
    // 使用动态图像尺寸（在ExportFrameData中已经缓存）
    cv::Mat depth_map = cv::Mat::zeros(current_image_height_, current_image_width_, CV_32F);
    
    const auto& keypoints_3d = frame.GetKeypointsWithStatus({TRACKED_WITH_3D});
    const auto& landmark_positions = frame.GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
    
    // 直接使用T_cw（世界 -> 相机）
    Eigen::Matrix4f world_to_camera = frame.CameraTransformationWorld().matrix();
    
    for (size_t i = 0; i < keypoints_3d.size() && i < landmark_positions.size(); ++i) {
        const cv::KeyPoint& keypoint = keypoints_3d[i];
        const Eigen::Vector3f& world_position = landmark_positions[i];
        
        // 将世界坐标转换为相机坐标系
        Eigen::Vector4f world_pos_homogeneous(world_position.x(), world_position.y(), world_position.z(), 1.0f);
        Eigen::Vector4f camera_pos_homogeneous = world_to_camera * world_pos_homogeneous;
        Eigen::Vector3f camera_pos = camera_pos_homogeneous.head<3>();
        
        // 计算真实深度（沿光轴方向的正向距离）
        float depth = camera_pos.z();
        
        if (depth > 0.01f) { // 过滤掉太近或落在相机后方的点
            int linear_index = ComputeLinearPixelIndex(keypoint);
            if (linear_index >= 0) {
                int x = linear_index % depth_map.cols;
                int y = linear_index / depth_map.cols;
                depth_map.at<float>(y, x) = depth;  // 使用光轴深度，单位为米
            }
        }
    }
    
    return depth_map;  // 直接返回32位浮点数矩阵
}

cv::Mat MarigoldDataExporter::CreateGeometricTopologyFeatures(
    const Frame& frame, const DDGConstraints& ddg) {
    
    // 使用动态图像尺寸
    cv::Mat features = cv::Mat::zeros(current_image_height_, current_image_width_, CV_32FC3);
    
    const auto& keypoints_3d = frame.GetKeypointsWithStatus({TRACKED_WITH_3D});
    const auto& index_to_id = frame.IndexToMapPointId();
    
    // 为每个特征点创建几何拓扑特征
    for (size_t i = 0; i < keypoints_3d.size(); ++i) {
        const cv::KeyPoint& kp = keypoints_3d[i];
        
        int x = static_cast<int>(kp.pt.x);
        int y = static_cast<int>(kp.pt.y);
        
        if (x >= 0 && x < features.cols && y >= 0 && y < features.rows) {
            // 从索引获取mappoint ID
            auto it = index_to_id.find(static_cast<int>(i));
            if (it == index_to_id.end()) continue;
            ID mp_id = it->second;
            
            // 计算该点的连接数
            int connection_count = 0;
            float avg_weight = 0.0f;
            float avg_distance = 0.0f;
            
            for (size_t j = 0; j < ddg.connections.size(); ++j) {
                if (ddg.connections[j].first == mp_id || ddg.connections[j].second == mp_id) {
                    connection_count++;
                    avg_weight += ddg.connection_weights[j];
                    avg_distance += ddg.initial_distances[j];
                }
            }
            
            if (connection_count > 0) {
                avg_weight /= connection_count;
                avg_distance /= connection_count;
            }
            
            // 存储几何拓扑特征 [连接数, 平均权重, 平均距离]
            features.at<cv::Vec3f>(y, x) = cv::Vec3f(
                static_cast<float>(connection_count),
                avg_weight,
                avg_distance
            );
        }
    }
    
    return features;
}

float MarigoldDataExporter::ComputeInterpolationWeight(float distance, float sigma) {
    return std::exp(-(distance * distance) / (2.0f * sigma * sigma));
}

void MarigoldDataExporter::EnsureDirectoryExists(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
    }
}

void MarigoldDataExporter::SaveFloatArrayToText(const std::vector<float>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        // 保存数组大小作为第一行
        file << data.size() << "\n";
        // 保存每个浮点数，每行一个，保持较高精度
        for (const float& value : data) {
            file << std::fixed << std::setprecision(6) << value << "\n";
        }
        file.close();
    }
}

void MarigoldDataExporter::SaveEigenVectorArrayToText(const std::vector<Eigen::Vector3f>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        // 保存向量数组大小作为第一行
        file << data.size() << "\n";
        // 保存每个3D向量，格式: x y z（每行一个向量）
        for (const auto& vec : data) {
            file << std::fixed << std::setprecision(6) 
                 << vec.x() << " " << vec.y() << " " << vec.z() << "\n";
        }
        file.close();
    }
}

void MarigoldDataExporter::SaveIDPairsToText(const std::vector<std::pair<ID, ID>>& pairs, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        // 保存对数作为第一行
        file << pairs.size() << "\n";
        // 保存每个ID对，格式: id1 id2（每行一对）
        for (const auto& pair : pairs) {
            file << pair.first << " " << pair.second << "\n";
        }
        file.close();
    }
}

void MarigoldDataExporter::SaveHashMapToText(const absl::flat_hash_map<ID, int>& hash_map, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        // 保存映射数量作为第一行
        file << hash_map.size() << "\n";
        // 保存每个映射，格式: map_point_id pixel_index（每行一对）
        for (const auto& pair : hash_map) {
            file << pair.first << " " << pair.second << "\n";
        }
        file.close();
    }
}

void MarigoldDataExporter::SaveFloatArrayAsNpy(const float* data, int height, int width, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }

    // numpy .npy文件头格式
    // 魔数
    const char magic[] = "\x93NUMPY";
    file.write(magic, 6);

    // 主版本号和次版本号
    char version[2] = {1, 0};
    file.write(version, 2);

    // 创建头部描述字符串
    std::string dtype = "'<f4'";  // 小端序32位浮点数
    bool fortran_order = false;   // C顺序（行优先）
    std::string shape = "(" + std::to_string(height) + ", " + std::to_string(width) + ")";
    
    std::string header = "{'descr': " + dtype + ", 'fortran_order': " + 
                        (fortran_order ? "True" : "False") + ", 'shape': " + shape + ", }";
    
    // 计算填充，使总头部长度为16字节的倍数
    size_t header_len = header.length();
    size_t total_header_len = header_len + 10; // 6字节魔数 + 2字节版本 + 2字节头长度
    size_t padding = (16 - (total_header_len % 16)) % 16;
    
    // 添加填充空格
    for (size_t i = 0; i < padding; ++i) {
        header += ' ';
    }
    header += '\n';
    
    // 写入头部长度（小端序）
    uint16_t header_size = static_cast<uint16_t>(header.length());
    file.write(reinterpret_cast<const char*>(&header_size), 2);
    
    // 写入头部描述
    file.write(header.c_str(), header.length());
    
    // 写入数据（已经是浮点数格式，直接写入）
    size_t data_size = static_cast<size_t>(height) * width * sizeof(float);
    file.write(reinterpret_cast<const char*>(data), data_size);
    
    file.close();
    
    std::cout << "Saved sparse depth map to " << filename 
              << " (shape: " << height << "x" << width << ")" << std::endl;
}

int MarigoldDataExporter::ComputeLinearPixelIndex(const cv::KeyPoint& keypoint) const {
    if (current_image_width_ <= 0 || current_image_height_ <= 0) {
        return -1;
    }
    
    int x = static_cast<int>(std::round(keypoint.pt.x));
    int y = static_cast<int>(std::round(keypoint.pt.y));
    
    if (x < 0 || x >= current_image_width_ || y < 0 || y >= current_image_height_) {
        return -1;
    }
    
    return y * current_image_width_ + x;
}

} // namespace nrslam