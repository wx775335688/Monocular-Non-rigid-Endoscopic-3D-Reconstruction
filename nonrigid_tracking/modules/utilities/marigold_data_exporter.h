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

#ifndef NRSLAM_MARIGOLD_DATA_EXPORTER_H
#define NRSLAM_MARIGOLD_DATA_EXPORTER_H

#include "map/frame.h"
#include "map/map.h"
#include "map/regularization_graph.h"
#include "calibration/camera_model.h"
#include "matching/lucas_kanade_tracker.h"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <absl/container/flat_hash_map.h>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <cstring>

namespace nrslam {

/**
 * @brief 为Marigold-DC联合优化导出NR-SLAM数据的类
 * 
 * 此类负责从NR-SLAM提取并保存以下数据：
 * 1. DDG几何约束参数：点对连接集、初始距离、连接权重、最大距离
 * 2. 粘弹性约束参数：变形偏移量、粘性权重、弹性系数
 * 3. 跟踪置信度：特征匹配SSIM值、跟踪帧成功率
 * 4. RGB图像、稀疏深度图、相机内参
 */
class MarigoldDataExporter {
public:
    struct Options {
        std::string output_dir = "./marigold_data/";  // 输出目录
        std::string log_file = "";                    // 日志文件路径
        bool save_rgb_images = true;                   // 是否保存RGB图像
        bool save_sparse_depth = true;                 // 是否保存稀疏深度图
        bool save_ddg_constraints = true;              // 是否保存DDG约束
        bool save_viscous_constraints = true;          // 是否保存粘弹性约束
        bool save_tracking_confidence = true;          // 是否保存跟踪置信度
        bool save_camera_params = true;                // 是否保存相机参数
        float distance_threshold = 1.0f;               // DDG连接距离阈值
        float sigma = 0.5f;                           // 权重计算参数
    };

    /**
     * @brief DDG几何约束数据结构
     */
    struct DDGConstraints {
        std::vector<std::pair<ID, ID>> connections;     // 点对连接集E
        std::vector<float> initial_distances;           // 初始距离l_ij^0
        std::vector<float> connection_weights;          // 连接权重w_ij
        std::vector<float> max_distances;               // 最大距离d_ij,max
        absl::flat_hash_map<ID, int> mappoint_to_pixel; // 地图点ID到像素索引的映射
    };

    /**
     * @brief 粘弹性约束数据结构
     */
    struct ViscousElasticConstraints {
        std::vector<Eigen::Vector3f> deformation_offsets;    // 变形偏移量δ_i^t
        std::vector<float> viscous_weights;                  // 粘性权重B^t
        float elastic_coefficient = 1.0f;                    // 弹性系数k
        std::vector<float> ssim_confidence;                  // SSIM跟踪置信度s_i
    };

    /**
     * @brief 跟踪统计信息
     */
    struct TrackingStatistics {
        float tracking_success_rate;                    // 跟踪帧成功率
        int total_tracked_points;                      // 总跟踪点数
        int successful_tracked_points;                 // 成功跟踪点数
        std::vector<float> per_point_ssim;            // 每个点的SSIM值
        float mean_ssim;                              // 平均SSIM值
        float std_ssim;                               // SSIM标准差
    };

    /**
     * @brief 帧数据包（传递给Marigold-DC的完整数据）
     */
    struct FrameDataPackage {
        cv::Mat rgb_image;                            // RGB图像I
        cv::Mat sparse_depth;                         // 稀疏深度图Z_sparse
        DDGConstraints ddg_constraints;               // DDG参数
        ViscousElasticConstraints viscous_constraints; // 粘弹性参数
        TrackingStatistics tracking_stats;            // 跟踪统计
        cv::Mat camera_intrinsics;                   // 相机内参矩阵K
        int frame_id;                                 // 帧ID
        int image_height;                            // 图像高度
        int image_width;                             // 图像宽度
    };

    /**
     * @brief 构造函数
     */
    MarigoldDataExporter() = delete;
    MarigoldDataExporter(const Options& options);

    /**
     * @brief 从当前帧导出数据包
     * @param rgb_image RGB图像
     * @param current_frame 当前帧
     * @param map 地图指针
     * @param frame_id 帧ID
     * @param previous_frame 上一帧（用于计算变形偏移）
     * @param klt_tracker KLT跟踪器引用（用于获取SSIM值）
     * @return 导出的数据包
     */
    FrameDataPackage ExportFrameData(const cv::Mat& rgb_image,
                                   Frame& current_frame,
                                   std::shared_ptr<Map> map,
                                   int frame_id,
                                   Frame* previous_frame = nullptr,
                                   const class LucasKanadeTracker* klt_tracker = nullptr);

    /**
     * @brief 将数据包保存到文件
     * @param data_package 数据包
     * @param frame_id 帧ID
     */
    void SaveDataPackage(const FrameDataPackage& data_package, int frame_id);

    /**
     * @brief 保存数据包的JSON元数据文件
     * @param data_package 数据包
     * @param frame_id 帧ID
     */
    void SaveMetadata(const FrameDataPackage& data_package, int frame_id);

private:
    /**
     * @brief 从Frame提取DDG约束
     */
    DDGConstraints ExtractDDGConstraints(const Frame& frame, 
                                       std::shared_ptr<Map> map);

    /**
     * @brief 计算粘弹性约束参数
     */
    ViscousElasticConstraints ComputeViscousElasticConstraints(
        const Frame& current_frame,
        const Frame& previous_frame,
        const DDGConstraints* ddg_constraints = nullptr);

    /**
     * @brief 计算跟踪统计信息
     */
    TrackingStatistics ComputeTrackingStatistics(const Frame& frame, const class LucasKanadeTracker* klt_tracker = nullptr);

    /**
     * @brief 生成稀疏深度图
     */
    cv::Mat GenerateSparseDepthMap(const Frame& frame);

    /**
     * @brief 创建几何拓扑特征图（3通道）
     */
    cv::Mat CreateGeometricTopologyFeatures(const Frame& frame, const DDGConstraints& ddg);

    /**
     * @brief 计算连接权重（插值权重）
     */
    float ComputeInterpolationWeight(float distance, float sigma);

    /**
     * @brief 确保输出目录存在
     */
    void EnsureDirectoryExists(const std::string& path);

    /**
     * @brief 保存浮点数组到文本文件
     */
    void SaveFloatArrayToText(const std::vector<float>& data, const std::string& filename);

    /**
     * @brief 保存Eigen向量数组到文本文件
     */
    void SaveEigenVectorArrayToText(const std::vector<Eigen::Vector3f>& data, const std::string& filename);

    /**
     * @brief 保存ID对数组到文本文件
     */
    void SaveIdPairArrayToText(const std::vector<std::pair<ID, ID>>& data, const std::string& filename);

    /**
     * @brief 保存2D浮点数组为numpy格式(.npy文件)
     * @param data 2D浮点数组，行优先存储
     * @param height 数组高度
     * @param width 数组宽度  
     * @param filename 输出文件名
     */
    void SaveFloatArrayAsNpy(const float* data, int height, int width, const std::string& filename);
    void SaveIDPairsToText(const std::vector<std::pair<ID, ID>>& pairs, const std::string& filename);

    /**
     * @brief 保存哈希映射到文本文件
     */
    void SaveHashMapToText(const absl::flat_hash_map<ID, int>& hash_map, const std::string& filename);

    int ComputeLinearPixelIndex(const cv::KeyPoint& keypoint) const;

    Options options_;
    std::ofstream log_file_;  // 日志文件流
    
    // 缓存当前图像尺寸，用于动态生成稀疏深度图
    int current_image_height_ = 0;
    int current_image_width_ = 0;
};

}  // namespace nrslam

#endif //NRSLAM_MARIGOLD_DATA_EXPORTER_H