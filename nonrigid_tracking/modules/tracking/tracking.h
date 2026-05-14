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

#ifndef NRSLAM_TRACKING_H
#define NRSLAM_TRACKING_H
 
 #include "calibration/camera_model.h"
 #include "features/feature.h"
 #include "map/frame.h"
 #include "map/map.h"
 #include "matching/lucas_kanade_tracker.h"
 #include "stereo/stereo_lucas_kanade.h"
 #include "stereo/stereo_pattern_matching.h"
 #include "tracking/monocular_map_initializer.h"
#include "utilities/time_profiler.h"
#include "utilities/marigold_data_exporter.h"
#include "visualization/image_visualizer.h"

#include <Eigen/Dense>
#include <absl/container/flat_hash_set.h>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <utility>

class Mapping;  // Forward declaration

class Tracking {
 public:
    struct Options {
        int klt_window_size = 21;
        int klt_max_level = 3;
        int klt_max_iters = 50;
        float klt_epsilon = 0.01;
        float klt_min_eig_th = 1e-4;
        // 对内窥镜等噪声较大的场景，进一步降低 SSIM 阈值以保留更多可跟踪特征
        float klt_min_SSIM = 0.15;

        int images_to_insert_keyframe = 3;  // 降低到3帧，加快关键帧插入以增加三角化机会

        float radians_per_pixel;
        
        // ------------ 深度/尺度相关可调参数（用于减少点云“扁平”现象） ------------
        // 单目初始化：假设场景中的典型深度（单位为 SLAM 世界坐标单位）
        // 原始实现中为 3.f，这里暴露为参数，便于针对不同数据集调节整体尺度
        float mono_init_target_median_depth = 3.0f;

        // 单目初始化：正则化图半径 = sigma_scaled * mono_init_reg_sigma_scale
        // 原始实现中为 3.f，可用于控制形变正则化强度（过大可能导致点云略扁）
        float mono_init_reg_sigma_scale = 3.0f;

        // Stereo 初始化：深度过滤范围（单位与输入相机坐标一致）
        float stereo_init_min_depth = 0.0f;      // 默认不特别裁剪下界
        float stereo_init_max_depth = 1e6f;      // 默认几乎不过滤上界

        // StereoPatternMatching 中使用的焦距/尺度参数（原代码为 3886.37）
        float stereo_pattern_focal = 3886.37f;

        // Stereo 初始化后的正则化图半径与地图尺度
        float stereo_reg_radius = 10.5f;         // 对应原来的 InitializeRegularizationGraph(10.5)
        float stereo_map_scale = 1.0f;           // 对应原来的 SetMapScale(1.f)
        
        // 特征提取方法：例如 "shi_tomasi" 或 "orb3"
        std::string feature_type = "shi_tomasi";
        int orb_n_features = 1000;
        float orb_scale_factor = 1.2f;
        int orb_n_levels = 8;
        
        // Marigold数据导出选项
        bool enable_marigold_export = false;
        std::string marigold_output_dir = "./marigold_data/";
        
        // 联合优化融合导出选项（创新点1）
        bool enable_json_ply_export = false;
        std::string json_path_template = "./outputs/frame_%d.json";
        std::string ply_path_template = "./outputs/sparse_%d.ply";
        
        // Debug 选项
        bool dump_init_pointcloud = false;
        bool dump_pre_ba_pointcloud = false;
        
        struct TissueParam {
            float k = 1.0f;
            float b = 0.5f;
        };

        // 组织分割选项（创新点2）
        bool enable_heterogeneous_viscoelastic = false;
        std::string mask_json_path = "./masks.json";  // SAM分割输出的mask JSON路径
        absl::flat_hash_map<std::string, TissueParam> tissue_params;
    };

    enum TrackingStatus {
         NOT_INITIALIZED,
         TRACKING,
         LOST
     };
 
     Tracking() = delete;
 
     Tracking(const Options options, std::shared_ptr<Map> map,
              std::shared_ptr<CameraModel> calibration,
              std::shared_ptr<StereoLucasKanade> stereo_matcher,
              std::shared_ptr<ImageVisualizer> image_visualizer,
              TimeProfiler* time_profiler,
              Mapping* mapper = nullptr);
 
     void TrackImage(const cv::Mat& im, const absl::flat_hash_map<std::string, cv::Mat>& masks,
                     const cv::Mat& additional_im = cv::Mat(), const cv::Mat& im_clahe = cv::Mat(),int frame_idx = -1);
 
     TrackingStatus GetTrackingStatus() const;
 
 private:
     void ExtractFeatures(const cv::Mat& im, const cv::Mat& mask,
                          std::vector<cv::KeyPoint>& keypoints);
 
     void MonocularMapInitialization(const cv::Mat& im_left,
                                  const cv::Mat& mask, const cv::Mat& im_clahe);
 
     void StereoMapInitialization(const cv::Mat& im_left, const cv::Mat& im_right,
                                  const cv::Mat& mask, const cv::Mat& im_clahe);
 
     absl::flat_hash_set<ID> TrackCameraAndDeformation(const cv::Mat& im, const cv::Mat& mask, int frame_idx);
 
    void DataAssociation(const cv::Mat& im, const cv::Mat& mask);

    void CameraPoseEstimation();

    void RedetectFeatures(const cv::Mat& im, const cv::Mat& mask);

    void SupplementFeaturesForTriangulation(const cv::Mat& im, const cv::Mat& mask);

    void AddVisibleMapPointsToCurrentFrame(const cv::Mat& im);

    void PerformActiveTriangulation();

     absl::flat_hash_set<ID> CameraPoseAndDeformationEstimation(const cv::Mat &im, int frame_idx);
 
     void KeyFrameInsertion(const cv::Mat& im, const absl::flat_hash_map<std::string, cv::Mat>& masks);
 
     bool NeedNewKeyFrame();
 
     void CreateNewKeyFrame(const cv::Mat& im, const absl::flat_hash_map<std::string, cv::Mat>& masks);
 
     void ExtractFeaturesInFrame(const cv::Mat& im, const cv::Mat& mask, Frame& frame);
 
     void SetKLTReference(const cv::Mat& im, Frame& frame, const cv::Mat& mask);
 
     void PointReuse(const cv::Mat& im, const cv::Mat& mask,
                     absl::flat_hash_set<ID> lost_mappoint_ids);
 
    void UpdateTriangulatedPoints();

    // 保存当前帧的RGB图像（点云和位姿由map_visualizer负责保存）
    void SaveCurrentFrameData(const cv::Mat& rgb_image, int frame_idx);
    
    // 保存RGB图像到文件
    void SaveImage(const cv::Mat& image, const std::string& filename);
    
    // 导出Marigold-DC联合优化所需数据
    void ExportMarigoldData(const cv::Mat& rgb_image, int frame_idx);
    
    // 导出JSON/PLY格式数据（创新点1：联合优化融合）
    void ExportJSONAndPLY(int frame_idx, const std::string& json_path_template, const std::string& ply_path_template);
    
    // 加载组织分割mask（创新点2）
    void LoadTissueMasks(const std::string& mask_json_path);
    
    // 获取异质粘弹性参数（创新点2）
    bool GetHeterogeneousViscoelasticParams(ID mappoint_id, int pixel_idx, float& k, float& b, std::string* class_name_out = nullptr) const;
    
    // 获取KLT跟踪器的引用（用于获取SSIM值）
    const LucasKanadeTracker& GetKLTTracker() const { return klt_tracker_; }

    Options options_;
 
     std::shared_ptr<Map> map_;
 
     std::shared_ptr<CameraModel> calibration_;
 
     std::shared_ptr<Feature> feature_extractor_;
 
     LucasKanadeTracker klt_tracker_;
 
     std::shared_ptr<Frame> current_frame_;
 
     std::shared_ptr<StereoLucasKanade> stereo_matcher_;
 
     Sophus::SE3f motion_model_;
 
     std::shared_ptr<ImageVisualizer> image_visualizer_;
 
     int n_images_from_last_keyframe_ = 0;
 
     std::unique_ptr<MonocularMapInitializer> monocular_map_initializer_;
 
     TrackingStatus tracking_status_;
 
     Sophus::SE3f previous_camera_transform_world_;
 
     TimeProfiler* time_profiler_;

    // RGB图像保存路径
    std::string point_cloud_and_pose_path_;
    std::string tracking_stats_path_;
    
    // 内部帧计数器（用于未提供frame_idx的情况）
    int internal_frame_counter_;
    
    // 帧间统计（替代原来的 static 局部变量）
    int prev_total_3d_ = 0;
    int prev_frame_idx_ = -1;
    
    // Marigold数据导出器
    std::unique_ptr<nrslam::MarigoldDataExporter> marigold_exporter_;
    
    // 上一帧指针（用于计算变形偏移）
    std::shared_ptr<Frame> previous_frame_;

    // Mapping对象引用（用于主动三角化）
    Mapping* mapper_;

    // 是否启用Marigold数据导出
    bool enable_marigold_export_;
    
    struct FrameMaskData {
        int image_width = 0;
        int image_height = 0;
        struct ClassMask {
            std::vector<uint8_t> mask;
            float score = 0.f;
            int area = 0;
            std::vector<int> color;  // RGB color [R, G, B] for visualization
        };
        absl::flat_hash_map<std::string, ClassMask> class_masks;
        
        // O(1) 查表结构：pixel_idx -> class_names 索引 (0 = 无归属, 1..N = 对应类别)
        std::vector<uint8_t> class_index_map;           // 大小 = image_width * image_height
        std::vector<std::string> class_names_by_index;   // index 0 = "", 1..N = 实际类别名
        
        // 在所有 class_masks 加载完毕后调用，构建 O(1) 查表
        void BuildClassIndexMap() {
            int total_pixels = image_width * image_height;
            if (total_pixels <= 0 || class_masks.empty()) return;
            
            class_index_map.assign(total_pixels, 0);  // 默认 0 = 无归属
            class_names_by_index.clear();
            class_names_by_index.push_back("");  // index 0 = 无归属
            
            uint8_t class_idx = 1;
            for (const auto& [name, cls] : class_masks) {
                if (class_idx >= 255) break;  // uint8_t 最多 254 个类别
                class_names_by_index.push_back(name);
                for (int px = 0; px < std::min(total_pixels, static_cast<int>(cls.mask.size())); ++px) {
                    if (cls.mask[px] > 0 && class_index_map[px] == 0) {
                        class_index_map[px] = class_idx;
                    }
                }
                ++class_idx;
            }
        }
    };

    // 组织分割mask数据（创新点2）
    mutable absl::flat_hash_map<int, FrameMaskData> frame_mask_data_;  // frame_id -> class_name -> mask (lazy loaded)
    absl::flat_hash_set<int> available_frame_ids_;  // 记录哪些帧有 mask 数据
    std::string mask_json_path_;  // mask 文件路径或目录路径
    bool masks_loaded_ = false;  // 是否已初始化
    bool use_per_frame_files_ = false;  // 是否使用按帧保存模式
    std::string frames_dir_;  // 按帧保存的目录路径
    int last_export_frame_id_ = -1;
    mutable absl::flat_hash_map<int, bool> frame_loaded_cache_;  // 记录哪些帧已加载到内存

    void GuessTissueParams(float area_ratio, float score, float& k, float& b) const;
    
    // 按需加载指定帧的 mask 数据（按帧保存模式）
    bool LoadFrameMaskDataFromFile(int frame_id) const;
}; 
 #endif //NRSLAM_TRACKING_H
 