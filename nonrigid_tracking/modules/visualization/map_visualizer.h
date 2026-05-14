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

#ifndef NRSLAM_MAP_VISUALIZER_H
#define NRSLAM_MAP_VISUALIZER_H

#include "map/map.h"

#include <pangolin/pangolin.h>

#include <memory>
#include <sophus/se3.hpp>

class MapVisualizer {
public:
    struct Options {
        Eigen::Matrix4f initial_left_view_;
        Eigen::Matrix4f initial_right_view_;

        float camera_size_;

        std::string render_save_path;  // 渲染图像保存路径（需包含文件名前缀）
    };

    MapVisualizer() = delete;

    MapVisualizer(Options& options, std::shared_ptr<Map> map);

    void Run();

    void SetFinish();

    // 新增：保存点云统计信息到文件
    void SavePointCloudStatistics(int frame_id, int total_points, int tracked_3d, int triangulated, int candidate_3d);

private:
    void RenderVisualization();

    void InitializePangolin();

    void ResetVisualization();

    void RenderLeftDisplay();

    void RenderRightDisplay();

    void FinishVisualization();

    void DrawLastFrame();

    void Draw3DPoints(std::vector<Eigen::Vector3f> point_cloud, Eigen::Vector3f color);

    void Draw3DFlow(const std::vector<std::vector<Eigen::Vector3f>>& flows, const Eigen::Vector3f color);

    void DrawGroundTruth(std::vector<Eigen::Vector3f>& estimated_point_cloud,
                         std::vector<absl::StatusOr<Eigen::Vector3f>>& ground_truth_point_cloud,
                         Eigen::Vector3f color);

    void DrawCamera(Sophus::SE3f camera_transformation_world, Eigen::Vector3f color);

    void DrawKeyFrames();

    void DrawLatestTrajectory();

    void DrawConnectedPoints(std::vector<Eigen::Vector3f>& trajectory, const Eigen::Vector3f color);

    void DrawNonTrackedLandmarks();

    void SaveRenderToDisk();

    // 点云保存相关
    void SaveLeftPointCloud(const std::vector<Eigen::Vector3f>& points,
                           const Eigen::Vector3f& color, int frame_id);

    // 相机位姿保存相关（新增）
    void SaveCameraPose(const Sophus::SE3f& pose, int frame_id);

    // Pangolin fields for visualization.
    pangolin::View left_display_, right_display;
    pangolin::OpenGlRenderState left_renderer_, right_renderer_;

    // GUI controls.
    std::shared_ptr<pangolin::Var<bool>> show_ground_truth_;

    pangolin::View main_display_;

    Options options_;

    std::shared_ptr<Map> map_;

    bool should_finish = false;

    int last_frame_id_drawn_, last_frame_id_saved_;
    std::string point_cloud_save_path_;  // 点云保存路径
    int last_saved_frame_id_ = -1;       // 上一次保存的帧ID

    std::string camera_pose_save_path_;  // 新增：相机位姿保存路径
};


#endif //NRSLAM_MAP_VISUALIZER_H