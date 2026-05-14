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
#include <filesystem>
namespace fs = std::filesystem;  // 简化命名空间使用
#include "map_visualizer.h"

#include <thread>

#include "absl/log/log.h"
#include "absl/log/check.h"
#include "Eigen/Geometry"

using namespace std;

MapVisualizer::MapVisualizer(Options& options, shared_ptr<Map> map) :
options_(options), map_(map), last_frame_id_drawn_(-1), last_frame_id_saved_(-1),
last_saved_frame_id_(-1) {  // 初始化last_saved_frame_id_
    // 初始化点云保存路径（统一到output/data/point_clouds/）
    point_cloud_save_path_ = "output/data/point_clouds/";
    try {
        fs::create_directories(point_cloud_save_path_);
    } catch (const fs::filesystem_error& e) {
        LOG(WARNING) << "Failed to create point cloud directory: " << e.what();
    }

    // 初始化相机位姿保存路径（统一到output/data/camera_poses/）
    camera_pose_save_path_ = "output/data/camera_poses/";
    try {
        fs::create_directories(camera_pose_save_path_);
    } catch (const fs::filesystem_error& e) {
        LOG(WARNING) << "Failed to create camera pose directory: " << e.what();
    }
}

void MapVisualizer::InitializePangolin() {
    pangolin::CreateWindowAndBind("DefSLAM",2*1024,768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    // Initial left view.
    Eigen::Matrix4f initial_left_view = options_.initial_left_view_;
    pangolin::OpenGlMatrix left_view;
    GLdouble* m1 = left_view.m;
#define M1(row,col)  m1[(col)*4+(row)]
    M1(0,0) = initial_left_view(0,0); M1(0,1) = initial_left_view(0,1); M1(0,2) = initial_left_view(0,2); M1(0,3) = initial_left_view(0,3);
    M1(1,0) = initial_left_view(1,0); M1(1,1) = initial_left_view(1,1); M1(1,2) = initial_left_view(1,2); M1(1,3) = initial_left_view(1,3);
    M1(2,0) = initial_left_view(2,0); M1(2,1) = initial_left_view(2,1); M1(2,2) = initial_left_view(2,2); M1(2,3) = initial_left_view(2,3);
    M1(3,0) = initial_left_view(3,0); M1(3,1) = initial_left_view(3,1); M1(3,2) = initial_left_view(3,2); M1(3,3) = initial_left_view(3,3);
#undef M

    // Initial right view.
    Eigen::Matrix4f initial_right_view = options_.initial_right_view_;
    pangolin::OpenGlMatrix right_view;
    GLdouble* m2 = right_view.m;
#define M2(row,col)  m2[(col)*4+(row)]
    M2(0,0) = initial_right_view(0,0); M2(0,1) = initial_right_view(0,1); M2(0,2) = initial_right_view(0,2); M2(0,3) = initial_right_view(0,3);
    M2(1,0) = initial_right_view(1,0); M2(1,1) = initial_right_view(1,1); M2(1,2) = initial_right_view(1,2); M2(1,3) = initial_right_view(1,3);
    M2(2,0) = initial_right_view(2,0); M2(2,1) = initial_right_view(2,1); M2(2,2) = initial_right_view(2,2); M2(2,3) = initial_right_view(2,3);
    M2(3,0) = initial_right_view(3,0); M2(3,1) = initial_right_view(3,1); M2(3,2) = initial_right_view(3,2); M2(3,3) = initial_right_view(3,3);
#undef M

    // Define Camera Render Object (for view / scene browsing).
    left_renderer_ = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,1000),
            left_view);

    right_renderer_ = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,1000),
            right_view);

    // Add named OpenGL viewport to window and provide 3D Handler
    left_display_ = pangolin::Display("[Left view] Map")
            .SetAspect(1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(left_renderer_));

    right_display = pangolin::Display("[Right view] Map")
            .SetAspect(1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(right_renderer_));

    main_display_ = pangolin::Display("multi")
            .SetBounds(0.0, 1.0, 0.0, 1.0)
            .SetLayout(pangolin::LayoutEqual)
            .AddDisplay(left_display_)
            .AddDisplay(right_display);

    // Set function to save rendered images.
    main_display_.extern_draw_function = [&](pangolin::View&) {
        if (map_->IsEmpty()) {
            return;
        }

        RenderLeftDisplay();

        RenderRightDisplay();
    };

    // Side menu.
    const int UI_WIDTH = 180;
    pangolin::CreatePanel("ui")
            .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    show_ground_truth_ = shared_ptr<pangolin::Var<bool>>(
            new pangolin::Var<bool>("ui.Show GroundTruth", false, true));

}

void MapVisualizer::SaveLeftPointCloud(const std::vector<Eigen::Vector3f>& points,
                                      const Eigen::Vector3f& color, int frame_id) {
    if (points.empty()) return;

    // 生成文件名（使用帧ID命名）
    std::string filename = point_cloud_save_path_ + std::to_string(frame_id) + ".ply";

    // 打开文件
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG(WARNING) << "Failed to open point cloud file: " << filename;
        return;
    }

    // 写入PLY头
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << points.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";

    // 写入点数据（坐标+颜色）
    for (const auto& pt : points) {
        file << pt.x() << " " << pt.y() << " " << pt.z() << " ";
        file << static_cast<int>(color.x() * 255) << " ";  // 颜色归一化到0-255
        file << static_cast<int>(color.y() * 255) << " ";
        file << static_cast<int>(color.z() * 255) << "\n";
    }

    file.close();
    LOG(INFO) << "Saved left camera point cloud: " << filename;
}

// 新增：保存相机位姿（TUM格式）
void MapVisualizer::SaveCameraPose(const Sophus::SE3f& pose, int frame_id) {
    // 文件名：camera_poses/frame_id.txt
    std::string filename = camera_pose_save_path_ + std::to_string(frame_id) + ".txt";
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG(WARNING) << "Failed to open camera pose file: " << filename;
        return;
    }

    // 提取平移向量（tx, ty, tz）
    Eigen::Vector3f t = pose.translation();
    // 提取旋转四元数（qx, qy, qz, qw）
    Eigen::Quaternionf q(pose.rotationMatrix());

    // TUM格式：timestamp tx ty tz qx qy qz qw（用帧ID作为时间戳）
    file << frame_id << " "       // timestamp
         << t.x() << " "          // tx
         << t.y() << " "          // ty
         << t.z() << " "          // tz
         << q.x() << " "          // qx
         << q.y() << " "          // qy
         << q.z() << " "          // qz
         << q.w() << std::endl;   // qw

    file.close();
    LOG(INFO) << "Saved camera pose: " << filename;
}

void MapVisualizer::Run() {
    InitializePangolin();

    while (!should_finish) {
        RenderVisualization();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void MapVisualizer::RenderVisualization() {
    if (map_->IsEmpty()) {
        return;
    }

    ResetVisualization();

    RenderLeftDisplay();

    RenderRightDisplay();

    SaveRenderToDisk();

    FinishVisualization();
}

void MapVisualizer::ResetVisualization() {
    glClearColor(1.0f,1.0f,1.0f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void MapVisualizer::RenderLeftDisplay() {
    left_display_.Activate(left_renderer_);

    DrawLastFrame();
    DrawKeyFrames();
    DrawLatestTrajectory();
    DrawNonTrackedLandmarks();

    // 保存左视图点云和相机位姿（仅当帧ID更新时）
    if (!map_->IsEmpty()) {
        Frame last_frame = map_->GetLastFrame();
        int current_frame_id = last_frame.GetId();  // 获取当前帧ID

        // 每帧都保存当前可见点云（包括所有状态的可见点）
        // 统计不同状态的点数量
        int tracked_3d_count = 0, tracked_count = 0, just_triangulated_count = 0, candidate_3d_count = 0;

        std::vector<Eigen::Vector3f> left_points;

        // 收集左相机所有可见的3D点（包括TRACKED、TRACKED_WITH_3D、JUST_TRIANGULATED、CANDIDATE_3D）
        for (int idx = 0; idx < last_frame.LandmarkPositions().size(); idx++) {
            LandmarkStatus status = last_frame.LandmarkStatuses()[idx];
            if (status == TRACKED_WITH_3D) {
                left_points.push_back(last_frame.LandmarkPositions()[idx]);
                tracked_3d_count++;
            } else if (status == JUST_TRIANGULATED) {
                left_points.push_back(last_frame.LandmarkPositions()[idx]);
                just_triangulated_count++;
            } else if (status == TRACKED || status == CANDIDATE_3D) {
                // 也保存TRACKED和CANDIDATE_3D状态的点（如果它们有有效的3D位置）
                if (last_frame.LandmarkPositions()[idx].norm() > 0) {
                    left_points.push_back(last_frame.LandmarkPositions()[idx]);
                    if (status == TRACKED) tracked_count++;
                    else candidate_3d_count++;
                }
            }
        }

        // 输出每帧点数统计
        LOG(INFO) << "Frame " << current_frame_id << " point statistics: "
                  << "3D_tracked=" << tracked_3d_count << ", "
                  << "tracked=" << tracked_count << ", "
                  << "triangulated=" << just_triangulated_count << ", "
                  << "candidate_3d=" << candidate_3d_count << ", "
                  << "total_visible=" << left_points.size();

        // 保存点云统计信息
        SavePointCloudStatistics(current_frame_id, left_points.size(),
                                tracked_3d_count, just_triangulated_count, candidate_3d_count);

            // 保存点云
            SaveLeftPointCloud(left_points, Eigen::Vector3f(1, 0, 0), current_frame_id);

            // 保存相机位姿（新增）
            SaveCameraPose(last_frame.CameraTransformationWorld(), current_frame_id);

            // 更新上一次保存的帧ID
            last_saved_frame_id_ = current_frame_id;
    }
}

void MapVisualizer::RenderRightDisplay() {
    right_display.Activate(right_renderer_);

    DrawLastFrame();
    DrawKeyFrames();
    DrawLatestTrajectory();
    DrawNonTrackedLandmarks();
}

void MapVisualizer::FinishVisualization() {
    pangolin::FinishFrame();
}

void MapVisualizer::DrawLastFrame() {
    Frame last_frame = map_->GetLastFrame();
    last_frame_id_drawn_ = last_frame.GetId();

    vector<Eigen::Vector3f> landmarks_with_3d, landmarks_recently_triangulated;
    vector<vector<Eigen::Vector3f>> landmarks_with_3d_flow;
    vector<absl::StatusOr<Eigen::Vector3f>> landmarks_ground_truth;
    for (int idx = 0; idx < last_frame.LandmarkPositions().size(); idx++){
        if (last_frame.LandmarkStatuses()[idx] == TRACKED_WITH_3D) {
            auto landmark_id = last_frame.IndexToMapPointId().at(idx);
            auto landmark = map_->GetMapPoint(landmark_id);
            if (!landmark) {
                continue;
            }

            landmarks_with_3d.push_back(last_frame.LandmarkPositions()[idx]);
            landmarks_ground_truth.push_back(last_frame.GroundTruth()[idx]);

            landmarks_with_3d_flow.push_back(landmark->GetLandmarkFlow(20));
        } else if (last_frame.LandmarkStatuses()[idx] == JUST_TRIANGULATED) {
            landmarks_recently_triangulated.push_back(last_frame.LandmarkPositions()[idx]);
        }
    }

    Draw3DFlow(landmarks_with_3d_flow, Eigen::Vector3f(1, 0, 0));
    Draw3DPoints(landmarks_with_3d, Eigen::Vector3f(1, 0, 0));

    Draw3DPoints(landmarks_recently_triangulated, Eigen::Vector3f(1, 0, 0));

    if(*show_ground_truth_) {
        DrawGroundTruth(landmarks_with_3d, landmarks_ground_truth, Eigen::Vector3f(1, 0.5, 0.3));
    }

    DrawCamera(last_frame.CameraTransformationWorld(), Eigen::Vector3f(0, 1, 0));
}

void MapVisualizer::Draw3DPoints(std::vector<Eigen::Vector3f> point_cloud, Eigen::Vector3f color) {
    glPointSize(3);
    glColor3f(color.x(), color.y(), color.z());

    glBegin(GL_POINTS);
    for (Eigen::Vector3f landmark : point_cloud){
        glVertex3f(landmark(0), landmark(1), landmark(2));
    }

    glEnd();
}

void MapVisualizer::Draw3DFlow(const std::vector<std::vector<Eigen::Vector3f>> &flows, const Eigen::Vector3f color) {
    for (auto flow : flows) {
        if (flow.size() < 2) {
            continue;
        }
        DrawConnectedPoints(flow, color);
    }
}

void MapVisualizer::DrawGroundTruth(std::vector<Eigen::Vector3f> &estimated_point_cloud,
                                    std::vector<absl::StatusOr<Eigen::Vector3f>> &ground_truth_point_cloud,
                                    Eigen::Vector3f color) {
    glPointSize(3);
    glLineWidth(2);
    glColor3f(color.x(), color.y(), color.z());


    for (int idx = 0; idx < ground_truth_point_cloud.size(); idx++){
        if (ground_truth_point_cloud[idx].ok()) {
            const Eigen::Vector3f ground_truth_point = *ground_truth_point_cloud[idx];
            const Eigen::Vector3f estimated_landmark = estimated_point_cloud[idx];

            glBegin(GL_LINES);
            glVertex3f(ground_truth_point.x(), ground_truth_point.y(), ground_truth_point.z());
            glVertex3f(estimated_landmark.x(), estimated_landmark.y(), estimated_landmark.z());
            glEnd();

            glBegin(GL_POINTS);
            glVertex3f(ground_truth_point(0), ground_truth_point(1), ground_truth_point(2));
            glEnd();
        }

    }
}

void MapVisualizer::DrawCamera(Sophus::SE3f camera_transformation_world, Eigen::Vector3f color) {
    const float &w = options_.camera_size_;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();
    glMultMatrixf(camera_transformation_world.inverse().matrix().data());

    glLineWidth(2);
    glBegin(GL_LINES);
    glColor3f(color.x(), color.y(), color.z());

    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}

void MapVisualizer::SetFinish() {
    should_finish = true;
}

void MapVisualizer::DrawKeyFrames() {
    auto keyframes = map_->GetKeyFrames();

    const int n_skip = 2;
    int n_from_last_drawn = 0;
    for (const auto& [id, keyframe] : keyframes) {
        if (n_from_last_drawn >= n_skip) {
            DrawCamera(keyframe->CameraTransformationWorld(), Eigen::Vector3f(0, 0, 1));
            n_from_last_drawn = 0;
        } else {
            n_from_last_drawn++;
        }
    }
}

void MapVisualizer::DrawLatestTrajectory() {
    auto latest_frames = map_->GetTemporalBuffer()->GetLatestCameraPoses();

    vector<Eigen::Vector3f> trajectory;
    for (auto& pose : latest_frames) {
        trajectory.push_back(pose.inverse().translation());
    }

    if (trajectory.size() < 2) {
        return;
    }

    DrawConnectedPoints(trajectory, Eigen::Vector3f(0, 0, 1));
}

void MapVisualizer::DrawConnectedPoints(std::vector<Eigen::Vector3f> &trajectory,
                                        const Eigen::Vector3f color) {
    CHECK_GE(trajectory.size(), 2);
    for(int idx = 0; idx < trajectory.size() - 1; idx++){
        Eigen::Vector3f current_point = trajectory[idx];
        Eigen::Vector3f next_point = trajectory[idx + 1];

        glBegin(GL_LINES);
        glColor3f(color.x(), color.y(), color.z());
        glVertex3f(current_point.x(), current_point.y(), current_point.z());
        glVertex3f(next_point.x(), next_point.y(), next_point.z());
        glEnd();
    }
}

void MapVisualizer::DrawNonTrackedLandmarks() {
    auto mappoints = map_->GetMapPoints();

    Frame last_frame = map_->GetLastFrame();

    vector<Eigen::Vector3f> non_tracked_3d, non_tracked_active_3d;
    vector<vector<Eigen::Vector3f>> non_tracked_active_3d_flow;
    for (const auto& [mappoint_id, mappoint] : mappoints) {
        if (last_frame.MapPointIdToIndex().contains(mappoint_id)) {
            continue;
        }

        if (mappoint->IsActive()) {
            non_tracked_active_3d.push_back(mappoint->GetLastWorldPosition());
            non_tracked_active_3d_flow.push_back(mappoint->GetLandmarkFlow(20));
        } else {
            non_tracked_3d.push_back(mappoint->GetLastWorldPosition());
        }
    }

    Draw3DPoints(non_tracked_3d, Eigen::Vector3f(0, 0, 0));
    Draw3DPoints(non_tracked_active_3d, Eigen::Vector3f(0, 0, 0));
}

void MapVisualizer::SaveRenderToDisk() {
    if (options_.render_save_path.empty()) {
        return;
    }

    // 确保渲染图像保存目录存在
    try {
        fs::create_directories(fs::path(options_.render_save_path).parent_path());
    } catch (const fs::filesystem_error& e) {
        LOG(WARNING) << "Failed to create render directory: " << e.what();
        return;
    }

    if(last_frame_id_saved_ < last_frame_id_drawn_) {
        // 图像文件名格式：<render_save_path><frame_id>.png（添加.png扩展名）
        const string render_path = options_.render_save_path +
                                 std::to_string(last_frame_id_drawn_) + ".png";
        main_display_.SaveRenderNow(render_path);

        last_frame_id_saved_ = last_frame_id_drawn_;
        LOG(INFO) << "Saved render image: " << render_path;
    }
}

// 新增：保存点云统计信息到文件
void MapVisualizer::SavePointCloudStatistics(int frame_id, int total_points, int tracked_3d, int triangulated, int candidate_3d) {
    static std::string stats_file_path = point_cloud_save_path_ + "point_cloud_statistics.csv";

    // 检查是否是第一次调用，如果是则写入表头
    static bool header_written = false;
    std::ofstream stats_file;
    if (!header_written) {
        stats_file.open(stats_file_path);
        if (stats_file.is_open()) {
            stats_file << "frame_id,total_points,tracked_3d,triangulated,candidate_3d,quality_score\n";
            header_written = true;
        }
    } else {
        stats_file.open(stats_file_path, std::ios::app);
    }

    if (stats_file.is_open()) {
        // 计算质量分数（基于点数和3D点比例）
        float quality_score = total_points > 0 ?
            (tracked_3d * 1.0f + triangulated * 0.8f + candidate_3d * 0.6f) / total_points : 0.0f;

        stats_file << frame_id << ","
                   << total_points << ","
                   << tracked_3d << ","
                   << triangulated << ","
                   << candidate_3d << ","
                   << std::fixed << std::setprecision(3) << quality_score << "\n";
        stats_file.close();
    }
}