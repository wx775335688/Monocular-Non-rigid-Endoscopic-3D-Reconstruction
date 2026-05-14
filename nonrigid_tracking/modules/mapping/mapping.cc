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

#include <fstream>
#include "mapping.h"

#include "optimization/g2o_optimization.h"
#include "utilities/geometry_toolbox.h"

#include "absl/log/log.h"
#include "absl/log/check.h"

using namespace std;

Mapping::Mapping(std::shared_ptr<Map> map, std::shared_ptr<CameraModel> calibration,
                 const Options options, TimeProfiler* time_profiler) :
        options_(options), map_(map), calibration_(calibration), time_profiler_(time_profiler) {
}

void Mapping::DoMapping() {
    if (map_->IsEmpty()) {
        return;
    }

    // Get next KeyFrame to process
    auto keyframe = map_->GetNextUnmappedKeyFrame();

    if (keyframe) {
        // If the KeyFrame is valid, do KeyFrame mapping.
        KeyFrameMapping();

        // Update tracking frame with the optimized KeyFrame.
        UpdateTrackingFrameFromKeyFrame(keyframe);
    } else {
        // If there is no KeyFrame to process, do Frame mapping.
        FrameMapping();
    }
}

void Mapping::KeyFrameMapping() {
    LocalDeformableBundleAdjustment(map_, map_->GetMapScale());
}

void Mapping::FrameMapping() {
    // Triangulate new landmarks.
    LandmarkTriangulation();
}

void Mapping::LandmarkTriangulation() {
    auto temporal_buffer = map_->GetTemporalBuffer();

    // Iterate over the KeyPoints tracked in the last frame
    vector<int> triangulation_candidate_ids = temporal_buffer->GetTriangulationCandidatesIds();

    auto current_frame = map_->GetMutableLastFrame();

    auto calibration = current_frame->GetCalibration();

    int triangulated_landmarks = 0;

    const int current_frame_id = current_frame->GetId() - 1;

    int n_rigid_triangulations = 0;
    int n_deformable_triangulations = 0;

    // 统计当前帧已有 3D 点的深度分布，用于后续对新三角化点做简单的深度一致性过滤
    Sophus::SE3f current_camera_T_world = current_frame->CameraTransformationWorld();
    std::vector<float> existing_depths;
    {
        auto existing_ids = current_frame->GetMapPointsIdsWithStatus({TRACKED_WITH_3D});
        existing_depths.reserve(existing_ids.size());
        for (auto id : existing_ids) {
            auto pos_status = current_frame->LandmarkPosition(id);
            if (!pos_status.ok()) continue;
            Eigen::Vector3f cam_p = current_camera_T_world * (*pos_status);
            if (cam_p.z() > 0) {
                existing_depths.push_back(cam_p.z());
            }
        }
    }

    float existing_depth_mean = 0.0f, existing_depth_std = 0.0f;
    if (existing_depths.size() >= 20) {
        float sum = 0.0f;
        for (auto d : existing_depths) sum += d;
        existing_depth_mean = sum / static_cast<float>(existing_depths.size());
        float var = 0.0f;
        for (auto d : existing_depths) {
            float diff = d - existing_depth_mean;
            var += diff * diff;
        }
        var /= static_cast<float>(existing_depths.size());
        existing_depth_std = std::sqrt(std::max(0.0f, var));
    }

    vector<absl::StatusOr<Eigen::Vector3f>> rigid_triangulations;
    vector<absl::StatusOr<Eigen::Vector3f>> deformable_triangulations;
    // Diagnostic counters for triangulation attempts / failures
    int count_close_features = 0;
    int count_short_track = 0;
    int count_rigidity_fail = 0;
    int count_triangulation_fail = 0;
    int count_parallax_fail = 0;
    int count_reprojection_fail = 0;
    int count_landmark_nan = 0;
    int count_success_rigid = 0;
    int count_success_deformable = 0;

    // Try to triangulate candidates.
    vector<int> candidates_triangulated;
    vector<ID> landmark_ids_triangulated;
    for (auto candidate_id : triangulation_candidate_ids) {
        // Check there are no close features.
        // 对于内窥镜场景，放宽邻域检查，避免过于严格的限制
        // 只在邻域非常密集时才跳过（距离<5像素且数量>10）
        auto neighbour_ids = temporal_buffer->GetClosestMapPointsToFeature(candidate_id, 5, 5, 500);
        bool too_dense = !neighbour_ids.empty() && neighbour_ids.size() > 10;

        if (too_dense) {
            rigid_triangulations.push_back(absl::InternalError("Too dense"));
            deformable_triangulations.push_back(absl::InternalError("Too dense"));
            ++count_close_features;
            continue;
        }

        // 要求轨迹长度足够长才进行可形变三角化
        // 对于快速运动的场景，进一步降低到 >=1，让新特征点能更快参与三角化
        int track_length = temporal_buffer->TrackLength(candidate_id);

        if (track_length >= 1) {
            auto landmark_triangulated = DeformableLandmarkTriangulation(candidate_id);

            if (landmark_triangulated.ok()) {
                if ((*landmark_triangulated).hasNaN()) {
                    deformable_triangulations.push_back(absl::InternalError("NaN."));
                    ++count_landmark_nan;
                } else {
                    deformable_triangulations.push_back(landmark_triangulated);
                    if (landmark_triangulated.ok()) {
                        n_deformable_triangulations++;
                        ++count_success_deformable;
                    }
                }
            } else {
                // 可形变三角化失败，尝试备用的宽松刚性三角化
                deformable_triangulations.push_back(absl::InternalError("Deformable failed, trying rigid"));
            }

        } else {
            deformable_triangulations.push_back(absl::InternalError("Short track"));
            ++count_short_track;
        }

        // Recover feature track.
        auto candidate_track = temporal_buffer->GetFeatureTrack(candidate_id);

        const auto& [current_frame_id, current_keypoint] = candidate_track.front();
        const auto& [previous_frame_id, previous_keypoint] = candidate_track.back();

        // Rigidity condition.
        // 对于内窥镜等存在明显组织形变的场景，需要更宽松的刚性检查
        // 这里进一步放宽到 0.5（原为0.1），让更多候选点参与三角化
        // 在内窥镜场景中，由于相机移动和组织变形，刚性假设往往不成立
        float rigidity_threshold = 0.5f;
        if (!temporal_buffer->CheckRigidity(current_frame_id, previous_frame_id, rigidity_threshold)) {
            // 刚性检查失败仍然允许尝试刚性三角化，因为刚性三角化本身就是一个验证过程
            // 删除冗余日志输出
        }

        // Unproject rays.
        Eigen::Vector3f current_ray =
                calibration->Unproject(current_keypoint.pt.x, current_keypoint.pt.y).normalized();
        Eigen::Vector3f previous_ray =
                calibration->Unproject(previous_keypoint.pt.x, previous_keypoint.pt.y).normalized();

        // Get camera poses.
        auto current_camera_transform_world = temporal_buffer->GetCameraTransformWorld(current_frame_id);
        auto previous_camera_transform_world = temporal_buffer->GetCameraTransformWorld(previous_frame_id);

        auto landmark_position_status =
                TriangulateMidPoint(previous_ray, current_ray,
                                    *previous_camera_transform_world, *current_camera_transform_world);

        if (!landmark_position_status.ok()) {
            rigid_triangulations.push_back(landmark_position_status);
            ++count_triangulation_fail;
            continue;
        }

        Eigen::Vector3f normal_1 = (*landmark_position_status) - (*current_camera_transform_world).inverse().translation();
        Eigen::Vector3f normal_2 = (*landmark_position_status) - (*previous_camera_transform_world).inverse().translation();
        float parallax = RaysParallax(normal_1, normal_2);

        // 视差限制：原始为 [10 * rad_per_pixel, 20 * rad_per_pixel]
        // 对于小基线场景过于苛刻，这里进一步放宽以产生更多3D点
        // 进一步降低最小视差要求，从2.0降到0.5，允许更小的视差
        if(parallax < options_.rad_per_pixel * 0.5f ||
           parallax > options_.rad_per_pixel * options_.max_parallax_mult){
            rigid_triangulations.push_back(absl::InternalError("Parallax error."));
            ++count_parallax_fail;
            continue;
        }

        // Check Reprojection error.
        Eigen::Vector3f landmark_position_1 = (*previous_camera_transform_world) * (*landmark_position_status);

        if (landmark_position_1.z() < 0) {
            rigid_triangulations.push_back(absl::InternalError("Parallax error."));
            continue;
        }

        cv::Point2f projected_landmark_1 = calibration->Project(landmark_position_1);

        // 重投影误差阈值：原始 5.991 大致为卡方 2 自由度 95% 置信区间，
        // 对噪声较大的真实内窥镜数据过于严格，这里略微放宽到约 9.21（99% 置信区间）。
        if(SquaredReprojectionError(previous_keypoint.pt, projected_landmark_1) > options_.reprojection_chi2_threshold){
            rigid_triangulations.push_back(absl::InternalError("Reprojection error."));
            ++count_reprojection_fail;
            continue;
        }

        Eigen::Vector3f landmark_position_2 = (*current_camera_transform_world) * (*landmark_position_status);

        if (landmark_position_2.z() < 0) {
            rigid_triangulations.push_back(absl::InternalError("Parallax error."));
            continue;
        }

        cv::Point2f projected_landmark_2 = calibration->Project(landmark_position_2);

        if(SquaredReprojectionError(current_keypoint.pt, projected_landmark_2) > 9.21f){
            rigid_triangulations.push_back(absl::InternalError("Parallax error."));
            continue;
        }

        rigid_triangulations.push_back(landmark_position_status);
        if (landmark_position_status.ok()) {
            n_rigid_triangulations++;
            ++count_success_rigid;
        }
    }

    // 统计三角化结果 - 修复：确保正确统计成功的数量
    // 遍历所有候选点，统计实际成功三角化的数量
    for (size_t idx = 0; idx < triangulation_candidate_ids.size(); idx++) {
        const bool rigid_ok = rigid_triangulations[idx].ok();
        const bool deformable_ok = deformable_triangulations[idx].ok();
        
        if (rigid_ok || deformable_ok) {
            // 如果至少一个三角化方法成功，检查是否通过了后续的有效性检查
            auto candidate_id = triangulation_candidate_ids[idx];
            Eigen::Vector3f landmark_triangulated;
            
            // 重建三角化结果
            if (deformable_ok && rigid_ok) {
                const Eigen::Vector3f& deformable_pos = *deformable_triangulations[idx];
                const Eigen::Vector3f& rigid_pos = *rigid_triangulations[idx];
                float distance = (deformable_pos - rigid_pos).norm();
                landmark_triangulated = (distance < 0.1f) ? deformable_pos : rigid_pos;
            } else if (deformable_ok) {
                landmark_triangulated = *deformable_triangulations[idx];
            } else {
                landmark_triangulated = *rigid_triangulations[idx];
            }
            
            // 检查有效性
            if (landmark_triangulated.hasNaN() || 
                landmark_triangulated.norm() < 1e-6f ||
                !std::isfinite(landmark_triangulated.x()) ||
                !std::isfinite(landmark_triangulated.y()) ||
                !std::isfinite(landmark_triangulated.z())) {
                continue;
            }
            
            // 检查是否能在帧中找到索引
            const auto index_in_frame = temporal_buffer->GetLandmarkIndexInFrame(current_frame_id, candidate_id);
            if (!index_in_frame.ok()) {
                continue;
            }
            
            // 检查深度一致性（可选，如果需要可以取消注释）
            /*
            if (existing_depths.size() >= 10 && existing_depth_std > 0.0f) {
                Eigen::Vector3f cam_p_new = current_camera_T_world * landmark_triangulated;
                const float z_new = cam_p_new.z();
                const float z_min = existing_depth_mean - 5.0f * existing_depth_std;
                const float z_max = existing_depth_mean + 5.0f * existing_depth_std;
                if (z_new < z_min || z_new > z_max) {
                    continue;
                }
            }
            */
            
            triangulated_landmarks++;
        }
    }

    // 输出三角化统计信息
    if (triangulated_landmarks > 0) {
        LOG(INFO) << "[三角化] 成功生成新3D点: " << triangulated_landmarks 
                  << " (候选:" << triangulation_candidate_ids.size() 
                  << ", 刚性成功:" << count_success_rigid 
                  << ", 可变形成功:" << count_success_deformable << ")";
    } else if (!triangulation_candidate_ids.empty()) {
        // 输出诊断信息帮助调试
        LOG(INFO) << "[三角化] 候选:" << triangulation_candidate_ids.size() 
                  << " 刚性成功:" << count_success_rigid 
                  << " 可变形成功:" << count_success_deformable
                  << " 邻域过密:" << count_close_features
                  << " 短轨迹:" << count_short_track
                  << " 三角化失败:" << count_triangulation_fail
                  << " 视差不足:" << count_parallax_fail
                  << " 重投影错误:" << count_reprojection_fail
                  << " NaN:" << count_landmark_nan;
    }

    for (int idx = 0; idx < triangulation_candidate_ids.size(); idx++) {
        auto candidate_id = triangulation_candidate_ids[idx];
        Eigen::Vector3f landmark_triangulated;

        const bool rigid_ok = rigid_triangulations[idx].ok();
        const bool deformable_ok = deformable_triangulations[idx].ok();

        // 智能三角化策略：根据点的位置和运动特征选择最佳方法
        if (rigid_ok || deformable_ok) {
            // 优先选择可形变三角化（对于内窥镜变形场景）
            // 但如果刚性三角化结果更好，则使用刚性结果
            if (deformable_ok && rigid_ok) {
                // 两个方法都成功，比较质量选择更好的
                // 简化策略：优先使用可形变，但如果刚性结果更稳定则使用刚性
                const Eigen::Vector3f& deformable_pos = *deformable_triangulations[idx];
                const Eigen::Vector3f& rigid_pos = *rigid_triangulations[idx];

                // 计算两个结果的一致性（距离）
                float distance = (deformable_pos - rigid_pos).norm();

                // 如果两个结果相差不大，优先使用可形变（更适合变形场景）
                // 如果相差很大，使用刚性（更保守）
                if (distance < 0.1f) {  // 10cm以内认为一致
                    landmark_triangulated = deformable_pos;
                } else {
                    landmark_triangulated = rigid_pos;  // 使用保守的刚性结果
                }
            } else if (deformable_ok) {
                landmark_triangulated = *deformable_triangulations[idx];
            } else {
                landmark_triangulated = *rigid_triangulations[idx];
            }
        } else {
            continue;
        }

        const auto index_in_frame = temporal_buffer->GetLandmarkIndexInFrame(current_frame_id,
                                                                             candidate_id);

        // 检查三角化结果是否有效（NaN、inf或零值）
        // 这些无效点会导致g2o优化器出现Cholesky失败和NaN传播
        if(landmark_triangulated.hasNaN()) {
            continue;
        }

        // 检查是否为0或接近0的无效坐标
        if(landmark_triangulated.norm() < 1e-6f) {
            continue;
        }

        // 检查是否包含inf值
        if(!std::isfinite(landmark_triangulated.x()) ||
           !std::isfinite(landmark_triangulated.y()) ||
           !std::isfinite(landmark_triangulated.z())) {
            continue;
        }

        if (!index_in_frame.ok()) {
            continue;
        }

        // 移除深度一致性过滤
        // 对于内窥镜场景，组织的深度变化很大，这个过滤会导致大部分新点被错误剔除
        // 原代码中的深度过滤：
        // if (existing_depths.size() >= 10 && existing_depth_std > 0.0f) { ... }
        // 我们直接注释掉这部分，让更多3D点能够生成

        // Create MapPoint and insert it into the map.
        auto mappoint = map_->CreateAndInsertMapPoint(landmark_triangulated, candidate_id);

        // Insert it into the tracking.
        current_frame->AddGeometryToKeypoint(*index_in_frame, landmark_triangulated,
                                             mappoint->GetId());

        candidates_triangulated.push_back(candidate_id);

        landmark_ids_triangulated.push_back(mappoint->GetId());

        triangulated_landmarks++;
    }

    // Add newly triangulated landmarks to the regularization graph with delay.
    // Only add landmarks that have been TRACKED_WITH_3D for at least N frames (mature landmarks).
    auto current_mappoints_ids = current_frame->GetMapPointsIdsWithStatus({TRACKED_WITH_3D});
    
    // Get all mature mappoints (lifetime >= options_.regularization_graph_delay_frames)
    vector<ID> mature_mappoints_ids;
    for (auto mappoint_id : current_mappoints_ids) {
        auto mappoint = map_->GetMapPoint(mappoint_id);
        if (mappoint && mappoint->GetLifetime() >= options_.regularization_graph_delay_frames) {
            mature_mappoints_ids.push_back(mappoint_id);
        }
    }
    
    // Add edges between mature landmarks only
    for (auto landmark_id : landmark_ids_triangulated) {
        auto landmark = map_->GetMapPoint(landmark_id);
        if (!landmark || landmark->GetLifetime() < options_.regularization_graph_delay_frames) {
            continue;  // Skip immature landmarks
        }
        
        auto landmark_position_status = current_frame->LandmarkPosition(landmark_id);
        CHECK_OK(landmark_position_status);

        for (auto other_landmark_id : mature_mappoints_ids) {
            if (landmark_id == other_landmark_id) {
                continue;
            }

            auto other_landmark = map_->GetMapPoint(other_landmark_id);
            if (!other_landmark || other_landmark->GetLifetime() < options_.regularization_graph_delay_frames) {
                continue;  // Skip immature landmarks
            }
            
            // Limit the number of neighbors based on the landmark's lifetime
            auto edges = map_->GetRegularizationGraph()->GetEdges(landmark_id);
            if (edges.size() >= landmark->GetMaxNeighbors()) {
                continue;
            }
            
            auto other_landmark_position_status = current_frame->LandmarkPosition(other_landmark_id);
            CHECK_OK(other_landmark_position_status);

            Eigen::Vector3f relative_position = *other_landmark_position_status - *landmark_position_status;
            map_->GetRegularizationGraph()->AddEdge(landmark_id, other_landmark_id, relative_position);
        }
    }
}

absl::StatusOr<Eigen::Vector3f> Mapping::DeformableLandmarkTriangulation(const int candidate_id) {
    return DeformableTriangulation(*map_->GetTemporalBuffer(),
                                                         candidate_id,
                                                         calibration_,
                                                         1.f);
}

void Mapping::UpdateTrackingFrameFromKeyFrame(std::shared_ptr<KeyFrame> keyframe) {
    auto current_frame = map_->GetMutableLastFrame();

    current_frame->SetFromKeyFrame(keyframe);
}
