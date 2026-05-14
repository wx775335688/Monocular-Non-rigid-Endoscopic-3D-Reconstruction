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

#include "g2o_optimization.h"

#include "optimization/landmark_vertex.h"
#include "optimization/position_regularizer.h"
#include "optimization/position_regularizer_with_deformation.h"
#include "optimization/reprojection_error.h"
#include "optimization/reprojection_error_only_deformation.h"
#include "optimization/reprojection_error_only_pose.h"
#include "optimization/reprojection_error_with_deformation.h"
#include "optimization/spatial_regularizer.h"
#include "optimization/spatial_regularizer_with_deformation.h"
#include "optimization/spatial_regularizer_with_observation.h"
#include "optimization/spatial_regularizer_fixed.h"
#include "utilities/geometry_toolbox.h"

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/container/btree_set.h"

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/core/robust_kernel_impl.h"

#include <cmath>

using namespace std;

void CameraPoseOptimization(Frame& frame, const Sophus::SE3f& previous_camera_transform_world) {
    // Create optimizer.
    g2o::SparseOptimizer optimizer;
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver =
            g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
    );

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    const float th_huber_2dof_squared = 5.99;
    const float th_huber_2dof = sqrt(th_huber_2dof_squared);

    // Set camera pose vertex.
    g2o::VertexSE3Expmap* camera_pose_vertex = new g2o::VertexSE3Expmap();
    Sophus::SE3f camera_transform_world = frame.CameraTransformationWorld();
    camera_pose_vertex->setEstimate(g2o::SE3Quat(
            camera_transform_world.unit_quaternion().cast<double>(),
            camera_transform_world.translation().cast<double>()));
    camera_pose_vertex->setId(0);

    optimizer.addVertex(camera_pose_vertex);

    vector<cv::KeyPoint> keypoints = frame.GetKeypointsWithStatus({TRACKED_WITH_3D});
    vector<Eigen::Vector3f> landmark_positions = frame.GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
    vector<ID> landmark_ids = frame.GetMapPointsIdsWithStatus({TRACKED_WITH_3D});

    // 过滤掉无效的3D点（坐标为0、NaN或inf）
    vector<cv::KeyPoint> valid_keypoints;
    vector<Eigen::Vector3f> valid_landmark_positions;
    vector<ID> valid_landmark_ids;

    for (size_t idx = 0; idx < keypoints.size(); ++idx) {
        const Eigen::Vector3f& pos = landmark_positions[idx];
        const cv::Point2f& px = keypoints[idx].pt;

        // 检查3D坐标是否有效（非零、非NaN、非inf）
        bool valid_3d = (pos.norm() > 1e-6) &&
                        !std::isnan(pos.x()) && !std::isnan(pos.y()) && !std::isnan(pos.z()) &&
                        !std::isinf(pos.x()) && !std::isinf(pos.y()) && !std::isinf(pos.z());

        // 检查2D坐标是否有效（非NaN）
        bool valid_2d = !std::isnan(px.x) && !std::isnan(px.y) &&
                        !std::isinf(px.x) && !std::isinf(px.y);

        // 额外检查：在相机坐标系下的深度是否合理
        // 使用更宽松的阈值以允许更多3D点参与优化
        bool valid_depth = false;
        if (valid_3d) {
            Eigen::Vector3f cam_pos = camera_transform_world.inverse() * pos;
            float depth = cam_pos.z();
            valid_depth = (depth > 0.0f) && (depth <= 50.0f) &&
                          std::isfinite(depth);
        }

        if (valid_3d && valid_2d && valid_depth) {
            valid_keypoints.push_back(keypoints[idx]);
            valid_landmark_positions.push_back(landmark_positions[idx]);
            valid_landmark_ids.push_back(landmark_ids[idx]);
        }
    }

    if (valid_keypoints.empty()) {
        LOG(WARNING) << "[相机位姿优化] 没有有效的3D-2D匹配点，跳过优化";
        return;
    }

    LOG(INFO) << "[相机位姿优化] 有效匹配点: " << valid_keypoints.size() << " / " << keypoints.size();

    vector<ReprojectionErrorOnlyPose*> reprojection_error_edges(valid_keypoints.size(), nullptr);

    for(int idx = 0; idx < valid_keypoints.size(); idx++){
        cv::Point2f pixel_coordinates = valid_keypoints[idx].pt;
        Eigen::Matrix<double,2,1> observation(pixel_coordinates.x, pixel_coordinates.y);

        ReprojectionErrorOnlyPose* edge = new ReprojectionErrorOnlyPose();

        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        edge->setMeasurement(observation);
        edge->setInformation(Eigen::Matrix2d::Identity());

        g2o::RobustKernelHuber* robust_kernel = new g2o::RobustKernelHuber;
        edge->setRobustKernel(robust_kernel);
        robust_kernel->setDelta(th_huber_2dof);

        edge->calibration_ = frame.GetCalibration();
        edge->landmark_world_ = valid_landmark_positions[idx].cast<double>();

        optimizer.addEdge(edge);
        reprojection_error_edges[idx] = edge;
    }

    vector<int> iterations = {10, 10, 10};
    vector<bool> inliers(valid_keypoints.size(), true);

    for(int iteration = 0; iteration < iterations.size(); iteration++){
        // Reset position to the initial seed.
        camera_pose_vertex->setEstimate(g2o::SE3Quat(
                camera_transform_world.unit_quaternion().cast<double>(),
                camera_transform_world.translation().cast<double>()));

        optimizer.initializeOptimization(0);
        optimizer.optimize(iterations[iteration]);

        for(int idx = 0; idx < reprojection_error_edges.size(); idx++){
            ReprojectionErrorOnlyPose* edge = reprojection_error_edges[idx];
            if(!edge)
                continue;

            if(!inliers[idx]) {
                edge->computeError();
            }

            const float chi_squared = edge->chi2();

            if(chi_squared > th_huber_2dof_squared) {
                inliers[idx] = false;
                edge->setLevel(1);
            }
            else {
                inliers[idx] = true;
                edge->setLevel(0);
            }

            // Deactivate robust kernel after 2 iterations as we should have removed
            // all the inlier observations.
            if(iteration == 2){
                edge->setRobustKernel(0);
            }
        }
    }

    // Recover the optimized camera pose.
    frame.MutableCameraTransformationWorld() = Sophus::SE3f(
            camera_pose_vertex->estimate().to_homogeneous_matrix().cast<float>());
}

absl::flat_hash_set<ID> CameraPoseAndDeformationOptimization(Frame& current_frame,
                                                     std::shared_ptr<Map> map,
                                                     const Sophus::SE3f& previous_camera_transform_world,
                                                     const float scale,
                                                     const absl::flat_hash_map<ID, std::pair<float, float>>* hetero_params) {
    // Create optimizer.
    g2o::SparseOptimizer optimizer;
    std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver =  g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver))
    );

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // Set camera vertex
    g2o::VertexSE3Expmap* camera_pose_vertex = new g2o::VertexSE3Expmap();
    Sophus::SE3f camera_transform_world = current_frame.CameraTransformationWorld();
    camera_pose_vertex->setEstimate(g2o::SE3Quat(
            camera_transform_world.unit_quaternion().cast<double>(),
            camera_transform_world.translation().cast<double>()));
    camera_pose_vertex->setId(0);
    camera_pose_vertex->setFixed(false);

    optimizer.addVertex(camera_pose_vertex);

    // 获取3D点，但需要过滤掉无效的坐标（零值或NaN）
    vector<cv::KeyPoint> keypoints_all = current_frame.GetKeypointsWithStatus({TRACKED_WITH_3D});
    vector<Eigen::Vector3f> landmark_positions_all = current_frame.GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
    vector<ID> mappoints_ids_all = current_frame.GetMapPointsIdsWithStatus({TRACKED_WITH_3D});

    // 过滤掉无效的地图点（坐标为0或包含NaN）
    vector<cv::KeyPoint> keypoints;
    vector<Eigen::Vector3f> landmark_positions;
    vector<ID> mappoints_ids;
    for (size_t i = 0; i < keypoints_all.size(); ++i) {
        const Eigen::Vector3f& pos = landmark_positions_all[i];
        // 检查坐标是否为0或包含NaN/Inf
        bool valid = (pos.norm() > 1e-6f) &&  // 非零
                     std::isfinite(pos.x()) && std::isfinite(pos.y()) && std::isfinite(pos.z()) &&
                     (std::abs(pos.x()) > 1e-6f || std::abs(pos.y()) > 1e-6f || std::abs(pos.z()) > 1e-6f);
        
        // 额外检查：在相机坐标系下的深度是否合理
        // 注意：depth <= 0 表示点在相机后面（无效），depth过小可能导致数值不稳定
        if (valid) {
            Eigen::Vector3f cam_pos = camera_transform_world.inverse() * pos;
            float depth = cam_pos.z();
            // 深度必须为正且在合理范围内（1mm到50m）
            // 使用更宽松的阈值以允许更多3D点参与优化
            if (depth <= 0.0f || depth > 50.0f || std::isnan(depth) || std::isinf(depth)) {
                valid = false;
            }
        }
        
        if (valid) {
            keypoints.push_back(keypoints_all[i]);
            landmark_positions.push_back(landmark_positions_all[i]);
            mappoints_ids.push_back(mappoints_ids_all[i]);
        }
    }

    // 如果过滤后没有有效的3D点，返回空集合
    if (keypoints.empty()) {
        LOG(WARNING) << "No valid 3D points for optimization, skipping...";
        absl::flat_hash_set<ID> empty_set;
        return empty_set;
    }

    absl::flat_hash_map<ID, int> mappoint_id_to_index;
    const int points_in_optimization = keypoints.size();

    auto regularization_graph = map->GetRegularizationGraph();

    // Set point vertices.
    vector<LandmarkVertex*> deformation_vertices(points_in_optimization, nullptr);
    for (int idx = 0; idx < points_in_optimization; idx++) {
        deformation_vertices[idx] = new LandmarkVertex();
        deformation_vertices[idx]->setId(idx + 1);
        deformation_vertices[idx]->setToOrigin();

        optimizer.addVertex(deformation_vertices[idx]);

        mappoint_id_to_index[mappoints_ids[idx]] = idx;
    }

    // Set error terms.
    const int regularizers_per_point = 10;

    const float th_huber_2dof_squared = 9.21f;  // 合理放宽重投影误差阈值到卡方2自由度99%置信区间，平衡精度和鲁棒性
    const float th_huber_2dof = sqrt(th_huber_2dof_squared);

    const float th_huber_3dof_squared = 0.584;
    const float th_huber_3dof = sqrt(th_huber_3dof_squared);

    const float sigma_reprojection = 0.5;   // pixels.
    const float info_reprojection = 1.0f / (sigma_reprojection * sigma_reprojection);

    float sigma_position = 0.1f;
    float info_position = 1.0f / (sigma_position * sigma_position);

    const float sigma_spatial = 0.1 * scale;   // mm.
    const float info_spatial = 1.0f / (sigma_spatial * sigma_spatial);

    vector<absl::flat_hash_map<int, SpatialRegularizerWithDeformation*>> spatial_regularizers(points_in_optimization);
    vector<absl::flat_hash_map<int, PositionRegularizerWithDeformation*>> position_regularizers(points_in_optimization);
    vector<ReprojectionErrorWithDeformation*> reprojection_errors(points_in_optimization);

    int n_reprojection_edges = 0;
    int n_spatial_edges = 0;
    int n_position_edges = 0;

    vector<vector<ID>> connected_mappoint_ids(points_in_optimization);

    absl::btree_set<ID> lost_mappoint_ids_ordered;
    absl::flat_hash_set<ID> lost_mappoint_ids;
    for (int idx = 0; idx < points_in_optimization; idx++) {
        // Set reprojection error.
        ReprojectionErrorWithDeformation* reprojection_error = new ReprojectionErrorWithDeformation();

        cv::Point2f pixel_coordinates = keypoints[idx].pt;
        Eigen::Matrix<double,2,1> observation(pixel_coordinates.x, pixel_coordinates.y);

        reprojection_error->setMeasurement(observation);

        reprojection_error->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        reprojection_error->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(idx+1)));

        reprojection_error->setInformation(Eigen::Matrix2d::Identity() * info_reprojection);

        g2o::RobustKernelHuber* robust_kernel = new g2o::RobustKernelHuber;
        robust_kernel->setDelta(th_huber_2dof);
        reprojection_error->setRobustKernel(robust_kernel);

        reprojection_error->calibration_ = current_frame.GetCalibration();
        reprojection_error->landmark_world_ = landmark_positions[idx].cast<double>();

        optimizer.addEdge(reprojection_error);
        reprojection_errors[idx] = reprojection_error;

        n_reprojection_edges++;

        // Set spatial regularizers.
        const ID mappoint_id = mappoints_ids[idx];
        auto regularization_edges =
                regularization_graph->GetEdges(mappoint_id);

        int n_regularizers = 0;
        for(const auto& [mappoint_id_other, regularization_edge] : regularization_edges) {
            // Check if there is already enough regularizers or if the connection is good.
            if (n_regularizers > regularizers_per_point ||
                regularization_edge->status == RegularizationGraph::BAD) {
                break;
            }

            // Check that the connected point is also being optimized.
            if (!current_frame.MapPointIdToIndex().contains(mappoint_id_other) ||
                current_frame.LandmarkStatuses()[current_frame.MapPointIdToIndex().at(mappoint_id_other)] != TRACKED_WITH_3D) {
                if (current_frame.MapPointIdToIndex().contains(mappoint_id_other) &&
                    current_frame.LandmarkStatuses()[current_frame.MapPointIdToIndex().at(mappoint_id_other)] != JUST_TRIANGULATED) {
                    lost_mappoint_ids.insert(mappoint_id_other);
                    lost_mappoint_ids_ordered.insert(mappoint_id_other);
                }

                continue;
            }

            // Check if this regularizer has already been inserted in the optimization.
            const int idx_other = mappoint_id_to_index[mappoint_id_other];
            if (spatial_regularizers[idx].contains(idx_other)) {
                continue;
            }

            // Set spatial regularizer.
            auto spatial_regularizer = new SpatialRegularizerWithDeformation();
            spatial_regularizer->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                    optimizer.vertex(idx + 1)));
            spatial_regularizer->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                    optimizer.vertex(idx_other + 1)));

            Eigen::Matrix3d spatial_information_matrix = Eigen::Matrix3d::Identity() * info_spatial;
            spatial_regularizer->setInformation(spatial_information_matrix);

            g2o::RobustKernelHuber* robust_kernel = new g2o::RobustKernelHuber;
            robust_kernel->setDelta(th_huber_3dof);
            spatial_regularizer->setRobustKernel(robust_kernel);

            // 应用异质粘弹性参数（创新点2）
            float base_weight = regularization_edge->weight;
            if (hetero_params != nullptr) {
                // 查找两个点的异质参数
                auto it_i = hetero_params->find(mappoint_id);
                auto it_j = hetero_params->find(mappoint_id_other);
                
                if (it_i != hetero_params->end() && it_j != hetero_params->end()) {
                    // 使用两个点的k值的几何平均来调整权重，更适合弹性系数的物理意义
                    float k_i = std::max(it_i->second.first, 1e-6f); // 避免零或负值
                    float k_j = std::max(it_j->second.first, 1e-6f);
                    float geo_mean_k = std::sqrt(k_i * k_j); // 几何平均
                    // 权重乘以几何平均k值（k越大，约束越强），限制范围防止失控
                    geo_mean_k = std::min(geo_mean_k, 10.0f); // 放宽最大限制到10
                    geo_mean_k = std::max(geo_mean_k, 0.05f); // 放宽最小限制到0.05
                    base_weight *= geo_mean_k;
                }
            }
            
            // 限制权重范围，防止过度约束或约束失效
            base_weight = std::min(base_weight, 20.0f);  // 放宽最大权重限制
            base_weight = std::max(base_weight, 0.1f);   // 提高最小权重阈值
            
            spatial_regularizer->weight_ = base_weight;

            optimizer.addEdge(spatial_regularizer);

            spatial_regularizers[idx][idx_other] = spatial_regularizer;
            spatial_regularizers[idx_other][idx] = spatial_regularizer;

            connected_mappoint_ids[idx].push_back(mappoint_id_other);
            connected_mappoint_ids[idx_other].push_back(mappoint_id);
            n_spatial_edges++;

            n_regularizers++;

            // Set position regularizer.
            PositionRegularizerWithDeformation* position_regularizer = new PositionRegularizerWithDeformation();
            position_regularizer->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                    optimizer.vertex(idx + 1)));
            position_regularizer->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                    optimizer.vertex(idx_other + 1)));

            position_regularizer->setMeasurement(regularization_edge->first_distance);

            Eigen::Matrix<double, 1, 1> position_information_matrix =
                    Eigen::Matrix<double, 1, 1>::Identity() * info_position;
            position_regularizer->setInformation(position_information_matrix);

            position_regularizer->rest_position_1_ = landmark_positions[idx].cast<double>();
            position_regularizer->rest_position_2_ = landmark_positions[idx_other].cast<double>();

            g2o::RobustKernelHuber* robust_kernel_position = new g2o::RobustKernelHuber;
            robust_kernel_position->setDelta(th_huber_3dof);
            position_regularizer->setRobustKernel(robust_kernel_position);

            position_regularizer->k_ = 1.1f;

            optimizer.addEdge(position_regularizer);

            n_position_edges++;

            position_regularizers[idx][idx_other] = position_regularizer;
            position_regularizers[idx_other][idx] = position_regularizer;
        }
    }
    vector<int> iterations = {10, 10};
    vector<bool> inliers(points_in_optimization, true);

    for (int iteration = 0; iteration < iterations.size(); iteration++){
        int n_good_regularizers = 0;

        // Reset estimations.
        camera_pose_vertex->setEstimate(g2o::SE3Quat(
                camera_transform_world.unit_quaternion().cast<double>(),
                camera_transform_world.translation().cast<double>()));

        for(auto vertex : deformation_vertices){
            if(!vertex) {
                continue;
            } else{
                vertex->setToOrigin();
            }
        }

        optimizer.initializeOptimization(0);
        optimizer.optimize(iterations[iteration]);

        // Check reprojection errors.
        for(int idx = 0; idx < points_in_optimization; idx++){
            ReprojectionErrorWithDeformation* reprojection_error = reprojection_errors[idx];

            reprojection_error->computeError();

            const float chi_squared = reprojection_error->chi2();

            if(chi_squared > th_huber_2dof_squared) {
                inliers[idx] = false;

                reprojection_error->setLevel(1);
                for (auto& [idx_other, spatial_regularizer] : spatial_regularizers[idx]) {
                    spatial_regularizer->setLevel(1);
                }
            } else {
                inliers[idx] = true;

                reprojection_error->setLevel(0);
                for (auto& [idx_other, spatial_regularizer] : spatial_regularizers[idx]) {
                    spatial_regularizer->setLevel(0);
                }
            }

            // Check spatial regularizers.
            for (auto& [idx_other, spatial_regularizer] : spatial_regularizers[idx]) {
                spatial_regularizer->computeError();

                if(spatial_regularizer->chi2() > th_huber_3dof_squared) {
                    spatial_regularizer->setLevel(1);
                } else {
                    spatial_regularizer->setLevel(0);
                }
            }
        }
    }

    // Recover the optimized camera pose.
    current_frame.MutableCameraTransformationWorld() = Sophus::SE3f(
            camera_pose_vertex->estimate().to_homogeneous_matrix().cast<float>());

    vector<float> deformation_magnitudes;
    for(int idx = 0; idx < points_in_optimization; idx++) {
        LandmarkVertex* deformation_vertex = static_cast<LandmarkVertex*>(optimizer.vertex(idx+1));
        Eigen::Vector3f deformation = deformation_vertex->estimate().cast<float>();
        deformation_magnitudes.push_back(deformation.norm());
    }

    vector<float> sorted_deformation_magnitudes = deformation_magnitudes;
    sort(sorted_deformation_magnitudes.begin(), sorted_deformation_magnitudes.end());
    size_t n = sorted_deformation_magnitudes.size();
    float interquartileRange = 0.0f;
    float q3 = 0.0f;
    if (n >= 4) {  // 至少需要4个点来计算有意义的IQR
        size_t q1_idx = std::max(size_t(0), std::min(n-1, size_t(n * 0.25f)));
        size_t q3_idx = std::max(size_t(0), std::min(n-1, size_t(n * 0.75f)));
        float q1 = sorted_deformation_magnitudes[q1_idx];
        q3 = sorted_deformation_magnitudes[q3_idx];
        interquartileRange = q3 - q1;
    } else {
        // 对于少量数据，使用简单的范围估计
        if (n > 1) {
            interquartileRange = sorted_deformation_magnitudes.back() - sorted_deformation_magnitudes.front();
            q3 = sorted_deformation_magnitudes.back();
        } else if (n == 1) {
            q3 = sorted_deformation_magnitudes[0];
        }
    }

    float th_ = 1.5f * interquartileRange;

    // Update point positions.
    for(int idx = 0; idx < points_in_optimization; idx++) {
        ReprojectionErrorWithDeformation* reprojection_error = reprojection_errors[idx];
        reprojection_error->computeError();

        int index_in_frame = current_frame.MapPointIdToIndex().at(mappoints_ids[idx]);

        const float chi_squared = reprojection_error->chi2();
        if(chi_squared > th_huber_2dof_squared) {
            inliers[idx] = false;

            // 对于重投影误差过大的点，使用智能不良记录系统
            auto mappoint = map->GetMapPoint(mappoints_ids[idx]);
            if (mappoint) {
                // 保护期内的点不增加不良记录
                if (!mappoint->IsProtected()) {
                    mappoint->IncrementBadCount();
                }
                int lifetime = mappoint->GetLifetime();

                // 根据点的成熟度和保护状态动态调整不良记录阈值
                int bad_threshold = 5;  // 默认5次
                if (mappoint->IsProtected()) {
                    bad_threshold = 10;  // 保护期内的点极度宽容
                } else if (lifetime > 30) {
                    bad_threshold = 8;  // 非常成熟的点更宽容
                } else if (lifetime > 15) {
                    bad_threshold = 6;  // 中期点适度宽容
                }

                if (mappoint->GetBadCount() >= bad_threshold) {
                    current_frame.LandmarkStatuses()[index_in_frame] = BAD;
                } else {
                    // 暂时保持状态，给机会恢复
                    // 对于重投影误差稍大的情况，不立即改变状态
                    if (chi_squared > 2 * th_huber_2dof_squared) {
                        // 误差很大的情况才可能降级
                        current_frame.LandmarkStatuses()[index_in_frame] = TRACKED_WITH_3D;
                    }
                    // 误差适中的情况保持现状
                }
            }
        } else {
            // 重投影误差正常，逐渐减少不良记录
            auto mappoint = map->GetMapPoint(mappoints_ids[idx]);
            if (mappoint && mappoint->GetBadCount() > 0) {
                // 表现良好时逐渐减少不良记录，而不是立即清零
                // 这样可以避免单次良好表现就"洗白"长期不良记录
                if (mappoint->GetBadCount() <= 1) {
                    mappoint->ResetBadCount();  // 小问题立即清除
                } else {
                    // 对于多次不良记录，每次良好表现减少1个不良记录
                    mappoint->DecrementBadCount();
                }
            }
        }

        LandmarkVertex* deformation_vertex = static_cast<LandmarkVertex*>(optimizer.vertex(idx+1));
        Eigen::Vector3f deformation = deformation_vertex->estimate().cast<float>();

        // 对于已经三角化成功的点，只有在变形非常大时才降级
        // 避免频繁的状态切换
        if (deformation.norm() >= q3 + th_) {
            // 检查当前点的状态，如果是TRACKED_WITH_3D且生命周期较长，则更宽容
            auto current_status = current_frame.LandmarkStatuses()[index_in_frame];
            auto mappoint = map->GetMapPoint(mappoints_ids[idx]);
            bool should_downgrade = true;

            if (current_status == TRACKED_WITH_3D && mappoint && mappoint->GetLifetime() > 10) {
                // 对于成熟的3D点，只有在变形极大时才降级
                if (deformation.norm() < q3 + 2 * th_) {
                    should_downgrade = false;
                }
            }

            if (should_downgrade) {
                current_frame.LandmarkStatuses()[index_in_frame] = TRACKED;
            }
            continue;
        }

        deformation_vertex->setFixed(true);

        Eigen::Vector3f previous_landmark_position = landmark_positions[idx];
        Eigen::Vector3f current_landmark_position = deformation + previous_landmark_position;

        current_frame.LandmarkPositions()[index_in_frame] = current_landmark_position;

        map->GetMapPoint(mappoints_ids[idx])->SetLastWorldPosition(current_landmark_position);

        ID mappoint_id = mappoints_ids[idx];
    }

    const int median_idx = deformation_magnitudes.size() / 2;
    nth_element(deformation_magnitudes.begin(), deformation_magnitudes.begin() + median_idx,
                deformation_magnitudes.end());

    current_frame.SetDeformationMaginitud(deformation_magnitudes[median_idx]);

    // Update regularization graph.
    for(int idx = 0; idx < points_in_optimization; idx++) {
        if (!inliers[idx]) {
            continue;
        }

        ID mappoint_id = mappoints_ids[idx];

        int index_in_frame = current_frame.MapPointIdToIndex().at(mappoints_ids[idx]);
        Eigen::Vector3f landmark_position = current_frame.LandmarkPositions()[index_in_frame];

        int good_connections = regularization_graph->UpdateVertex(mappoint_id);

        // 平衡连接质量和点云密度，避免过度严格导致点云稀疏
        float connection_threshold = 0.03f; // 进一步放宽连接阈值

        // 获取地图点并检查是否有效
        auto mappoint = map->GetMapPoint(mappoint_id);
        if (mappoint) {
            int lifetime = mappoint->GetLifetime();
            // 对于新点适当放宽，但不至于过低
            connection_threshold = (lifetime < 3) ? 0.02f : 0.03f;
        }
        
        if (good_connections < regularizers_per_point * connection_threshold) {
    // 删除这个冗余的输出
    /*
    LOG(INFO) << "Removing features because graph error: ";
    */
            current_frame.LandmarkStatuses()[index_in_frame] = BAD;
        }
    }

    if (lost_mappoint_ids.empty()) {
        return lost_mappoint_ids;
    }

    int current_id = points_in_optimization + 2;
    absl::flat_hash_map<ID, int> lost_mappoint_id_to_index;

    for (ID lost_mappoint_id : lost_mappoint_ids_ordered) {
        LandmarkVertex* deformation_vertex = new LandmarkVertex();
        deformation_vertex->setId(current_id);
        deformation_vertex->setToOrigin();

        lost_mappoint_id_to_index[lost_mappoint_id] = current_id;

        current_id++;

        optimizer.addVertex(deformation_vertex);

        // Add regularization edges.
        auto regularization_edges =
                regularization_graph->GetEdges(lost_mappoint_id);

        int n_regularizers = 0;
        for(const auto& [mappoint_id_other, regularization_edge] : regularization_edges) {
            if (n_regularizers > 10) {
                break;
            }

            if (!mappoint_id_to_index.contains(mappoint_id_other)) {
                continue;
            }

            const int idx_other = mappoint_id_to_index[mappoint_id_other];

            float weight = regularization_edge->weight;

            SpatialRegularizerFixed* spatial_regularizer = new SpatialRegularizerFixed();
            spatial_regularizer->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                    optimizer.vertex(current_id - 1)));

            Eigen::Matrix3d spatial_information_matrix = Eigen::Matrix3d::Identity() * info_spatial;
            spatial_regularizer->setInformation(spatial_information_matrix);

            spatial_regularizer->flow_fixed = deformation_vertices[idx_other];
            spatial_regularizer->id1 = lost_mappoint_id;
            spatial_regularizer->id2 = mappoint_id_other;

            g2o::RobustKernelHuber* robust_kernel = new g2o::RobustKernelHuber;
            robust_kernel->setDelta(th_huber_3dof);
            spatial_regularizer->setRobustKernel(robust_kernel);

            spatial_regularizer->weight_ = weight;

            optimizer.addEdge(spatial_regularizer);

            n_regularizers++;
        }
    }

    camera_pose_vertex->setFixed(true);

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    lost_mappoint_ids.clear();
    for (const auto& [mappoint_id, idx] : lost_mappoint_id_to_index) {
        auto mappoint = map->GetMapPoint(mappoint_id);

        LandmarkVertex* deformation_vertex = static_cast<LandmarkVertex*>(optimizer.vertex(idx));
        Eigen::Vector3f deformation = deformation_vertex->estimate().cast<float>();

        Eigen::Vector3f previous_landmark_position = mappoint->GetLastWorldPosition();
        Eigen::Vector3f current_landmark_position = deformation + previous_landmark_position;

        mappoint->SetLastWorldPosition(current_landmark_position);

        lost_mappoint_ids.insert(mappoint_id);
    }

    return lost_mappoint_ids;

}

absl::StatusOr<Eigen::Vector3f> DeformableTriangulation(TemporalBuffer& temporal_buffer,
                                                        int candidate_id,
                                                        std::shared_ptr<CameraModel> calibration,
                                                        const float scale) {
    // Recover feature track.
    auto candidate_track = temporal_buffer.GetFeatureTrack(candidate_id);

    // Get candidate neighbors.
    // 原实现中使用 min_image_distance = 20 像素，在特征密集的内窥镜场景下会过滤掉大量候选点。
    // 为了让非刚性三角化也能更密集地利用邻域点，这里与刚性三角化保持一致，放宽到 10 像素。
    auto neighbour_ids = temporal_buffer.GetClosestMapPointsToFeature(candidate_id, 10, 10, 500);

    if (neighbour_ids.empty()) {
        return absl::InternalError("Feature too close to other ones.");
    }

    int last_frame_id = candidate_track.back().first;
    int first_frame_id = candidate_track.front().first;

    // Create optimizer.
    g2o::SparseOptimizer optimizer;
    std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver =  g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver))
    );

    optimizer.setAlgorithm(solver);

    const float sigma_reprojection = 0.5;   // pixels.
    const float info_reprojection = 1.0f / (sigma_reprojection * sigma_reprojection);

    // Set up vertices.
    absl::flat_hash_map<int, LandmarkVertex*> landmark_vertices;
    absl::flat_hash_map<int, Eigen::Vector3d> landmark_seeds;
    absl::flat_hash_map<int, ReprojectionErrorOnlyDeformation*> frame_id_to_reprojection_edge;

    const auto& [current_frame_id, current_keypoint] = candidate_track.front();
    const auto& [previous_frame_id, previous_keypoint] = candidate_track.back();

    // Unproject rays.
    Eigen::Vector3f current_ray =
            calibration->Unproject(current_keypoint.pt.x, current_keypoint.pt.y).normalized();
    Eigen::Vector3f previous_ray =
            calibration->Unproject(previous_keypoint.pt.x, previous_keypoint.pt.y).normalized();

    // Get camera poses.
    auto current_camera_transform_world = temporal_buffer.GetCameraTransformWorld(current_frame_id);
    auto previous_camera_transform_world = temporal_buffer.GetCameraTransformWorld(previous_frame_id);

    // Rigid triangulation.
    auto landmark_position_status =
            TriangulateMidPoint(previous_ray, current_ray,
                                *previous_camera_transform_world, *current_camera_transform_world);

    if (!landmark_position_status.ok()) {
        return absl::InternalError(landmark_position_status.status().message());
    }

    Eigen::Vector3f landmark_current_position = (*current_camera_transform_world) * (*landmark_position_status);
    cv::Point2f projected_landmark_1 = calibration->Project(landmark_current_position);

    // 与刚性三角化一致，放宽重投影误差阈值到约 9.21（卡方 2 自由度 99% 置信区间），
    // 提高真实噪声条件下非刚性三角化的通过率。
    if(SquaredReprojectionError(current_keypoint.pt, projected_landmark_1) > 9.21f){
        return absl::InternalError("High reprojection error at first camera.");
    }

    Eigen::Vector3f landmark_previous_position = (*previous_camera_transform_world) * (*landmark_position_status);
    cv::Point2f projected_landmark_2 = calibration->Project(landmark_previous_position);

    if(SquaredReprojectionError(previous_keypoint.pt, projected_landmark_2) > 9.21f){
        return absl::InternalError("High reprojection error at second camera.");
    }

    Eigen::Vector3f normal_1 = (*landmark_position_status) - (*current_camera_transform_world).inverse().translation();
    Eigen::Vector3f normal_2 = (*landmark_position_status) - (*previous_camera_transform_world).inverse().translation();
    float parallax = RaysParallax(normal_1, normal_2);

    if(parallax < 0.0025 * 5.f){
        return absl::InternalError("Low parallax.");
    }

    for (const auto& [frame_id, keypoint] : candidate_track) {
        auto camera_transform_world = temporal_buffer.GetCameraTransformWorld(frame_id);
        CHECK_OK(camera_transform_world);

        vector<float> neighbor_depths;
        neighbor_depths.reserve(neighbour_ids.size());
        for (const auto neighbor_id : neighbour_ids) {
            auto landmark_position = temporal_buffer.GetLandmarkPosition(frame_id, neighbor_id);

            if(landmark_position.ok()) {
                float depth = ((*camera_transform_world) * (*landmark_position)).z();
                // 只考虑合理的深度值（正值且不太极端）
                if (depth > 1e-3f && depth < 100.0f) {
                    neighbor_depths.push_back(depth);
                }
            }
        }

        if (neighbor_depths.empty()) {
            return absl::InternalError("Found no valid neighbours in a temporal point.");
        }

        // 使用中位数而不是平均值，更鲁棒地处理异常值
        sort(neighbor_depths.begin(), neighbor_depths.end());
        float depth_seed;
        size_t n = neighbor_depths.size();
        if (n % 2 == 0) {
            depth_seed = (neighbor_depths[n/2 - 1] + neighbor_depths[n/2]) / 2.0f;
        } else {
            depth_seed = neighbor_depths[n/2];
        }

        if (depth_seed < 0) {
            return absl::InternalError("Negative initial depth.");
        }

        Eigen::Vector3d landmark_position_seed = (calibration->Unproject(keypoint.pt) * depth_seed)
                .cast<double>();

        LandmarkVertex* landmark_vertex = new LandmarkVertex();
        landmark_vertex->setId(frame_id);
        landmark_vertex->setEstimate(landmark_position_seed);

        optimizer.addVertex(landmark_vertex);
        landmark_vertices[frame_id] = landmark_vertex;
        landmark_seeds[frame_id] = landmark_position_seed;

        // Set reprojection error.
        ReprojectionErrorOnlyDeformation* reprojection_error =
                new ReprojectionErrorOnlyDeformation();

        cv::Point2f pixel_coordinates = keypoint.pt;
        Eigen::Matrix<double,2,1> observation(pixel_coordinates.x, pixel_coordinates.y);
        reprojection_error->setMeasurement(observation);

        reprojection_error->setVertex(0, optimizer.vertex(frame_id));

        Eigen::Matrix2d informationMatrix = Eigen::Matrix2d::Identity() * info_reprojection;
        reprojection_error->setInformation(informationMatrix);

        reprojection_error->calibration_ = calibration;

        optimizer.addEdge(reprojection_error);
        frame_id_to_reprojection_edge[frame_id] = reprojection_error;
    }

    const float th_huber_3dof_squared = 7.815;
    const float th_huber_3dof = sqrt(th_huber_3dof_squared);

    // Set up regularization edges.
    const float sigma_spatial = 0.1;
    const float info_spatial = 1.0f / (sigma_spatial * sigma_spatial);

    vector<SpatialRegularizerWithObservation*> regularization_terms;
    for (auto current_iterator = candidate_track.begin();
         current_iterator != candidate_track.end(); current_iterator++) {
        int current_frame_id = current_iterator->first;

        auto current_camera_transform_world = temporal_buffer.GetCameraTransformWorld(current_frame_id);
        CHECK_OK(current_camera_transform_world);

        g2o::SE3Quat current_pose_g2o = g2o::SE3Quat(
                (*current_camera_transform_world).inverse().unit_quaternion().cast<double>(),
                (*current_camera_transform_world).inverse().translation().cast<double>());

        for (auto next_iterator = next(current_iterator);
             next_iterator != candidate_track.end(); next_iterator++) {
            int next_frame_id = next_iterator->first;

            auto next_camera_transform_world = temporal_buffer.GetCameraTransformWorld(next_frame_id);
            CHECK_OK(next_camera_transform_world);

            g2o::SE3Quat next_pose_g2o = g2o::SE3Quat(
                    (*next_camera_transform_world).inverse().unit_quaternion().cast<double>(),
                    (*next_camera_transform_world).inverse().translation().cast<double>());

            for (auto neighbor_id : neighbour_ids) {
                auto current_landmark_position = temporal_buffer.GetLandmarkPosition(current_frame_id,
                                                                                     neighbor_id);
                auto next_landmark_position = temporal_buffer.GetLandmarkPosition(next_frame_id,
                                                                                  neighbor_id);

                auto first_landmark_position = temporal_buffer.GetLandmarkPosition(first_frame_id,
                                                                                  neighbor_id);

                if (!current_landmark_position.ok() || !next_landmark_position.ok() ||
                    !first_landmark_position.ok()) {
                    continue;
                }

                Eigen::Vector3f flow = (*next_landmark_position) - (*current_landmark_position);

                SpatialRegularizerWithObservation* spatial_regularizer =
                        new SpatialRegularizerWithObservation();

                spatial_regularizer->setMeasurement(flow.cast<double>());

                spatial_regularizer->setVertex(0, optimizer.vertex(current_frame_id));
                spatial_regularizer->setVertex(1, optimizer.vertex(next_frame_id));

                Eigen::Matrix3d informationMatrix = Eigen::Matrix3d::Identity() * info_spatial;
                spatial_regularizer->setInformation(informationMatrix);

                spatial_regularizer->weight_ = 1.0f;

                spatial_regularizer->next_world_transform_camera_ = next_pose_g2o;
                spatial_regularizer->current_world_transform_camera_ = current_pose_g2o;

                optimizer.addEdge(spatial_regularizer);

                regularization_terms.push_back(spatial_regularizer);
            }
        }
    }

    if(optimizer.edges().size() == 0 || optimizer.vertices().size() == 0) {
        return absl::InternalError("Optimization is empty.");
    }

    // optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // Remove outlier regularization terms.
    int bad_edges = 0;
    for (auto regularization_edge : regularization_terms) {
        regularization_edge->computeError();

        regularization_edge->setRobustKernel(nullptr);

        if (regularization_edge->chi2() > th_huber_3dof_squared) {
            regularization_edge->setLevel(1);
            bad_edges++;
        }
    }

    // 原始实现中要求超过 50% 正则项为 outlier 就直接判定失败，
    // 在高度非刚性的内窥镜场景中过于激进，这里放宽到 70% 以上才判定失败。
    if ((float)bad_edges / (float)regularization_terms.size() > 0.7f) {
        return absl::InternalError("Triangulation has to many bad neighbors.");
    }

    int n_bad_edges = 0;

    for (const auto [id, reprojection_edge] : frame_id_to_reprojection_edge) {
        reprojection_edge->computeError();
        if (reprojection_edge->chi2() > 5.99 * 10) {
            n_bad_edges++;
        }
    }

    // 同理，对重投影误差的全局“失败率”阈值从 50% 放宽到 70%，
    // 以避免因为少量边界/遮挡导致的局部误差而彻底放弃一条轨迹。
    if ((float) n_bad_edges / (float) optimizer.vertices().size() > 0.7f) {
        return absl::InternalError("Triangulation has to much error.");
    }

    cv::Point2f latest_pixel_coordinates = candidate_track.back().second.pt;
    float current_depth = landmark_vertices[last_frame_id]->estimate().z();

    auto last_camera_transform_world = temporal_buffer.GetCameraTransformWorld(last_frame_id);
    CHECK_OK(last_camera_transform_world);

    Eigen::Vector3f unprojected_keypoint = calibration->Unproject(latest_pixel_coordinates);
    unprojected_keypoint /= unprojected_keypoint.z();

    Eigen::Vector3f triangulated_landmark_position =
            ((*last_camera_transform_world).inverse() * (unprojected_keypoint * current_depth));

    return triangulated_landmark_position;
}

class TemporalPoint {
public:
    TemporalPoint() = delete;

    TemporalPoint(ID mappoint_id_1, ID mappoint_id_2, ID keyframe_id_1, ID keyframe_id_2){
        mappoint_id_1_ = min(mappoint_id_1, mappoint_id_2);
        mappoint_id_2_ = max(mappoint_id_1, mappoint_id_2);
        keyframe_id_1_ = min(keyframe_id_1, keyframe_id_2);
        keyframe_id_2_ = max(keyframe_id_1, keyframe_id_2);
    }

    bool operator==(const TemporalPoint& rhs) const {
        return this->mappoint_id_1_ == rhs.mappoint_id_1_ && this->mappoint_id_2_ == rhs.mappoint_id_2_ &&
               this->keyframe_id_1_ == rhs.keyframe_id_1_ && this->keyframe_id_2_ == rhs.keyframe_id_2_;
    }

    struct HashFunction
    {
        size_t operator()(const TemporalPoint& point) const
        {
            size_t mappoint_1_hash = std::hash<ID>()(point.mappoint_id_1_);
            size_t mappoint_2_hash = std::hash<ID>()(point.mappoint_id_2_) << 1;
            size_t keyframe_1_hash = std::hash<ID>()(point.keyframe_id_1_) << 2;
            size_t keyframe_2_hash = std::hash<ID>()(point.keyframe_id_2_) << 3;
            return mappoint_1_hash ^ mappoint_2_hash ^ keyframe_1_hash ^ keyframe_2_hash;
        }
    };

private:
    ID mappoint_id_1_, mappoint_id_2_;
    ID keyframe_id_1_, keyframe_id_2_;
};

class SpatialPoint {
public:
    SpatialPoint() = delete;

    SpatialPoint(ID mappoint_id_1, ID mappoint_id_2, ID keyframe_id){
        mappoint_id_1_ = min(mappoint_id_1, mappoint_id_2);
        mappoint_id_2_ = max(mappoint_id_1, mappoint_id_2);
        keyframe_id_ = keyframe_id;
    }

    bool operator==(const SpatialPoint& rhs) const {
        return this->mappoint_id_1_ == rhs.mappoint_id_1_ && this->mappoint_id_2_ == rhs.mappoint_id_2_ &&
               this->keyframe_id_ == rhs.keyframe_id_;
    }

    struct HashFunction
    {
        size_t operator()(const SpatialPoint& point) const
        {
            size_t mappoint_1_hash = std::hash<ID>()(point.mappoint_id_1_);
            size_t mappoint_2_hash = std::hash<ID>()(point.mappoint_id_2_) << 1;
            size_t keyframe_hash = std::hash<ID>()(point.keyframe_id_) << 2;
            return mappoint_1_hash ^ mappoint_2_hash ^ keyframe_hash;
        }
    };

private:
    ID mappoint_id_1_, mappoint_id_2_;
    ID keyframe_id_;
};

void LocalDeformableBundleAdjustment(std::shared_ptr<Map> map,
                                     const float scale) {
    // Create optimizer.
    g2o::SparseOptimizer optimizer;
    std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver =  g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver))
    );

    optimizer.setAlgorithm(solver);

    auto keyframes = map->GetKeyFrames();

    const int max_keyframes_in_optimization = 5;
    vector<shared_ptr<KeyFrame>> keyframes_in_optimization;

    // Get window of KeyFrames and set their vertices
    int biggest_keyframe_idx = 0;
    for (auto it = keyframes.rbegin(); it != keyframes.rend(); it++) {
        if (keyframes_in_optimization.size() >= max_keyframes_in_optimization) {
            break;
        }

        auto keyframe = it->second;

        g2o::VertexSE3Expmap* camera_pose_vertex = new g2o::VertexSE3Expmap();
        Sophus::SE3f camera_transform_world = keyframe->CameraTransformationWorld();
        camera_pose_vertex->setEstimate(g2o::SE3Quat(
                camera_transform_world.unit_quaternion().cast<double>(),
                camera_transform_world.translation().cast<double>()));
        camera_pose_vertex->setId(keyframe->GetId());

        optimizer.addVertex(camera_pose_vertex);

        keyframes_in_optimization.push_back(keyframe);

        if (biggest_keyframe_idx < keyframe->GetId()) {
            biggest_keyframe_idx = keyframe->GetId();
        }
    }

    if (keyframes_in_optimization.size() < 3) {
        return;
    }

    // Set landmark vertices.
    absl::btree_map<ID, absl::btree_map<ID, int>> inserted_landmarks;
    int current_optimization_idx = biggest_keyframe_idx + 1;
    int n_inserted_landmarks = 0;
    for (auto it = keyframes_in_optimization.rbegin(); it != keyframes_in_optimization.rend(); it++) {
        auto keyframe = *it;
        ID keyframe_id = keyframe->GetId();

        auto landmark_positions = keyframe->GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
        auto mappoint_ids = keyframe->GetMapPointsIdsWithStatus({TRACKED_WITH_3D});

        for (int idx = 0; idx < landmark_positions.size(); idx++) {
            ID mappoint_id = mappoint_ids[idx];
            Eigen::Vector3f landmark_position = landmark_positions[idx];

            auto landmark_vertex = new LandmarkVertex();
            landmark_vertex->setId(current_optimization_idx);
            landmark_vertex->setEstimate(landmark_position.cast<double>());

            optimizer.addVertex(landmark_vertex);

            inserted_landmarks[keyframe_id][mappoint_id] = current_optimization_idx;

            current_optimization_idx++;
            n_inserted_landmarks++;
        }
    }

    // 简洁的优化信息
    LOG(INFO) << "[局部BA] 关键帧:" << keyframes_in_optimization.size() 
              << "个, 地标:" << n_inserted_landmarks << "个";

    const int regularizers_per_point = 10;

    // Increase robust thresholds for noisy / deformable scenes
    const float th_huber_2dof_squared = 9.21; // more permissive for 2DOF reprojection residuals
    const float th_huber_2dof = sqrt(th_huber_2dof_squared);

    const float th_huber_3dof_squared = 1.0;
    const float th_huber_3dof = sqrt(th_huber_3dof_squared);

    // Reduce reprojection information (increase sigma) to make optimization less brittle
    const float sigma_reprojection = 1.0f;   // pixels (was 0.5)
    const float info_reprojection = 1.0f / (sigma_reprojection * sigma_reprojection);

    const float sigma_position = 0.1f;//  * 0.0394105;
    const float info_position = 1.0f / (sigma_position * sigma_position);

    const float sigma_spatial = 0.1 * scale;   // mm.
    const float info_spatial = 1.0f / (sigma_spatial * sigma_spatial);
    
    // 早期地图优先刚性，后期再非刚性
    bool is_early_stage = map->IsEarlyMapStage();
    float rigidity_factor = 1.0f;
    float deformable_factor = 1.0f;
    
    if (is_early_stage) {
        // 早期阶段：增强刚性约束，减弱非刚性约束
        rigidity_factor = 2.0f;  // 刚性约束权重加倍
        deformable_factor = 0.5f; // 非刚性约束权重减半
    } else {
        // 后期阶段：正常权重
        rigidity_factor = 1.0f;
        deformable_factor = 1.0f;
    }
    
    // 根据地图阶段调整信息矩阵
    const float adjusted_info_position = info_position * rigidity_factor;
    const float adjusted_info_spatial = info_spatial * deformable_factor;

    auto regularization_graph = map->GetRegularizationGraph();

    // Point vertices will be added as we add observations. This map stores the landmarks vertex index
    // in the optimization in the following manner:
    //      inserted_landmarks[keyframe_id][mappoint_id] = idx_in_optimization;
    absl::flat_hash_set<SpatialPoint, SpatialPoint::HashFunction> spring_edges;
    absl::flat_hash_set<TemporalPoint,TemporalPoint::HashFunction> dumper_edges;
    for (auto it = keyframes_in_optimization.rbegin(); it != keyframes_in_optimization.rend(); it++) {
        auto keyframe = *it;
        ID keyframe_id = keyframe->GetId();

        auto next_it = next(it);
        shared_ptr<KeyFrame> next_keyframe = nullptr;
        int next_keyframe_id = -1;
        if (next_it != keyframes_in_optimization.rend()) {
            next_keyframe = *next_it;
            next_keyframe_id = next_keyframe->GetId();
        }

        auto landmark_positions = keyframe->GetLandmarkPositionsWithStatus({TRACKED_WITH_3D});
        auto mappoint_ids = keyframe->GetMapPointsIdsWithStatus({TRACKED_WITH_3D});
        auto keypoints = keyframe->GetKeypointsWithStatus({TRACKED_WITH_3D});

        for (int idx = 0; idx < landmark_positions.size(); idx++) {
            ID mappoint_id = mappoint_ids[idx];
            Eigen::Vector3f landmark_position = landmark_positions[idx];
            cv::KeyPoint keypoint = keypoints[idx];

            int landmark_optimization_index = inserted_landmarks[keyframe_id][mappoint_id];

            // Set reprojection error.
            auto reprojection_error = new ReprojectionError();

            cv::Point2f pixel_coordinates = keypoint.pt;
            Eigen::Matrix<double, 2, 1> observation(pixel_coordinates.x, pixel_coordinates.y);

            reprojection_error->setMeasurement(observation);

            reprojection_error->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                    optimizer.vertex(keyframe_id)));
            reprojection_error->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                    optimizer.vertex(landmark_optimization_index)));

            reprojection_error->setInformation(Eigen::Matrix2d::Identity() * info_reprojection);

            g2o::RobustKernelHuber *robust_kernel = new g2o::RobustKernelHuber;
            robust_kernel->setDelta(th_huber_2dof);
            reprojection_error->setRobustKernel(robust_kernel);

            reprojection_error->calibration_ = keyframe->GetCalibration();

            optimizer.addEdge(reprojection_error);

            // Set up position regularizers.
            auto regularization_edges =
                    regularization_graph->GetEdges(mappoint_id);

            int n_regularizers = 0;
            for (const auto &[mappoint_id_other, regularization_edge]: regularization_edges) {
                // Check if there is already enough regularizers or if the connection is good.
                if (n_regularizers > regularizers_per_point ||
                    regularization_edge->status == RegularizationGraph::BAD) {
                    break;
                }

                // Check the other landmark is observed by the Keyframe.
                if (!inserted_landmarks[keyframe_id].contains(mappoint_id_other)) {
                    continue;
                }

                // Check if this regularizer has already been inserted in the optimization.
                SpatialPoint spring_connection(mappoint_id, mappoint_id_other, keyframe_id);
                if (spring_edges.contains(spring_connection)) {
                    n_regularizers++;
                    continue;
                }

                spring_edges.insert(spring_connection);

                const int other_landmark_optimization_index =
                        inserted_landmarks[keyframe_id][mappoint_id_other];

                auto position_regularizer = new PositionRegularizer();
                position_regularizer->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                        optimizer.vertex(landmark_optimization_index)));
                position_regularizer->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                        optimizer.vertex(other_landmark_optimization_index)));

                position_regularizer->setMeasurement(regularization_edge->first_distance);

                Eigen::Matrix<double, 1, 1> position_information_matrix = 
                        Eigen::Matrix<double, 1, 1>::Identity() * adjusted_info_position;
                position_regularizer->setInformation(position_information_matrix);

                position_regularizer->k_ = 1.1f;

                optimizer.addEdge(position_regularizer);

                n_regularizers++;
            }

            if (next_keyframe) {
                if (!inserted_landmarks[next_keyframe_id].contains(mappoint_id)) {
                    continue;
                }

                const int next_landmark_optimization_index =
                        inserted_landmarks[next_keyframe_id][mappoint_id];

                // Set up position regularizers
                int n_regularizers = 0;
                for (const auto &[mappoint_id_other, regularization_edge]: regularization_edges) {
                    // Check if there is already enough regularizers or if the connection is good.
                    if (n_regularizers > regularizers_per_point ||
                        regularization_edge->status == RegularizationGraph::BAD) {
                        break;
                    }

                    // Check both Mappoints are observed by the next Keyframe
                    if (!inserted_landmarks[keyframe_id].contains(mappoint_id_other) ||
                        !inserted_landmarks[next_keyframe_id].contains(mappoint_id_other)) {
                        continue;
                    }

                    // Check if this regularizer has already been inserted in the optimization.
                    TemporalPoint dumper_connection(mappoint_id, mappoint_id_other, keyframe_id, next_keyframe_id);
                    if (dumper_edges.contains(dumper_connection)) {
                        n_regularizers++;
                        continue;
                    }

                    dumper_edges.insert(dumper_connection);

                    const int other_landmark_optimization_index =
                            inserted_landmarks[keyframe_id][mappoint_id_other];
                    const int next_other_landmark_optimization_index =
                            inserted_landmarks[next_keyframe_id][mappoint_id_other];

                    SpatialRegularizer* spatial_regularizer = new SpatialRegularizer();
                    spatial_regularizer->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                            optimizer.vertex(landmark_optimization_index)));
                    spatial_regularizer->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                            optimizer.vertex(other_landmark_optimization_index)));
                    spatial_regularizer->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                            optimizer.vertex(next_landmark_optimization_index)));
                    spatial_regularizer->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                            optimizer.vertex(next_other_landmark_optimization_index)));

                    Eigen::Matrix3d spatial_information_matrix = Eigen::Matrix3d::Identity() * adjusted_info_spatial;
                    spatial_regularizer->setInformation(spatial_information_matrix);

                    g2o::RobustKernelHuber* robust_kernel = new g2o::RobustKernelHuber;
                    robust_kernel->setDelta(th_huber_3dof);
                    spatial_regularizer->setRobustKernel(robust_kernel);

                    spatial_regularizer->weight_ = regularization_edge->weight * deformable_factor;

                    optimizer.addEdge(spatial_regularizer);

                    n_regularizers++;
                }
            }
        }
    }

    // --- Add soft constraints from TRACKED (2D-only) keypoints when a prior 3D estimate exists in the temporal buffer.
    // These are added as low-information reprojection edges to provide weak guidance.
    // IMPORTANT: Deduplicate by keypoint_id to avoid creating multiple unconnected vertices for the same feature.
    {
        auto temporal_buffer = map->GetTemporalBuffer();
        if (temporal_buffer) {
            auto& raw = temporal_buffer->GetRawBuffer();
            int latest_frame_id = -1;
            if (!raw.empty()) latest_frame_id = raw.rbegin()->first;

            // Map from keypoint_id -> optimization vertex index (for deduplication)
            absl::flat_hash_map<int, int> tracked_keypoint_vertex_map;

            for (auto it_kf = keyframes_in_optimization.rbegin(); it_kf != keyframes_in_optimization.rend(); ++it_kf) {
                auto keyframe = *it_kf;
                ID keyframe_id = keyframe->GetId();

                auto tracked_keypoints = keyframe->GetKeypointsWithStatus({TRACKED});
                for (int tk = 0; tk < tracked_keypoints.size(); ++tk) {
                    const cv::KeyPoint& tkp = tracked_keypoints[tk];
                    int keypoint_id = tkp.class_id;
                    if (latest_frame_id < 0) continue;

                    int vertex_idx;
                    auto existing_it = tracked_keypoint_vertex_map.find(keypoint_id);

                    if (existing_it == tracked_keypoint_vertex_map.end()) {
                        // First time seeing this feature: create vertex
                        auto pos_status = temporal_buffer->GetLandmarkPosition(latest_frame_id, keypoint_id);
                        if (!pos_status.ok()) continue;

                        Eigen::Vector3f approx_world = *pos_status;
                        if (!std::isfinite(approx_world.x()) || !std::isfinite(approx_world.z())) continue;

                        auto soft_landmark_vertex = new LandmarkVertex();
                        soft_landmark_vertex->setId(current_optimization_idx);
                        soft_landmark_vertex->setEstimate(approx_world.cast<double>());
                        optimizer.addVertex(soft_landmark_vertex);

                        vertex_idx = current_optimization_idx;
                        tracked_keypoint_vertex_map[keypoint_id] = vertex_idx;

                        ++current_optimization_idx;
                        ++n_inserted_landmarks;
                    } else {
                        // Already created vertex for this feature: reuse it
                        vertex_idx = existing_it->second;
                    }

                    // Add reprojection edge linking this keyframe to the (shared) vertex
                    auto reproj_edge = new ReprojectionError();
                    cv::Point2f pixel_coordinates = tkp.pt;
                    Eigen::Matrix<double,2,1> observation(pixel_coordinates.x, pixel_coordinates.y);
                    reproj_edge->setMeasurement(observation);
                    reproj_edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(keyframe_id)));
                    reproj_edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(vertex_idx)));
                    Eigen::Matrix2d low_info = Eigen::Matrix2d::Identity() * (info_reprojection * 0.01);
                    reproj_edge->setInformation(low_info);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    rk->setDelta(th_huber_2dof);
                    reproj_edge->setRobustKernel(rk);
                    reproj_edge->calibration_ = keyframe->GetCalibration();
                    optimizer.addEdge(reproj_edge);
                }
            }
        }
    }

    // Perform optimization.
    optimizer.setVerbose(false);
    optimizer.initializeOptimization(0);
    optimizer.optimize(5);

    // Recover optimized variables.
    for (const auto [keyframe_id, mappoint_id_to_idx] : inserted_landmarks) {
        // Update KeyFrame pose
        auto keyframe_vertex = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(keyframe_id));
        auto keyframe = map->GetKeyFrame(keyframe_id);
        keyframe->CameraTransformationWorld() = Sophus::SE3f(
                keyframe_vertex->estimate().to_homogeneous_matrix().cast<float>());

        // Update landmark positions
        for (const auto [mappoint_id, idx_in_optimization] : mappoint_id_to_idx) {
            auto landmark_vertex = static_cast<LandmarkVertex*>(optimizer.vertex(idx_in_optimization));

            int idx_in_keyframe = keyframe->MapPointIdToIndex().at(mappoint_id);
            keyframe->LandmarkPositions()[idx_in_keyframe] = landmark_vertex->estimate().cast<float>();
        }
    }
}
