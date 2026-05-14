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

#include "essential_matrix_initialization.h"

#include "utilities/geometry_toolbox.h"

#include "absl/log/log.h"

using namespace std;

EssentialMatrixInitialization::EssentialMatrixInitialization(Options& options,
                                                             std::shared_ptr<CameraModel> calibration,
                                                             std::shared_ptr<ImageVisualizer> image_visualizer) :
        options_(options) {
    // Reserve memory.
    reference_keypoints_.reserve(options_.max_features);
    current_keypoints_.reserve(options_.max_features);

    reference_rays_.resize(options_.max_features, 3);
    current_rays_.resize(options_.max_features, 3);

    calibration_ = calibration;
    image_visualizer_ = image_visualizer;
}

void EssentialMatrixInitialization::ChangeReference(std::vector<cv::KeyPoint> &keypoints) {
    reference_keypoints_ = keypoints;
}

absl::Status EssentialMatrixInitialization::Initialize(const std::vector<cv::KeyPoint>& current_keypoints,
                                                       const std::vector<LandmarkStatus>& keypoint_statuses,
                                                       const int n_matches, Sophus::SE3f& camera_transform_world,
                                                       std::vector<absl::StatusOr<Eigen::Vector3f>>& landmarks_position) {
    if (n_matches < 8) {
        return absl::InternalError("Not enough matches");
    }
    // Set up input data.
    current_keypoints_ = current_keypoints;
    feature_tracks_statuses_ = keypoint_statuses;

    // Unproject tracked features.
    UnprojectTrackedFeatures();


    // Check if we have enough tracked points for RANSAC
    if (reference_rays_.rows() < options_.min_sample_set_size) {
        return absl::InternalError("Not enough tracked points for essential matrix estimation");
    }

    landmarks_position.resize(current_keypoints.size());
    fill(landmarks_position.begin(), landmarks_position.end(), absl::InternalError("Not triangulated"));

    // If enough features are tracked, try to find an Essential matrix with RANSAC.
    vector<bool> inliers_of_E(reference_rays_.rows(), false);
    int n_inliers_of_E = 0;  // Initialize to 0
    Eigen::Matrix3f E = FindEssentialWithRANSAC(n_matches, inliers_of_E, n_inliers_of_E);

    // Reconstruct the environment with the Essential matrix found.
    auto status =  ReconstructEnvironment(E, camera_transform_world, landmarks_position, n_inliers_of_E, inliers_of_E);

    if (image_visualizer_)
        image_visualizer_->DrawFeatures(current_keypoints_, landmarks_position);

    return status;
}

int EssentialMatrixInitialization::ComputeMaxTries(const float inlier_fraction, const float success_likelihood){
    return log(1 - success_likelihood) /
           log(1 - pow(inlier_fraction, options_.min_sample_set_size));
}

void EssentialMatrixInitialization::UnprojectTrackedFeatures() {
    ransac_idx_to_keypoint_idx_.clear();
    reference_keypoints_tracked_.clear();

    // First count tracked points to resize matrices safely (some datasets have more matches than max_features)
    int n_tracked_points = 0;
    for (int idx = 0; idx < static_cast<int>(feature_tracks_statuses_.size()); ++idx) {
        if (feature_tracks_statuses_[idx] == TRACKED) ++n_tracked_points;
    }


    if (n_tracked_points == 0) {
        // Nothing to unproject.
        reference_keypoints_tracked_.clear();
        ransac_idx_to_keypoint_idx_.clear();
        reference_rays_.resize(0, 3);
        current_rays_.resize(0, 3);
        return;
    }

    // Resize bearing ray storage to exact number of tracked points to avoid out-of-bounds block writes.
    reference_rays_.resize(n_tracked_points, 3);
    current_rays_.resize(n_tracked_points, 3);

    int write_idx = 0;
    for (int idx = 0; idx < static_cast<int>(feature_tracks_statuses_.size()); ++idx) {
        if (feature_tracks_statuses_[idx] == TRACKED) {
            reference_rays_.block(write_idx, 0, 1, 3) =
                    calibration_->Unproject(reference_keypoints_[idx].pt.x, reference_keypoints_[idx].pt.y).normalized();
            current_rays_.block(write_idx, 0, 1, 3) =
                    calibration_->Unproject(current_keypoints_[idx].pt.x, current_keypoints_[idx].pt.y).normalized();

            reference_keypoints_tracked_.push_back(reference_keypoints_[idx].pt);
            ransac_idx_to_keypoint_idx_.push_back(idx);
            ++write_idx;
        }
    }
}

Eigen::Matrix3f EssentialMatrixInitialization::FindEssentialWithRANSAC(const int n_matches,
                                                                       vector<bool>& inliers, int& n_inliers) {
    int best_score = 0;
    vector<bool> inliers_best_model;
    inliers_best_model.reserve(reference_rays_.rows());
    Eigen::Matrix<float,3,3> best_E;

    srand(4);

    // Cluster data for random selection.
    const int n_clusters = options_.min_sample_set_size;
    const int n_attemps = 3;
    auto termination_criteria = cv::TermCriteria(cv::TermCriteria::EPS, 10, 1.0);
    vector<int> labels;
    vector<cv::Point2f> centers;

    LOG(INFO) << "K-means: points=" << reference_keypoints_tracked_.size() << ", clusters=" << n_clusters;

    if (reference_keypoints_tracked_.size() < n_clusters) {
        LOG(ERROR) << "Not enough points for k-means clustering: " << reference_keypoints_tracked_.size() << " < " << n_clusters;
        return Eigen::Matrix3f::Zero();
    }

    cv::kmeans(reference_keypoints_tracked_, n_clusters, labels, termination_criteria,
               n_attemps, cv::KMEANS_PP_CENTERS, centers);

    LOG(INFO) << "K-means completed, labels.size=" << labels.size();

    // Assign indices to clusters with the given labels.
    vector<vector<int>> clusters = vector<vector<int>>(n_clusters);
    for(int idx = 0; idx < labels.size(); idx++){
        clusters[labels[idx]].push_back(idx);
    }

    // Compute number of RANSAC iterations
    const float inlier_fraction = 0.8;
    const float sucess_likelihood = 0.95;
    int max_iterations = ComputeMaxTries(inlier_fraction, sucess_likelihood);
    int current_iteration = 0;

    // Do all iterations.
    while(current_iteration < max_iterations){
        current_iteration++;

        // Get minimum sample set of data.
        Eigen::Matrix<float,8,3> reference_rays_sample, current_rays_sample;
        int valid_samples = 0;
        for(int idx = 0; idx < n_clusters && valid_samples < 8; idx++){
            if (clusters[idx].empty()) {
                LOG(WARNING) << "Cluster " << idx << " is empty";
                continue;
            }
            random_shuffle(clusters[idx].begin(), clusters[idx].end());
            int feature_idx = clusters[idx][0];

            if (feature_idx < 0 || feature_idx >= reference_rays_.rows()) {
                LOG(ERROR) << "Invalid feature_idx: " << feature_idx << ", rays.rows=" << reference_rays_.rows();
                continue;
            }

            reference_rays_sample.block<1, 3>(valid_samples, 0) = reference_rays_.block(feature_idx, 0, 1, 3);
            current_rays_sample.block<1, 3>(valid_samples, 0) = current_rays_.block(feature_idx, 0, 1, 3);
            valid_samples++;
        }

        LOG(INFO) << "RANSAC sampling: valid_samples=" << valid_samples;

        // If we don't have enough samples, we can't compute the essential matrix
        if (valid_samples < 8) {
            LOG(WARNING) << "Not enough valid clusters for essential matrix computation: " << valid_samples << "/8";
            return Eigen::Matrix3f::Zero();
        }

        // Compute model with the sample.
        Eigen::Matrix<float,3,3> E = ComputeE(reference_rays_sample, current_rays_sample);

        // Get score and inliers for the computed model.
        // inliers vector should be sized for the ray matrices, not n_matches
        vector<bool> temp_inliers(reference_rays_.rows(), false);
        int score = ComputeScoreAndInliers(reference_rays_.rows(), E, temp_inliers);
        if(score > best_score){
            best_score = score;
            inliers_best_model = temp_inliers;
            best_E = E;
        }
    }

    // Recompute E with all the inliers.
    LOG(INFO) << "Best score: " << best_score << ", inliers_best_model.size: " << inliers_best_model.size();
    if (best_score == 0) {
        LOG(WARNING) << "No inliers found, returning zero matrix";
        // Ensure inliers vector is properly sized and set to false
        inliers.assign(reference_rays_.rows(), false);
        n_inliers = 0;
        return Eigen::Matrix3f::Zero();
    }
    Eigen::MatrixXf best_model_reference_rays(best_score, 3);
    Eigen::MatrixXf best_model_current_rays(best_score, 3);

    int current_idx = 0;
    for(int idx = 0; idx < inliers_best_model.size(); idx++){
        if(inliers_best_model[idx]){
            best_model_reference_rays.block<1,3>(current_idx,0) = reference_rays_.block(idx, 0, 1, 3);
            best_model_current_rays.block<1,3>(current_idx,0) = current_rays_.block(idx, 0, 1, 3);
            current_idx++;
        }
    }

    int score = ComputeScoreAndInliers(reference_rays_.rows(), best_E, inliers);
    n_inliers = score;

    return best_E;
}

Eigen::Matrix3f EssentialMatrixInitialization::ComputeE(Eigen::Matrix<float, 8, 3> &reference_rays_sample,
                                                        Eigen::Matrix<float, 8, 3> &current_rays_sample) {
    // Compute a first estimation of the Essential matrix.
    Eigen::Matrix<float, 8, 9> A;
    for(int idx = 0; idx < 8; idx++){
        A.block<1,3>(idx, 0) = reference_rays_sample.row(idx) * current_rays_sample(idx, 0);
        A.block<1,3>(idx, 3) = reference_rays_sample.row(idx) * current_rays_sample(idx, 1);
        A.block<1,3>(idx, 6) = reference_rays_sample.row(idx) * current_rays_sample(idx, 2);
    }

    Eigen::JacobiSVD<Eigen::Matrix<float, 8, 9>> svd_solver(A, Eigen::ComputeFullV);
    svd_solver.computeV();
    Eigen::Matrix<float, 9, 1> eigen_vectors = svd_solver.matrixV().col(8);
    Eigen::Matrix3f E;
    E.row(0) = eigen_vectors.block<3, 1>(0, 0).transpose();
    E.row(1) = eigen_vectors.block<3, 1>(3, 0).transpose();
    E.row(2) = eigen_vectors.block<3, 1>(6, 0).transpose();

    // Force eigen values into the Essential matrix
    Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3>> svd_essential_matrix(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3f eigen_values;
    eigen_values << 1, 1, 0;
    Eigen::Matrix<float,3,3> Ef = svd_essential_matrix.matrixU() * eigen_values.asDiagonal() *
            svd_essential_matrix.matrixV().transpose();

    return -Ef;
}

Eigen::Matrix3f EssentialMatrixInitialization::RefineSolution(Eigen::MatrixXf &reference_rays,
                                                              Eigen::MatrixXf &current_rays) {
    int data_size = reference_rays.rows();
    Eigen::MatrixXf A(data_size, 9);
    for(int idx = 0; idx < data_size; idx++){
        A.block<1, 3>(idx, 0) = reference_rays.row(idx) * current_rays(idx,0);
        A.block<1, 3>(idx, 3) = reference_rays.row(idx) * current_rays(idx,1);
        A.block<1, 3>(idx, 6) = reference_rays.row(idx) * current_rays(idx,2);
    }

    Eigen::BDCSVD<Eigen::MatrixXf> svd_solver(A,Eigen::ComputeFullV);
    svd_solver.computeV();
    Eigen::MatrixXf eigen_vectors = svd_solver.matrixV().col(8);
    Eigen::Matrix3f E;
    E.row(0) = eigen_vectors.block<3, 1>(0, 0).transpose();
    E.row(1) = eigen_vectors.block<3, 1>(3, 0).transpose();
    E.row(2) = eigen_vectors.block<3, 1>(6, 0).transpose();

    Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3>> svd_essential_matrix(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3f eigen_values;
    eigen_values << 1, 1, 0;
    Eigen::Matrix<float, 3, 3> Ef = svd_essential_matrix.matrixU() * eigen_values.asDiagonal() *
            svd_essential_matrix.matrixV().transpose();

    return -Ef;
}


int EssentialMatrixInitialization::ComputeScoreAndInliers(const int n_matched,
                                                          Eigen::Matrix<float,3,3>& E,
                                                          std::vector<bool>& inliers) {
    // Use the actual size of the ray matrices, which may be smaller than n_matched
    // since only TRACKED points are stored in the matrices
    int actual_rays_size = reference_rays_.rows();
    int rays_to_check = std::min(n_matched, actual_rays_size);

    // Calculate epipolar errors
    Eigen::MatrixXf reference_rays_transformed = (E * reference_rays_.block(0, 0, rays_to_check, 3)
            .transpose()).transpose().rowwise().normalized();

    Eigen::MatrixXf reference_rays_norm = reference_rays_.block(0, 0, rays_to_check, 3).rowwise().normalized();
    Eigen::MatrixXf current_rays_norm = current_rays_.block(0, 0, rays_to_check, 3).rowwise().normalized();
    
    // Epipolar angle error
    auto epipolar_errors = (M_PI / 2 - (reference_rays_transformed.array() *
            current_rays_norm.array())
            .rowwise().sum().acos()).abs() < options_.epipolar_threshold;
    
    // Ray-ray angle (coarse parallax) error - require minimum parallax
        float min_init_parallax = max(options_.radians_per_pixel * 5.0f, static_cast<float>(0.5f * M_PI / 180.0f)); // 0.5 degrees or 5 pixels
    vector<bool> ray_angle_ok(rays_to_check, false);

    for (int idx = 0; idx < rays_to_check; idx++) {
        float cos_angle = reference_rays_norm.row(idx).dot(current_rays_norm.row(idx));
        float angle = acos(std::max(std::min(cos_angle, 1.0f), -1.0f));
        ray_angle_ok[idx] = (angle > min_init_parallax);
    }

    // Combine epipolar error and ray-ray angle check
    int score = 0;
    // Resize inliers vector if needed
    if (inliers.size() < rays_to_check) {
        inliers.resize(rays_to_check, false);
    }
    fill(inliers.begin(), inliers.begin() + rays_to_check, false);
    for (int idx = 0; idx < rays_to_check; idx++){
        if (epipolar_errors(idx) && ray_angle_ok[idx]) {
            inliers[idx] = true;
            score++;
        }
    }

    return score;
}

absl::Status EssentialMatrixInitialization::ReconstructEnvironment(
        Eigen::Matrix3f& E, Sophus::SE3f& camera_transform_world,
        std::vector<absl::StatusOr<Eigen::Vector3f>>& landmarks_position,
        int& n_inliers, std::vector<bool>& essential_matrix_inliers) {

    // Check if we have any inliers
    if (n_inliers == 0) {
        LOG(WARNING) << "No inliers found, cannot reconstruct environment";
        return absl::InternalError("No inliers for environment reconstruction");
    }

    // Compute rays of the inliers points.
    Eigen::MatrixXf reference_rays(n_inliers, 3);
    Eigen::MatrixXf current_rays(n_inliers, 3);
    int current_idx = 0;
    for(int idx = 0; idx < essential_matrix_inliers.size(); idx++){
        if(essential_matrix_inliers[idx]){
            reference_rays.row(current_idx) =
                    calibration_->Unproject(reference_keypoints_[idx].pt.x, reference_keypoints_[idx].pt.y).normalized();
            current_rays.row(current_idx) =
                    calibration_->Unproject(current_keypoints_[idx].pt.x, current_keypoints_[idx].pt.y).normalized();

            current_idx++;
        }
    }

    //Reconstruct camera poses
    ReconstructCameras(E, camera_transform_world, reference_rays, current_rays);

    //Reconstruct 3D points (try with the 2 possible translations)
    return ReconstructPoints(camera_transform_world, landmarks_position, essential_matrix_inliers);
}

void EssentialMatrixInitialization::SelectBestRtByCheiralityAndParallax(const Eigen::Matrix3f& R1, const Eigen::Matrix3f& R2, const Eigen::Vector3f& t,
                                                                         const Eigen::MatrixXf& rays_1, const Eigen::MatrixXf& rays_2,
                                                                         Eigen::Matrix3f& best_R, Eigen::Vector3f& best_t) {
    // Define 4 possible (R, t) combinations
    vector<Eigen::Matrix3f> rotations = {R1, R1, R2, R2};
    vector<Eigen::Vector3f> translations = {t, -t, t, -t};
    
    struct RtCandidate {
        int positive_depth_count;
        float average_parallax;
        Eigen::Matrix3f R;
        Eigen::Vector3f t;
    };
    
    vector<RtCandidate> candidates;
    
    // Sample 30-50 inliers for fast evaluation
    const int sample_size = min(50, max(30, static_cast<int>(rays_1.rows() / 5)));
    vector<int> sample_indices(rays_1.rows());
    iota(sample_indices.begin(), sample_indices.end(), 0);
    random_shuffle(sample_indices.begin(), sample_indices.end());
    sample_indices.resize(sample_size);
    
    for (size_t i = 0; i < rotations.size(); ++i) {
        const auto& R = rotations[i];
        const auto& t_candidate = translations[i];
        
        int positive_depth = 0;
        float total_parallax = 0.0f;
        
        Sophus::SE3f T_world_camera2(R, t_candidate);
        
        for (int idx : sample_indices) {
            Eigen::Vector3f ray1 = rays_1.row(idx).normalized();
            Eigen::Vector3f ray2 = rays_2.row(idx).normalized();
            
            // Triangulate midpoint
            auto landmark_status = TriangulateMidPoint(ray1, ray2, Sophus::SE3f(), T_world_camera2);
            if (!landmark_status.ok()) continue;
            
            Eigen::Vector3f landmark = *landmark_status;
            
            // Check positive depth in both cameras
            if (landmark.z() > 0.0f) {
                Eigen::Vector3f landmark_cam2 = T_world_camera2 * landmark;
                if (landmark_cam2.z() > 0.0f) {
                    positive_depth++;
                    
                    // Calculate parallax
                    Eigen::Vector3f normal1 = landmark.normalized();
                    Eigen::Vector3f normal2 = (landmark - t_candidate).normalized();
                    float parallax = RaysParallax(normal1, normal2);
                    total_parallax += parallax;
                }
            }
        }
        
        float avg_parallax = (positive_depth > 0) ? total_parallax / positive_depth : 0.0f;
        candidates.push_back({positive_depth, avg_parallax, R, t_candidate});
    }
    
    // Select the best candidate: max positive depth + max average parallax
    auto best_candidate = max_element(candidates.begin(), candidates.end(), 
        [](const RtCandidate& a, const RtCandidate& b) {
            if (a.positive_depth_count != b.positive_depth_count) {
                return a.positive_depth_count < b.positive_depth_count;
            } else {
                return a.average_parallax < b.average_parallax;
            }
        });
    
    best_R = best_candidate->R;
    best_t = best_candidate->t;
}

void EssentialMatrixInitialization::ReconstructCameras(Eigen::Matrix3f &E, Sophus::SE3f &camera_transform_world, 
                                                       Eigen::MatrixXf& rays_1, Eigen::MatrixXf& rays_2) {
    // Decompose E into 2 rotation hypotheses (R_1 and R_2) and a translation vector.
    Eigen::Matrix3f R_1, R_2, best_R;
    Eigen::Vector3f t, best_t;
    DecomposeEssentialMatrix(E, R_1, R_2, t);

    // Select the best (R, t) combination based on cheirality and parallax
    SelectBestRtByCheiralityAndParallax(R_1, R_2, t, rays_1, rays_2, best_R, best_t);

    camera_transform_world = Sophus::SE3f(best_R, best_t);
}

void EssentialMatrixInitialization::DecomposeEssentialMatrix(Eigen::Matrix3f &E, Eigen::Matrix3f &R_1,
                                                             Eigen::Matrix3f &R_2, Eigen::Vector3f &t) {
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(E,Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, 1;

    R_1 = svd.matrixU() * W.transpose() * svd.matrixV().transpose();
    if(R_1.determinant() < 0)
        R_1 = -R_1;

    R_2 = svd.matrixU() * W * svd.matrixV().transpose();
    if(R_2.determinant() < 0)
        R_2 = -R_2;

    t = svd.matrixU().col(2).normalized();
}

absl::Status EssentialMatrixInitialization::ReconstructPoints(const Sophus::SE3f &camera_transform_world,
                                                      std::vector<absl::StatusOr<Eigen::Vector3f>>&landmarks_position,
                                                      std::vector<bool>& essential_matrix_inliers) {
    vector<float> landmarks_parallax;
    vector<float> valid_depths;
    int n_triangulated = 0;
    int n_parallax = 0, n_reprojection_error_1 = 0, n_reprojection_error_2 = 0, n_triangulation_error = 0;
    int n_depth_1 = 0, n_depth_2 = 0, n_depth_ratio = 0, n_far_point = 0;
    int N = 0;

    Eigen::Vector3f world_t_camera = camera_transform_world.inverse().translation();

    // First pass: collect all valid depths for median calculation
    for (int idx = 0; idx < essential_matrix_inliers.size(); idx++){
        if( essential_matrix_inliers[idx]){
            N++;
            // Map back to original keypoint indices
            int keypoint_idx = ransac_idx_to_keypoint_idx_[idx];
            // Unproject KeyPoints to rays.
            Eigen::Vector3f reference_ray =
                    calibration_->Unproject(reference_keypoints_[keypoint_idx].pt.x, reference_keypoints_[keypoint_idx].pt.y).normalized();
            Eigen::Vector3f current_ray =
                    calibration_->Unproject(current_keypoints_[keypoint_idx].pt.x, current_keypoints_[keypoint_idx].pt.y).normalized();

            // Triangulate point.
            auto landmark_position_status =
                    TriangulateMidPoint(reference_ray, current_ray, Sophus::SE3f(), camera_transform_world);
            if (!landmark_position_status.ok()) {
                continue;
            }

            Eigen::Vector3f landmark_position = *landmark_position_status;
            Eigen::Vector3f landmark_position_camera2 = camera_transform_world * landmark_position;

            // Check positive depth in both cameras
            if(landmark_position(2) > 0.0f && landmark_position_camera2(2) > 0.0f) {
                valid_depths.push_back(landmark_position(2));
            }
        }
    }

    // Calculate median depth for far point filtering
    float median_depth = 0.0f;
    if (!valid_depths.empty()) {
        size_t median_idx = valid_depths.size() / 2;
        nth_element(valid_depths.begin(), valid_depths.begin() + median_idx, valid_depths.end());
        median_depth = valid_depths[median_idx];
    }

    // Reset valid_depths for second pass
    valid_depths.clear();

    // Second pass: apply all filters and generate final points
    for (int idx = 0; idx < essential_matrix_inliers.size(); idx++){
        if( essential_matrix_inliers[idx]){
            // Map back to original keypoint indices
            int keypoint_idx = ransac_idx_to_keypoint_idx_[idx];
            // Unproject KeyPoints to rays.
            Eigen::Vector3f reference_ray =
                    calibration_->Unproject(reference_keypoints_[keypoint_idx].pt.x, reference_keypoints_[keypoint_idx].pt.y).normalized();
            Eigen::Vector3f current_ray =
                    calibration_->Unproject(current_keypoints_[keypoint_idx].pt.x, current_keypoints_[keypoint_idx].pt.y).normalized();

            // Triangulate point.
            auto landmark_position_status =
                    TriangulateMidPoint(reference_ray, current_ray, Sophus::SE3f(), camera_transform_world);
            if (!landmark_position_status.ok()) {
                landmarks_position[keypoint_idx] = absl::InternalError("Internal triangulation error.");
                n_triangulation_error++;
                continue;
            }

            Eigen::Vector3f landmark_position = *landmark_position_status;

            // Check the parallax of the triangulated point with dynamic threshold
            Eigen::Vector3f normal_1 = landmark_position;
            Eigen::Vector3f normal_2 = landmark_position - world_t_camera;
            float parallax = RaysParallax(normal_1, normal_2);

            // Dynamic parallax threshold: max(5 * rad_per_pixel, 0.5 deg)
            float min_parallax = max(options_.radians_per_pixel * 5.0f, static_cast<float>(0.5f * M_PI / 180.0f));
            if(parallax < min_parallax){
                landmarks_position[keypoint_idx] = absl::InternalError("Low parallax error.");
                n_parallax++;
                continue;
            }

            // Check that the point has been triangulated in front of the first camera (positive depth).
            if(landmark_position(2) < 0.0f){
                landmarks_position[keypoint_idx] = absl::InternalError("Negative depth at first camera.");
                n_depth_1++;
                continue;
            }

            // Check Reprojection error.
            cv::Point2f projected_landmark_1 = calibration_->Project(landmark_position);
            if(SquaredReprojectionError(reference_keypoints_[keypoint_idx].pt, projected_landmark_1) > 5.991){
                landmarks_position[keypoint_idx] = absl::InternalError("High reprojection error at first camera.");
                n_reprojection_error_1++;
                continue;
            }

            Eigen::Vector3f landmark_position_camera2 = camera_transform_world * landmark_position;
            if(landmark_position_camera2(2) < 0.0f){
                landmarks_position[keypoint_idx] = absl::InternalError("Negative depth at second camera.");
                n_depth_2++;
                continue;
            }

            // Check depth ratio between cameras (z2/z1 ∈ [0.2, 5.0])
            float depth_ratio = landmark_position_camera2(2) / landmark_position(2);
            if(depth_ratio < 0.2f || depth_ratio > 5.0f){
                landmarks_position[keypoint_idx] = absl::InternalError("Bad depth ratio.");
                n_depth_ratio++;
                continue;
            }

            cv::Point2f projected_landmark_2 = calibration_->Project(landmark_position_camera2);
            if(SquaredReprojectionError(current_keypoints_[keypoint_idx].pt, projected_landmark_2) > 5.991){
                landmarks_position[keypoint_idx] = absl::InternalError("High reprojection error at second camera.");
                n_reprojection_error_2++;
                continue;
            }

            // Check for far points (z < median(z_all) * 5)
            if(median_depth > 0.0f && landmark_position(2) > median_depth * 5.0f){
                landmarks_position[keypoint_idx] = absl::InternalError("Too far point.");
                n_far_point++;
                continue;
            }

            landmarks_position[keypoint_idx] = landmark_position;
            valid_depths.push_back(landmark_position(2));

            n_triangulated++;
            landmarks_parallax.push_back(parallax);

        }
    }

    if(n_triangulated < 50){
        return absl::InternalError("Not enough triangulated landmarks");
    }

    if(n_parallax > N * 0.25){
        return absl::InternalError("Not enough triangulated landmarks");
    }

    LOG(INFO) << "Initialization triangulation stats: " 
              << "triangulated=" << n_triangulated << ", "
              << "parallax_fail=" << n_parallax << ", "
              << "depth_ratio_fail=" << n_depth_ratio << ", "
              << "far_point_fail=" << n_far_point << ", "
              << "reproj1_fail=" << n_reprojection_error_1 << ", "
              << "reproj2_fail=" << n_reprojection_error_2;

    return absl::OkStatus();
}
