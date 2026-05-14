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

#include "temporal_buffer.h"

#include "absl/log/log.h"

using namespace std;

TemporalBuffer::TemporalBuffer(TemporalBuffer::Options &options) : options_(options) {}

void TemporalBuffer::InsertSnapshotFromFrame(Frame &frame, const std::vector<float>& ssim_values) {
    Snapshot snapshot;

    // 修复：添加 JUST_TRIANGULATED 状态，确保三角化后的点能进入temporal_buffer参与后续跟踪
    auto keypoints = frame.GetKeypointsWithStatus({TRACKED_WITH_3D, TRACKED, JUST_TRIANGULATED});
    auto landmark_positions_ = frame.GetLandmarkPositionsWithStatus({TRACKED_WITH_3D, TRACKED, JUST_TRIANGULATED});
    auto statuses = frame.GetLandmarkStatusesWithStatus({TRACKED_WITH_3D, TRACKED, JUST_TRIANGULATED});
    auto indexes = frame.GetIndexWithStatus({TRACKED_WITH_3D, TRACKED, JUST_TRIANGULATED});

    // Ensure all arrays have the same size to prevent out-of-bounds access
    size_t min_size = std::min({keypoints.size(), landmark_positions_.size(),
                               statuses.size(), indexes.size()});

    for (size_t idx = 0; idx < min_size; idx++) {
        int keypoint_id = keypoints[idx].class_id;

        snapshot.keypoint_tracks[keypoint_id] = keypoints[idx];
        snapshot.keypoint_tracks_status[keypoint_id] = statuses[idx];
        snapshot.mapppoint_tracks_[keypoint_id] = landmark_positions_[idx];
        snapshot.keypoint_id_to_frame_idx[keypoint_id] = indexes[idx];
        // store SSIM if available (matches keypoints order)
        float ssim = (idx < ssim_values.size()) ? ssim_values[idx] : 0.0f;
        snapshot.keypoint_ssim[keypoint_id] = ssim;
    }

    snapshot.camera_transform_world = frame.CameraTransformationWorld();
    snapshot.deformation_magnitude = frame.GetDeformationMagnitud();
    snapshot.frame_id = frame.GetId();

    buffer_[frame.GetId()] = snapshot;

    // 淘汰逻辑：保持buffer大小不超过max_buffer_size
    while (buffer_.size() > options_.max_buffer_size) {
        // Pop oldest snapshot
        int oldest_frame_id = buffer_.begin()->first;
        buffer_.erase(oldest_frame_id);
    }
}

absl::btree_map<ID, TemporalBuffer::Snapshot>& TemporalBuffer::GetRawBuffer() {
    return buffer_;
}

std::vector<int> TemporalBuffer::GetTriangulationCandidatesIds() {
    if (buffer_.empty()) {
        return {};
    }

    TemporalBuffer::Snapshot last_snapshot = buffer_.rbegin()->second;

    // Return candidate ids sorted by descending track length (prefer longer tracks)
    vector<pair<int,int>> cand_and_length; // (keypoint_id, track_length)

    for (const auto& [keypoint_id, status] : last_snapshot.keypoint_tracks_status) {
        if (status == TRACKED) {
            int length = TrackLength(keypoint_id);
            cand_and_length.emplace_back(keypoint_id, length);
        }
    }

    sort(cand_and_length.begin(), cand_and_length.end(),
         [](const pair<int,int>& a, const pair<int,int>& b){
             return a.second > b.second; // longer first
         });

    vector<int> candidate_ids;
    candidate_ids.reserve(cand_and_length.size());
    for (const auto &p : cand_and_length) candidate_ids.push_back(p.first);
    return candidate_ids;
}

int TemporalBuffer::TrackLength(const int keypoint_id) {
    int track_lenght = 0;
    for (const auto& [frame_id, snapshot] : buffer_) {
        if (snapshot.keypoint_tracks.contains(keypoint_id)) {
            track_lenght++;
        }
    }

    return track_lenght;
}

std::vector<Sophus::SE3f> TemporalBuffer::GetLatestCameraPoses() {
    vector<Sophus::SE3f> latest_poses;

    for (const auto& [frame_id, snapshot] : buffer_) {
        latest_poses.push_back(snapshot.camera_transform_world);
    }

    return latest_poses;
}

std::vector<int> TemporalBuffer::GetClosestMapPointsToFeature(const int keypoint_id,
                                                              const int num_neighbors,
                                                              const int min_image_distance,
                                                              const int max_image_distance) {
    if (buffer_.empty()) {
        return {};
    }

    TemporalBuffer::Snapshot& last_snapshot = buffer_.rbegin()->second;

    auto keypoint_it = last_snapshot.keypoint_tracks.find(keypoint_id);
    if (keypoint_it == last_snapshot.keypoint_tracks.end()) {
        return {};
    }
    cv::KeyPoint keypoint = keypoint_it->second;

    vector<pair<float, int>> distances;
    for (const auto& [neighbor_keypoint_id, neighbor_keypoint] : last_snapshot.keypoint_tracks) {
        if (neighbor_keypoint_id == keypoint_id) {
            continue;
        }

        auto status_it = last_snapshot.keypoint_tracks_status.find(neighbor_keypoint_id);
        if (status_it == last_snapshot.keypoint_tracks_status.end() ||
            status_it->second != TRACKED_WITH_3D) {
            continue;
        }

        // Compute distance between KeyPoints
        float distance = cv::norm(keypoint.pt - neighbor_keypoint.pt);

        if (distance > max_image_distance || distance < min_image_distance) {
            continue;
        }

        distances.push_back(make_pair(distance, neighbor_keypoint_id));
    }

    // Sort neighbors by distance
    sort(distances.begin(), distances.end());

    // Recover the K closest points
    vector<int> closest_mappoint_ids_;
    for (const auto& [distance, keypoint_id] : distances) {
        if (closest_mappoint_ids_.size() >= num_neighbors) {
            break;
        }

        closest_mappoint_ids_.push_back(keypoint_id);
    }

    return closest_mappoint_ids_;
}

vector<pair<float, int>>
TemporalBuffer::GetClosestMapPointsToLocation(const Eigen::Vector3f location, const int keypoint_id,
                                              const int num_neighbors) {
    if (buffer_.empty()) {
        return {};
    }

    TemporalBuffer::Snapshot& last_snapshot = buffer_.rbegin()->second;

    vector<pair<float, int>> distances;

    for (const auto [neighbor_keypoint_id, neighbor_landmark] : last_snapshot.mapppoint_tracks_) {
        if (neighbor_keypoint_id == keypoint_id) {
            continue;
        }

        if (last_snapshot.keypoint_tracks_status[neighbor_keypoint_id] != TRACKED_WITH_3D) {
            continue;
        }

        // Compute distance between KeyPoints
        float distance = (neighbor_landmark - location).norm();

        distances.push_back(make_pair(distance, neighbor_keypoint_id));
    }

    // Sort neighbors by distance
    sort(distances.begin(), distances.end());

    // Avoid out-of-bounds access
    size_t k = std::min<size_t>(num_neighbors, distances.size());
    return vector<pair<float, int>>(distances.begin(), distances.begin() + k);
}

std::vector<std::pair<ID, cv::KeyPoint>> TemporalBuffer::GetFeatureTrack(const int keypoint_id) {
    vector<pair<ID, cv::KeyPoint>> keypoint_track;

    for (const auto& [frame_id, snapshot] : buffer_) {
        if (snapshot.keypoint_tracks.contains(keypoint_id)) {
            keypoint_track.push_back(make_pair(frame_id, snapshot.keypoint_tracks.at(keypoint_id)));
        }
    }

    return keypoint_track;
}

absl::StatusOr<Sophus::SE3f> TemporalBuffer::GetCameraTransformWorld(const int frame_id) {
    if (!buffer_.contains(frame_id)) {
        return absl::InternalError("Frame Id not found in the buffer");
    }

    return buffer_[frame_id].camera_transform_world;
}

absl::StatusOr<Eigen::Vector3f> TemporalBuffer::GetLandmarkPosition(const int frame_id,
                                                                    const int keypoint_id) {
    if (!buffer_.contains(frame_id)) {
        return absl::InternalError("Frame Id not found in the buffer");
    }

    if (!buffer_[frame_id].mapppoint_tracks_.contains(keypoint_id)) {
        return absl::InternalError("KeyPoint Id not found in the snapshot");
    }

    return buffer_[frame_id].mapppoint_tracks_[keypoint_id];
}

absl::StatusOr<int> TemporalBuffer::GetLandmarkIndexInFrame(const int frame_id, const int keypoint_id) {
    if (!buffer_.contains(frame_id)) {
        return absl::InternalError("Frame Id not found in the buffer");
    }

    if (!buffer_[frame_id].mapppoint_tracks_.contains(keypoint_id)) {
        return absl::InternalError("KeyPoint Id not found in the snapshot");
    }

    return buffer_[frame_id].keypoint_id_to_frame_idx[keypoint_id];
}

bool TemporalBuffer::CheckRigidity(const int first_frame_id, const int last_frame_id, const float rigidity_th) {
    bool to_return = true;
    for (int frame_id = first_frame_id; frame_id <= last_frame_id; frame_id++) {
        auto it = buffer_.find(frame_id);
        if (it == buffer_.end()) {
            // Frame not found, consider it as non-rigid
            to_return = false;
            break;
        }

        if (it->second.deformation_magnitude > rigidity_th) {
            to_return = false;
            break;
        }
    }

    return to_return;
}
