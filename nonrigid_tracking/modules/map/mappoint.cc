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

#include "mappoint.h"

using namespace std;

long unsigned int MapPoint::nextId_ = 0;

MapPoint::MapPoint(const Eigen::Vector3f &landmark_position, const int keypoint_id) {
    last_landmark_position_ = landmark_position;
    landmark_position_history_.push_back(landmark_position);
    keypoint_id_ = keypoint_id;
    is_active_ = false;

    id_ = nextId_++;
    
    // 初始化生命周期相关字段
    lifetime_ = 0;
    max_neighbors_ = 5;  // 限制新点的邻居数量为5
    initial_weight_factor_ = 0.5f;  // 初期权重因子为0.5

    // 初始化不良记录
    bad_count_ = 0;

    // 初始化保护期
    protection_frames_ = 0;
}

Eigen::Vector3f MapPoint::GetLastWorldPosition() {
    return last_landmark_position_;
}

void MapPoint::SetLastWorldPosition(Eigen::Vector3f &landmark_position) {
    last_landmark_position_ = landmark_position;

    landmark_position_history_.push_back(landmark_position);

    is_active_ = true;
}

void MapPoint::SetPhotometricInformation(LucasKanadeTracker::PhotometricInformation& photometric_information) {
    photometric_information_ = photometric_information;
}

LucasKanadeTracker::PhotometricInformation MapPoint::GetPhotometricInformation() {
    return photometric_information_;
}

int MapPoint::GetKeyPointId() {
    return keypoint_id_;
}

bool &MapPoint::IsActive() {
    return is_active_;
}

std::vector<Eigen::Vector3f> MapPoint::GetLandmarkFlow(const int n) {
    vector<Eigen::Vector3f>::iterator start_iterator;
    if (landmark_position_history_.size() < n) {
        start_iterator = landmark_position_history_.begin();
    } else {
        start_iterator = landmark_position_history_.begin() + (landmark_position_history_.size() - n);
    }

    auto end_iterator = landmark_position_history_.begin() + (landmark_position_history_.size());
    return vector<Eigen::Vector3f>(start_iterator, end_iterator);
}

long unsigned int MapPoint::GetId() {
    return id_;
}

int MapPoint::GetLifetime() {
    return lifetime_;
}

void MapPoint::IncrementLifetime() {
    lifetime_++;
    DecrementProtectionFrames();  // 减少保护期
    UpdateLifetime();
}

int MapPoint::GetMaxNeighbors() {
    return max_neighbors_;
}

float MapPoint::GetInitialWeightFactor() {
    return initial_weight_factor_;
}

void MapPoint::UpdateLifetime() {
    // 根据生命周期调整最大邻居数和权重因子
    if (lifetime_ < 5) {
        // 非常早期的点：限制邻居数量，降低权重
        max_neighbors_ = 3;
        initial_weight_factor_ = 0.3f;
    } else if (lifetime_ < 15) {
        // 早期的点：适度增加邻居数量，提高权重
        max_neighbors_ = 5;
        initial_weight_factor_ = 0.5f;
    } else if (lifetime_ < 30) {
        // 中期的点：进一步增加邻居数量，提高权重
        max_neighbors_ = 8;
        initial_weight_factor_ = 0.7f;
    } else {
        // 成熟的点：正常邻居数量，正常权重
        max_neighbors_ = 10;
        initial_weight_factor_ = 1.0f;
    }
}

void MapPoint::IncrementBadCount() {
    bad_count_++;
}

void MapPoint::ResetBadCount() {
    bad_count_ = 0;
}

void MapPoint::DecrementBadCount() {
    if (bad_count_ > 0) {
        bad_count_--;
    }
}

int MapPoint::GetBadCount() {
    return bad_count_;
}

void MapPoint::SetProtectionFrames(int frames) {
    protection_frames_ = frames;
}

void MapPoint::DecrementProtectionFrames() {
    if (protection_frames_ > 0) {
        protection_frames_--;
    }
}

bool MapPoint::IsProtected() {
    return protection_frames_ > 0;
}
