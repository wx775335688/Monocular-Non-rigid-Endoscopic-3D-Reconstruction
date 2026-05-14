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

#ifndef NRSLAM_MAPPOINT_H
#define NRSLAM_MAPPOINT_H

#include "matching/lucas_kanade_tracker.h"

#include <eigen3/Eigen/Core>

typedef long unsigned int ID;

class MapPoint {
public:
    MapPoint() = delete;

    // Constructor with a given 3D position.
    MapPoint(const Eigen::Vector3f& landmark_position, const int keypoint_id);

    // Gets the position of the MapPoint in the world reference.
    Eigen::Vector3f GetLastWorldPosition();

    void SetLastWorldPosition(Eigen::Vector3f& landmark_position);

    void SetPhotometricInformation(LucasKanadeTracker::PhotometricInformation& photometric_information);

    LucasKanadeTracker::PhotometricInformation GetPhotometricInformation();

    int GetKeyPointId();

    bool& IsActive();

    std::vector<Eigen::Vector3f> GetLandmarkFlow(const int n);

    // Gets the unique if of the MapPoint.
    long unsigned int GetId();
    
    // 生命周期相关方法
    int GetLifetime();
    void IncrementLifetime();
    int GetMaxNeighbors();
    float GetInitialWeightFactor();
    
    // 更新生命周期，调整最大邻居数和权重因子
    void UpdateLifetime();

    // 不良记录相关方法
    void IncrementBadCount();
    void ResetBadCount();
    void DecrementBadCount();  // 减少不良记录
    int GetBadCount();

    // 保护期相关方法
    void SetProtectionFrames(int frames);
    void DecrementProtectionFrames();
    bool IsProtected();

private:
    // 3D position of the point in the world reference.
    // TODO: is this necessary? We should rethink the map
    Eigen::Vector3f last_landmark_position_;

    // 3D position history.
    std::vector<Eigen::Vector3f> landmark_position_history_;

    // Is this a current active landmark?
    bool is_active_;

    // KeyPoint id. Used to identify feature tracks related with this MapPoint.
    int keypoint_id_;

    // Photometric information for KLT.
    LucasKanadeTracker::PhotometricInformation photometric_information_;

    //Unique id.
    long unsigned int id_;
    static long unsigned int nextId_;
    
    // 点的生命周期计数，用于调整正则化权重和邻居数量
    int lifetime_;
    
    // 该点的邻居数量限制
    int max_neighbors_;
    
    // 该点的初期权重因子
    float initial_weight_factor_;

    // 连续重投影误差大的次数，用于判断是否应该移除该点
    int bad_count_;

    // 新三角化点的保护期，避免立即被优化淘汰
    int protection_frames_;
};


#endif //NRSLAM_MAPPOINT_H
