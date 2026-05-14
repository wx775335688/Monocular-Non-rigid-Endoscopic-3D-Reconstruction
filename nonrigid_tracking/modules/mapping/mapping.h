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

#ifndef NRSLAM_MAPPING_H
#define NRSLAM_MAPPING_H

#include "map/map.h"
#include "utilities/time_profiler.h"

class Mapping {
public:
    struct Options {
        float rad_per_pixel;
        float depth_consistency_threshold = 5.0f; // 深度一致性标准差倍数（从3.0放宽到5.0）
        int regularization_graph_delay_frames = 2; // 正则化图延迟接入帧数
        // Deformable triangulation options
        int deformable_min_track_len = 1; // 最小轨迹长度用于可形变三角化（降低到1以增加三角化机会）
        float min_parallax_mult = 2.0f;   // min parallax = min_parallax_mult * rad_per_pixel（降低以增加通过率）
        float max_parallax_mult = 50.0f;  // max parallax = max_parallax_mult * rad_per_pixel（增加以容纳更大视差）
        float reprojection_chi2_threshold = 15.0f; // 重投影卡方阈值（提高以增加通过率）
    };

    Mapping() = delete;

    Mapping(std::shared_ptr<Map> map, std::shared_ptr<CameraModel> calibration,
            const Options options, TimeProfiler* time_profiler);

    void DoMapping();

private:
    void KeyFrameMapping();

    void UpdateTrackingFrameFromKeyFrame(std::shared_ptr<KeyFrame> keyframe);

    void FrameMapping();

    void LandmarkTriangulation();

    absl::StatusOr<Eigen::Vector3f> DeformableLandmarkTriangulation(const int candidate_id);

    std::shared_ptr<Map> map_;

    std::shared_ptr<CameraModel> calibration_;

    TimeProfiler* time_profiler_;

    Options options_;
};


#endif //NRSLAM_MAPPING_H
