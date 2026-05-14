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

#include "position_regularizer.h"

PositionRegularizer::PositionRegularizer() {}

bool PositionRegularizer::read(std::istream& is) {
    return true;
}

bool PositionRegularizer::write(std::ostream& os) const {
    return true;
}

void PositionRegularizer::computeError() {
    const LandmarkVertex* vertex_landmark_1 = static_cast<const LandmarkVertex*>(_vertices[0]);
    const LandmarkVertex* vertex_landmark_2 = static_cast<const LandmarkVertex*>(_vertices[1]);

    Eigen::Vector3d current_position_1 = vertex_landmark_1->estimate();
    Eigen::Vector3d current_position_2 = vertex_landmark_2->estimate();

    double current_distance = (current_position_1 - current_position_2).norm();

    _error(0) = k_ * (current_distance - _measurement) / _measurement;
}

void PositionRegularizer::linearizeOplus() {
    const LandmarkVertex* vertex_landmark_1 = static_cast<const LandmarkVertex*>(_vertices[0]);
    const LandmarkVertex* vertex_landmark_2 = static_cast<const LandmarkVertex*>(_vertices[1]);

    Eigen::Vector3d current_position_1 = vertex_landmark_1->estimate();
    Eigen::Vector3d current_position_2 = vertex_landmark_2->estimate();

    const Eigen::Vector3d difference_vector = current_position_1 - current_position_2;
    const double current_distance = difference_vector.norm();

    // d(error)/d(x1) = k/d0 * d(||x1-x2||)/d(x1) = k/d0 * (x1-x2)/||x1-x2||
    if (current_distance < 1e-10) {
        _jacobianOplusXi.setZero();
        _jacobianOplusXj.setZero();
        return;
    }

    const Eigen::Vector3d normalized_diff = difference_vector / current_distance;

    _jacobianOplusXi = (k_ / _measurement) * normalized_diff.transpose();
    _jacobianOplusXj = -(k_ / _measurement) * normalized_diff.transpose();
}

