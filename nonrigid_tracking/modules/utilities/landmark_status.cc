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
#include "utilities/landmark_status.h"

bool IsUsable(LandmarkStatus status) {
    // 修复：添加 BAD_FEATURE 和 OUT_IMAGE_BOUNDARIES 到可用状态
    // 这些状态通常是临时的（如SSIM低、图像边界外），应该有机会在下一帧恢复
    return status == TRACKED_WITH_3D || status == TRACKED || 
           status == JUST_TRIANGULATED || status == BAD_FEATURE || 
           status == OUT_IMAGE_BOUNDARIES;
}
