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

#include "system.h"

#include "absl/log/log.h"

using namespace std;

System::System(const string settings_file_path) {
    // Output welcome message
    LOG(INFO).NoPrefix() << "NR-SLAM Copyright (C) Copyright (C) 2022-2023 Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.";
    LOG(INFO).NoPrefix() << "This program comes with ABSOLUTELY NO WARRANTY;";
    LOG(INFO).NoPrefix() << "This is free software, and you are welcome to redistribute it";
    LOG(INFO).NoPrefix() << "under certain conditions. See LICENSE.txt.";

    settings_ = make_unique<Settings>(settings_file_path);
    LOG(INFO) << *settings_;

    // Initialize image processing stuff
    clahe_ = cv::createCLAHE(3.0, cv::Size(8, 8));
    masker_ = settings_->getMasker();

    // Create map
    Map::Options map_options;
    map_options.max_temporal_buffer_size = 40;  // 从20增加到40，存储更多帧以增加三角化机会
    map_ = make_shared<Map>(map_options);

    StereoLucasKanade::Options stereo_matcher_options;
    stereo_matcher_options.klt_window_size = settings_->GetStereoKLTWindowSize();
    stereo_matcher_options.klt_max_level = settings_->GetStereoKLTMaxLevel();
    stereo_matcher_options.klt_max_iters = settings_->GetStereoKLTMaxIterations();
    stereo_matcher_options.klt_epsilon = settings_->GetStereoKLTEpsilon();
    stereo_matcher_options.klt_min_eig_th = settings_->GetStereoKLTMinEigThreshold();
    stereo_matcher_options.klt_min_SSIM = settings_->GetStereoKLTMinSSIM();

    stereo_matcher_ = make_shared<StereoLucasKanade>(stereo_matcher_options, settings_->getCalibration(),
                                                     settings_->getBf());

    stereo_pattern_matcher_ = make_shared<StereoPatternMatching>(settings_->getCalibration(),
                                                                 settings_->getBf());

    // Initialize map visualizer and launch it in a different thread
    MapVisualizer::Options map_visualizer_options;
    map_visualizer_options.camera_size_ = settings_->GetCameraSize();
    map_visualizer_options.initial_left_view_ = settings_->GetLeftMapVisualizationView().matrix();
    map_visualizer_options.initial_right_view_ = settings_->GetRightMapVisualizationView().matrix();
    map_visualizer_options.render_save_path = settings_->GetMapVisualizerPath();

    map_visualizer_ = make_unique<MapVisualizer>(map_visualizer_options, map_);

    map_visualizer_thread_ = make_unique<thread>(&MapVisualizer::Run, map_visualizer_.get());

    //Initialize image visualizer.
    ImageVisualizer::Options image_visualizer_options;
    image_visualizer_options.wait_for_user_button = !settings_->GetAutoplay();
    image_visualizer_options.image_save_path = settings_->GetImageVisualizerPath();
    image_visualizer_ = make_shared<ImageVisualizer>(image_visualizer_options);

    // Initialize Tracking.
    Tracking::Options tracking_options;
    tracking_options.klt_window_size = settings_->GetKLTWindowSize();
    tracking_options.klt_max_level = settings_->GetKLTMaxLevel();
    tracking_options.klt_max_iters = settings_->GetKLTMaxIterations();
    tracking_options.klt_epsilon = settings_->GetKLTEpsilon();
    tracking_options.klt_min_eig_th = settings_->GetKLTMinEigThreshold();
    tracking_options.klt_min_SSIM = settings_->GetKLTMinSSIM();
    tracking_options.radians_per_pixel = settings_->getRadPerPixel();
    
    // 默认的尺度/正则化参数（可被配置文件覆盖）
    tracking_options.mono_init_target_median_depth = 3.0f;
    tracking_options.mono_init_reg_sigma_scale = 3.0f;
    tracking_options.stereo_init_min_depth = 0.0f;
    tracking_options.stereo_init_max_depth = 1e6f;
    tracking_options.stereo_pattern_focal = 3886.37f;
    tracking_options.stereo_reg_radius = 10.5f;
    tracking_options.stereo_map_scale = 1.0f;
    
    // Marigold数据导出选项（可以通过环境变量或配置文件控制）
    tracking_options.enable_marigold_export = true;  // 默认启用
    tracking_options.marigold_output_dir = "./output/marigold_data/";
    
    // 联合优化融合导出选项（创新点1）
    // 组织分割选项（创新点2）
    // 从settings读取（如果存在）
    cv::FileStorage fSettings(settings_file_path, cv::FileStorage::READ);
    if (fSettings.isOpened()) {
        cv::FileNode output_node = fSettings["OutputPerFrame"];
        if (!output_node.empty()) {
            tracking_options.enable_json_ply_export = (int)output_node != 0;
        }
        cv::FileNode json_path_node = fSettings["JSONPath"];
        if (!json_path_node.empty()) {
            tracking_options.json_path_template = (string)json_path_node;
        }
        cv::FileNode ply_path_node = fSettings["PLYPath"];
        if (!ply_path_node.empty()) {
            tracking_options.ply_path_template = (string)ply_path_node;
        }

        // 读取可选的深度/尺度调节参数（如果未配置则保持默认值）
        cv::FileNode mono_target_depth_node = fSettings["MonoInit.TargetMedianDepth"];
        if (!mono_target_depth_node.empty()) {
            mono_target_depth_node >> tracking_options.mono_init_target_median_depth;
        }

        cv::FileNode mono_reg_scale_node = fSettings["MonoInit.RegSigmaScale"];
        if (!mono_reg_scale_node.empty()) {
            mono_reg_scale_node >> tracking_options.mono_init_reg_sigma_scale;
        }

        cv::FileNode stereo_min_depth_node = fSettings["StereoInit.MinDepth"];
        if (!stereo_min_depth_node.empty()) {
            stereo_min_depth_node >> tracking_options.stereo_init_min_depth;
        }

        cv::FileNode stereo_max_depth_node = fSettings["StereoInit.MaxDepth"];
        if (!stereo_max_depth_node.empty()) {
            stereo_max_depth_node >> tracking_options.stereo_init_max_depth;
        }

        cv::FileNode stereo_pattern_focal_node = fSettings["StereoInit.PatternFocal"];
        if (!stereo_pattern_focal_node.empty()) {
            stereo_pattern_focal_node >> tracking_options.stereo_pattern_focal;
        }

        cv::FileNode stereo_reg_radius_node = fSettings["StereoInit.RegRadius"];
        if (!stereo_reg_radius_node.empty()) {
            stereo_reg_radius_node >> tracking_options.stereo_reg_radius;
        }

        cv::FileNode stereo_map_scale_node = fSettings["StereoInit.MapScale"];
        if (!stereo_map_scale_node.empty()) {
            stereo_map_scale_node >> tracking_options.stereo_map_scale;
        }

        // 特征提取方法配置
        cv::FileNode feature_type_node = fSettings["FeatureType"];
        if (!feature_type_node.empty()) {
            std::string ft;
            feature_type_node >> ft;
            tracking_options.feature_type = ft;
        }
        cv::FileNode orb_n_features_node = fSettings["ORB.NFeatures"];
        if (!orb_n_features_node.empty()) {
            orb_n_features_node >> tracking_options.orb_n_features;
        }
        cv::FileNode orb_scale_factor_node = fSettings["ORB.ScaleFactor"];
        if (!orb_scale_factor_node.empty()) {
            orb_scale_factor_node >> tracking_options.orb_scale_factor;
        }
        cv::FileNode orb_n_levels_node = fSettings["ORB.NLevels"];
        if (!orb_n_levels_node.empty()) {
            orb_n_levels_node >> tracking_options.orb_n_levels;
        }
        
        // 组织分割选项（创新点2）
        cv::FileNode hetero_node = fSettings["EnableHeterogeneousViscoelastic"];
        if (!hetero_node.empty()) {
            tracking_options.enable_heterogeneous_viscoelastic = (int)hetero_node != 0;
        }
        cv::FileNode mask_json_node = fSettings["MaskJSONPath"];
        if (!mask_json_node.empty()) {
            tracking_options.mask_json_path = (string)mask_json_node;
        }

        cv::FileNode tissue_params_node = fSettings["TissueParams"];
        if (!tissue_params_node.empty()) {
            for (auto it = tissue_params_node.begin(); it != tissue_params_node.end(); ++it) {
                std::string class_name = (*it).name();
                Tracking::Options::TissueParam param;
                cv::FileNode node = *it;
                if (!node["k"].empty()) {
                    node["k"] >> param.k;
                }
                if (!node["b"].empty()) {
                    node["b"] >> param.b;
                }
                tracking_options.tissue_params[class_name] = param;
            }
        }

        fSettings.release();
    }

    if (!tracking_options.tissue_params.contains("soft")) {
        tracking_options.tissue_params["soft"] = {0.8f, 1.2f};
    }
    if (!tracking_options.tissue_params.contains("hard")) {
        tracking_options.tissue_params["hard"] = {1.5f, 0.5f};
    }

    // Time profiler.
    time_profiler_ = make_unique<TimeProfiler>();

    // Initialize Mapping.
    Mapping::Options mapping_options;
    mapping_options.rad_per_pixel = settings_->getRadPerPixel();
    // Deformable/small-baseline friendly defaults - 优化三角化参数以增加点云数量
    mapping_options.deformable_min_track_len = 1;  // 降低到1，让新特征点更快参与三角化
    mapping_options.min_parallax_mult = 2.0f;     // 降低最小视差要求
    mapping_options.max_parallax_mult = 50.0f;    // 提高最大视差限制
    mapping_options.reprojection_chi2_threshold = 15.0f;  // 放宽重投影误差阈值
    mapping_options.reprojection_chi2_threshold = 25.0f; // more permissive for endoscopy scenes
    mapper_ = make_unique<Mapping>(map_, settings_->getCalibration(), mapping_options, time_profiler_.get());

    tracker_ = make_unique<Tracking>(tracking_options, map_, settings_->getCalibration(),
                                     stereo_matcher_, image_visualizer_, time_profiler_.get(), mapper_.get());

    // Initialize frame evaluator.
    FrameEvaluator::Options frame_evaluator_options;
    frame_evaluator_options.results_file_path = settings_->GetEvaluationPath();
    frame_evaluator_options.precomputed_depth_ = true;
    frame_evaluator_ = make_unique<FrameEvaluator>(frame_evaluator_options, stereo_pattern_matcher_,
                                                   map_visualizer_.get());
}

System::~System() {
    // Send signal to the visualizer to finish
    map_visualizer_->SetFinish();

    // Wait until is done
    map_visualizer_thread_->join();
}

void System::TrackImage(const cv::Mat &im) {
    // Preprocess image.
    cv::Mat im_gray;
    cv::Mat processed_image = ImageProcessing(im, im_gray);

    // Insert image in the image visualizer.
    image_visualizer_->SetCurrentImage(im, processed_image);

    // Generate image mask.
    auto masks = masker_->GetAllMasks(im_gray);

    // Perform tracking.
    tracker_->TrackImage(im_gray, masks, cv::Mat(), processed_image);

    // Perform mapping.
    mapper_->DoMapping();

    // Draw images.
    image_visualizer_->UpdateWindows();
}

void System::TrackImageWithStereo(const cv::Mat &im_left, const cv::Mat &im_right,int frame_idx) {
    // Preprocess images.
    cv::Mat im_gray_left, im_gray_right;
    cv::Mat processed_image_left = ImageProcessing(im_left, im_gray_left);
    cv::Mat processed_image_right = ImageProcessing(im_right, im_gray_right);

    // Insert image in the image visualizer.
    image_visualizer_->SetCurrentImage(im_left, processed_image_left);

    // Generate image mask.
    auto masks = masker_->GetAllMasks(im_gray_left);

    // Perform tracking.
    tracker_->TrackImage(im_gray_left, masks, im_gray_right, processed_image_left,frame_idx);

    // Perform mapping.
    mapper_->DoMapping();

    // Evaluate reconstruction.
    if (settings_->GetEvaluationEnabled() && tracker_->GetTrackingStatus() == Tracking::TRACKING) {
        frame_evaluator_->EvaluateFrameReconstruction(*(map_->GetMutableLastFrame()), im_gray_left, im_gray_right);
        frame_evaluator_->SaveResultsToFile();
    }

    // Draw images.
    image_visualizer_->UpdateWindows();
}

void System::TrackImageWithDepth(const cv::Mat &im_left, const cv::Mat &im_depth) {
    // Preprocess images.
    cv::Mat im_gray_left;
    cv::Mat processed_image_left = ImageProcessing(im_left, im_gray_left);

    // Insert image in the image visualizer.
    image_visualizer_->SetCurrentImage(im_left, processed_image_left);

    // Generate image mask.
    auto masks = masker_->GetAllMasks(im_gray_left);

    // Perform tracking. Note: im_depth is used for evaluation, not for tracking
    tracker_->TrackImage(im_gray_left, masks, cv::Mat(), processed_image_left);

    // Perform mapping.
    mapper_->DoMapping();

    // Evaluate reconstruction.
    if (settings_->GetEvaluationEnabled() && tracker_->GetTrackingStatus() == Tracking::TRACKING) {
        frame_evaluator_->EvaluateFrameReconstruction(*(map_->GetMutableLastFrame()), im_gray_left, im_depth);
        frame_evaluator_->SaveResultsToFile();
    }

    // Draw images.
    image_visualizer_->UpdateWindows();
}

cv::Mat System::ImageProcessing(const cv::Mat &im, cv::Mat& im_gray) {
    cv::Mat processed_image;

    // Convert to grayscale.
    cv::cvtColor(im, processed_image, cv::COLOR_RGB2GRAY);

    im_gray = processed_image.clone();

    // Apply Clahe to the image.
    clahe_->apply(processed_image, processed_image);

    return processed_image;
}
