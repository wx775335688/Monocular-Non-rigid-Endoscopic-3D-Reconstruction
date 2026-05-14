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

#include "settings.h"

#include "calibration/kannala_brandt_8.h"
#include "calibration/pin_hole.h"
#include "utilities/types_conversions.h"

#include "absl/log/log.h"

using namespace std;

template<>
Sophus::SE3f Settings::readParameter<Sophus::SE3f>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
    cv::FileNode node = fSettings[name];
    if(node.empty()){
        if(required){
            LOG(ERROR) << name << " required parameter does not exist, aborting...";
            exit(-1);
        }
        else{
            LOG(WARNING) << name << " optional parameter does not exist, aborting...";
            found = false;
            return Sophus::SE3f();
        }
    }
    else{
        found = true;
        cv::Mat cvT = node.mat();

        //Convert to Sophus
        Sophus::SE3f sT = cvToSophus(cvT);

        return sT;
    }
}

template<>
bool Settings::readParameter<bool>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
    cv::FileNode node = fSettings[name];
    if(node.empty()){
        if(required){
            LOG(ERROR) << name << " required parameter does not exist, aborting...";
            exit(-1);
        }
        else{
            LOG(WARNING) << name << " optional parameter does not exist, aborting...";
            found = false;
            return false;
        }
    }
    else{
        found = true;
        int value = (int) node;

        //Convert to bool
        bool return_value = (bool) value;

        return return_value;
    }
}

Settings::Settings() {
    evaluation_enabled_ = true;  // 默认启用评估
}

Settings::Settings(const std::string& configFile) {
    //Open settings file
    cv::FileStorage fSettings(configFile, cv::FileStorage::READ);
    if(!fSettings.isOpened()){
        LOG(ERROR) << "[ERROR]: could not open configuration file at: " << configFile;
        exit(-1);
    }

    //Read camera model
    string cameraModel = (string) fSettings["Camera.model"];
    vector<float> vCalibration;
    if(cameraModel == "PinHole"){
        //Read camera calibration
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        vCalibration = {fx,fy,cx,cy};

        calibration_ = shared_ptr<CameraModel>(new PinHole(vCalibration));
    }
    else if(cameraModel == "KannalaBrandt8"){
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        float k0 = fSettings["Camera.k0"];
        float k1 = fSettings["Camera.k1"];
        float k2 = fSettings["Camera.k2"];
        float k3 = fSettings["Camera.k3"];

        vCalibration = {fx,fy,cx,cy,k0,k1,k2,k3};

        calibration_ = shared_ptr<CameraModel>(new KannalaBrandt8(vCalibration));
    }
    else{
        LOG(ERROR) << "Error: " << cameraModel << " not known";
        exit(-1);
    }

    bf_ = fSettings["Stereo.bf"];

    // Camera drawing size.
    camera_size_ = (float) fSettings["Visualization.cameraSize"];

    //Read filter for mask generation
    string filterFile = (string) fSettings["Masking.filterFile"];
    masker_ = std::make_shared<Masker>();
    masker_->loadFromTxt(filterFile);
    LOG(INFO) << masker_->printFilters();

    radPerPixel_ = (float) fSettings["Camera.radiansPerPixel"];

    //Read Map Visualization initial view points
    bool found;
    left_map_view_ = readParameter<Sophus::SE3f>(fSettings,"MapVisualizer.left_view",found,true);
    if(!found){
        LOG(ERROR) << "Parameter MapVisualizer.left_view not found";
        exit(-1);
    }

    right_map_view_ = readParameter<Sophus::SE3f>(fSettings,"MapVisualizer.right_view",found,true);
    if(!found){
        LOG(ERROR) << "Parameter MapVisualizer.right_view not found";
        exit(-1);
    }

    autoplay_ = readParameter<bool>(fSettings, "System.autoplay", found, true);
    if(!found){
        LOG(ERROR) << "Parameter System.autoplay not found";
        exit(-1);
    }

    map_visualizer_save_path_ = readParameter<string>(fSettings, "MapVisualizer.save_path", found, true);
    if(!found){
        LOG(ERROR) << "Parameter MapVisualizer.save_path not found";
        exit(-1);
    }

    image_visualizer_save_path_ = readParameter<string>(fSettings, "ImageVisualizer.save_path", found, true);
    if(!found){
        LOG(ERROR) << "Parameter ImageVisualizer.save_path not found";
        exit(-1);
    }

    evaluation_save_path_ = readParameter<string>(fSettings, "Evaluation.save_path", found, true);
    if(!found){
        LOG(ERROR) << "Parameter Evaluation.save_path not found";
        exit(-1);
    }

    evaluation_enabled_ = readParameter<bool>(fSettings, "Evaluation.enabled", found, false);
    if(!found){
        evaluation_enabled_ = true;  // 默认启用评估
        LOG(INFO) << "Parameter Evaluation.enabled not found, using default: true";
    }

    // Read tracking parameters with defaults
    klt_window_size_ = (int) fSettings["Tracking.klt_window_size"];
    if (klt_window_size_ <= 0) klt_window_size_ = 21;

    klt_max_level_ = (int) fSettings["Tracking.klt_max_level"];
    if (klt_max_level_ <= 0) klt_max_level_ = 4;

    klt_max_iters_ = (int) fSettings["Tracking.klt_max_iters"];
    if (klt_max_iters_ <= 0) klt_max_iters_ = 10;

    klt_epsilon_ = (float) fSettings["Tracking.klt_epsilon"];
    if (klt_epsilon_ <= 0) klt_epsilon_ = 0.0001f;

    klt_min_eig_th_ = (float) fSettings["Tracking.klt_min_eig_th"];
    if (klt_min_eig_th_ <= 0) klt_min_eig_th_ = 1e-6f;  // Updated default

    klt_min_ssim_ = (float) fSettings["Tracking.klt_min_ssim"];
    if (klt_min_ssim_ <= 0) klt_min_ssim_ = 0.5f;

    // Read stereo matcher parameters with defaults
    stereo_klt_window_size_ = (int) fSettings["StereoMatcher.klt_window_size"];
    if (stereo_klt_window_size_ <= 0) stereo_klt_window_size_ = 21;

    stereo_klt_max_level_ = (int) fSettings["StereoMatcher.klt_max_level"];
    if (stereo_klt_max_level_ <= 0) stereo_klt_max_level_ = 4;

    stereo_klt_max_iters_ = (int) fSettings["StereoMatcher.klt_max_iters"];
    if (stereo_klt_max_iters_ <= 0) stereo_klt_max_iters_ = 10;

    stereo_klt_epsilon_ = (float) fSettings["StereoMatcher.klt_epsilon"];
    if (stereo_klt_epsilon_ <= 0) stereo_klt_epsilon_ = 0.0001f;

    stereo_klt_min_eig_th_ = (float) fSettings["StereoMatcher.klt_min_eig_th"];
    if (stereo_klt_min_eig_th_ <= 0) stereo_klt_min_eig_th_ = 0.0001f;

    stereo_klt_min_ssim_ = (float) fSettings["StereoMatcher.klt_min_ssim"];
    if (stereo_klt_min_ssim_ <= 0) stereo_klt_min_ssim_ = 0.5f;
}

ostream &operator<<(std::ostream& output, const Settings& settings){
    output << "SLAM settings: " << endl;

    output << "\t-Camera parameters: [ ";
    output << settings.calibration_->GetParameter(0) << " , " << settings.calibration_->GetParameter(1) << " , ";
    output << settings.calibration_->GetParameter(2) << " , " << settings.calibration_->GetParameter(3) << " ]" << endl;

    output << "\t-Visualization settings:" << endl;
    output << "\t\t-[MapVisualizer] camera size: " << settings.camera_size_ << endl;

    return output;
}

std::shared_ptr<CameraModel> Settings::getCalibration() {
    return calibration_;
}

float Settings::getBf() {
    return bf_;
}

std::shared_ptr<Masker> Settings::getMasker() {
    return masker_;
}

float Settings::getRadPerPixel() {
    return radPerPixel_;
}

Sophus::SE3f Settings::GetLeftMapVisualizationView() {
    return left_map_view_;
}

Sophus::SE3f Settings::GetRightMapVisualizationView() {
    return right_map_view_;
}

float Settings::GetCameraSize() {
    return camera_size_;
}

bool Settings::GetAutoplay() {
    return autoplay_;
}

std::string Settings::GetMapVisualizerPath() {
    return map_visualizer_save_path_;
}

std::string Settings::GetImageVisualizerPath() {
    return image_visualizer_save_path_;
}

std::string Settings::GetEvaluationPath() {
    return evaluation_save_path_;
}

bool Settings::GetEvaluationEnabled() {
    return evaluation_enabled_;
}

// Tracking parameters
int Settings::GetKLTWindowSize() {
    return klt_window_size_;
}

int Settings::GetKLTMaxLevel() {
    return klt_max_level_;
}

int Settings::GetKLTMaxIterations() {
    return klt_max_iters_;
}

float Settings::GetKLTEpsilon() {
    return klt_epsilon_;
}

float Settings::GetKLTMinEigThreshold() {
    return klt_min_eig_th_;
}

float Settings::GetKLTMinSSIM() {
    return klt_min_ssim_;
}

// Stereo matcher parameters
int Settings::GetStereoKLTWindowSize() {
    return stereo_klt_window_size_;
}

int Settings::GetStereoKLTMaxLevel() {
    return stereo_klt_max_level_;
}

int Settings::GetStereoKLTMaxIterations() {
    return stereo_klt_max_iters_;
}

float Settings::GetStereoKLTEpsilon() {
    return stereo_klt_epsilon_;
}

float Settings::GetStereoKLTMinEigThreshold() {
    return stereo_klt_min_eig_th_;
}

float Settings::GetStereoKLTMinSSIM() {
    return stereo_klt_min_ssim_;
}
