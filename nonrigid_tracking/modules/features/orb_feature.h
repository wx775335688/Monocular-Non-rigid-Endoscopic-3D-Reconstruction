/*
 * ORB-based feature extractor for NR-SLAM
 */

#ifndef NRSLAM_ORB_FEATURE_H
#define NRSLAM_ORB_FEATURE_H

#include "feature.h"
#include <opencv2/features2d.hpp>

class ORBFeature : public Feature {
public:
    struct Options {
        int n_features = 1000;
        float scale_factor = 1.2f;
        int n_levels = 8;
        int edge_threshold = 31;
        int first_level = 0;
        int WTA_K = 2;
        cv::ORB::ScoreType score_type = cv::ORB::HARRIS_SCORE;
        int patch_size = 31;
        int fast_threshold = 20;
    };

    explicit ORBFeature(const Options& options) {
        orb_ = cv::ORB::create(
            options.n_features,
            options.scale_factor,
            options.n_levels,
            options.edge_threshold,
            options.first_level,
            options.WTA_K,
            options.score_type,
            options.patch_size,
            options.fast_threshold
        );
    }

    void Extract(const cv::Mat& im, std::vector<cv::KeyPoint>& keypoints) override {
        CV_Assert(!im.empty());

        cv::Mat gray;
        if (im.channels() == 3) {
            cv::cvtColor(im, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = im;
        }

        cv::Mat descriptors;
        orb_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);

        // Use the keypoint index as class_id to be compatible with existing code
        for (size_t i = 0; i < keypoints.size(); ++i) {
            keypoints[i].class_id = static_cast<int>(i);
        }
    }

private:
    cv::Ptr<cv::ORB> orb_;
};

#endif // NRSLAM_ORB_FEATURE_H


