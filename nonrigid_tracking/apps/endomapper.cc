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

#include "datasets/endomapper.h"
#include "SLAM/system.h"

#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"

#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <memory>

using namespace std;

ABSL_FLAG(std::string, dataset_path, "", "Path to the video dataset");
ABSL_FLAG(std::string, settings_path, "", "Path to the settings file");
ABSL_FLAG(int, starting_frame, 0, "First frame of the dataset to process");
ABSL_FLAG(int, end_frame, 0, "Last frame of the dataset to process");

int main(int argc, char **argv) {
    // ========== 日志文件输出设置 ==========
    // 生成带时间戳的日志文件名
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::stringstream timestamp_ss;
    timestamp_ss << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");
    timestamp_ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
    std::string log_filename = "nr_slam_log_" + timestamp_ss.str() + ".txt";

    // 打开日志文件
    std::ofstream log_file(log_filename);
    if (!log_file.is_open()) {
        std::cerr << "Warning: Cannot open log file: " << log_filename << std::endl;
    } else {
        std::cout << "Log file created: " << log_filename << std::endl;
    }

    // 保存原始的cout和cerr流指针
    std::streambuf* original_cout_buf = std::cout.rdbuf();
    std::streambuf* original_cerr_buf = std::cerr.rdbuf();

    // TeeBuffer类：同时输出到两个流
    class TeeBuffer : public std::streambuf {
    public:
        TeeBuffer(std::streambuf* buf1, std::streambuf* buf2) : buf1_(buf1), buf2_(buf2) {}
    protected:
        int overflow(int c) override {
            if (c != EOF) {
                buf1_->sputc(static_cast<char>(c));
                buf2_->sputc(static_cast<char>(c));
            }
            return !EOF;
        }
        int sync() override {
            buf1_->pubsync();
            buf2_->pubsync();
            return 0;
        }
    private:
        std::streambuf* buf1_;
        std::streambuf* buf2_;
    };

    // 如果成功打开日志文件，则将cout和cerr重定向到日志文件（TeeBuffer）
    std::unique_ptr<TeeBuffer> tee_cout;
    std::unique_ptr<TeeBuffer> tee_cerr;
    if (log_file.is_open()) {
        tee_cout = std::make_unique<TeeBuffer>(original_cout_buf, log_file.rdbuf());
        tee_cerr = std::make_unique<TeeBuffer>(original_cerr_buf, log_file.rdbuf());
        std::cout.rdbuf(tee_cout.get());
        std::cerr.rdbuf(tee_cerr.get());
    }
    // ========== 日志文件输出设置结束 ==========

    // Parse command line argumemnts
    absl::ParseCommandLine(argc, argv);

    // Process command arguments
    string dataset_path = absl::GetFlag(FLAGS_dataset_path);
    if(dataset_path.empty()){
        LOG(ERROR) << "Must specify an input dataset path." << endl;
        return -1;
    }
    string settings_path = absl::GetFlag(FLAGS_settings_path);
    if(settings_path.empty()){
        LOG(ERROR) << "Must specify an input settings file." << endl;
        return -1;
    }

    int starting_frame = absl::GetFlag(FLAGS_starting_frame);
    int end_frame = absl::GetFlag(FLAGS_end_frame);

    Endomapper dataset(dataset_path);

    // If end_frame is 0 or greater than total frames, process all frames from starting_frame
    int total_frames = dataset.GetTotalFrames();
    if (end_frame == 0 || end_frame > total_frames) {
        end_frame = total_frames;
    }

    // Create SLAM system.
    System SLAM(settings_path);

    LOG(INFO) << "Processing frames from " << starting_frame << " to " << (end_frame - 1)
              << " (total frames: " << total_frames << ")";

    for (int idx = starting_frame; idx < end_frame; idx++) {
        LOG(INFO) << "Processing image " << idx;
        auto image = dataset.GetImage(idx);
        CHECK_OK(image);

        // Resize input image.
        // cv::Size newSize((*image).cols/2.0f, (*image).rows/2.0f);
        // cv::resize(*image, *image, newSize);

        SLAM.TrackImage(*image);
    }

    // ========== 恢复原始流并关闭日志文件 ==========
    std::cout.rdbuf(original_cout_buf);
    std::cerr.rdbuf(original_cerr_buf);
    if (log_file.is_open()) {
        log_file.close();
        std::cout << "Log file saved: " << log_filename << std::endl;
    }
    // ========== 日志文件输出设置结束 ==========

    return 0;
}