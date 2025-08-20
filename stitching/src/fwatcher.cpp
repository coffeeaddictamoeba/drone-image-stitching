#include "../include/fwatcher.h"
#include <algorithm>
#include <thread>

FolderWatcher::FolderWatcher(const Config& config_ref, std::queue<BatchTask>& batch_queue, std::mutex& queue_mutex, std::condition_variable& queue_cv, std::atomic<bool>& stop_signal)
    : config_(config_ref), batchQueue_(batch_queue), queueMutex_(queue_mutex), queueCV_(queue_cv), stopSignal_(stop_signal) {
    fs::create_directories(config_.incomingDir);
}

inline bool FolderWatcher::isJpg(const fs::path& p) {
    auto ext = p.extension().string();
    for (auto& c : ext) c = std::tolower(c);
    return ext == ".jpg" || ext == ".jpeg";
}

void FolderWatcher::watchFolderLoop() {
    int next_batch_id = 1;
    while (!stopSignal_) {
        std::vector<fs::path> new_images;

        for (const auto& entry : fs::directory_iterator(config_.incomingDir)) {
            if (isJpg(entry.path()) && seenFiles_.find(entry.path()) == seenFiles_.end()) {
                // potential improvement: check if file is fully written (e.g., by size stability)
                new_images.push_back(entry.path());
            }
        }

        {
            std::lock_guard<std::mutex> lock(queueMutex_);

            if (!new_images.empty()) {
                imageBuffer_.insert(imageBuffer_.end(), new_images.begin(), new_images.end());
                for (const auto& img_path : new_images) {
                    seenFiles_.insert(img_path);
                }

                if (!bufferStartTime_.has_value()) {
                    bufferStartTime_ = std::chrono::steady_clock::now();
                }
            }

            auto now = std::chrono::steady_clock::now();

            // Condition 1: Batch size reached
            while (imageBuffer_.size() >= config_.batchSize) {
                std::vector<fs::path> batch_images(imageBuffer_.begin(), imageBuffer_.begin() + config_.batchSize);
                batchQueue_.push({batch_images, next_batch_id++}); // Push a BatchTask
                imageBuffer_.erase(imageBuffer_.begin(), imageBuffer_.begin() + config_.batchSize);
                queueCV_.notify_one();

                // Reset buffer start time if buffer is empty or new images were just added (to avoid immediate timeout)
                if (imageBuffer_.empty()) {
                    bufferStartTime_ = std::nullopt;
                } else {
                    bufferStartTime_ = now;
                }
            }

            // Condition 2: Timeout reached for accumulated images (if not waiting for full batch size)
            if (!config_.waitForBatchSize && bufferStartTime_.has_value() &&
                (now - *bufferStartTime_ >= std::chrono::seconds(config_.batchTimeoutSec)) &&
                !imageBuffer_.empty()) {

                batchQueue_.push({imageBuffer_, next_batch_id++});
                imageBuffer_.clear();
                bufferStartTime_ = std::nullopt;
                queueCV_.notify_one();
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    std::cout << "[INFO] FolderWatcher loop stopped." << std::endl;
}