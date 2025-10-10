#include "../include/fwatcher.h"
#include <algorithm>
#include <thread>

FolderWatcher::FolderWatcher(const Config& config, std::queue<BatchTask>& batchQueue, std::mutex& queueMutex, std::condition_variable& queueCV, std::atomic<bool>& stopSignal)
    : config_(config), 
    batchQueue_(batchQueue), 
    queueMutex_(queueMutex), 
    queueCV_(queueCV), 
    stopSignal_(stopSignal) {
    fs::create_directories(config_.incomingDir);
}

inline bool FolderWatcher::isJpg(const fs::path& p) {
    auto ext = p.extension().string();
    for (auto& c : ext) c = std::tolower(c);
    return ext == ".jpg" || ext == ".jpeg";
}

void FolderWatcher::watchFolderLoop() {
    int nextBatchId = 1;
    while (!stopSignal_) {
        std::vector<fs::path> newImages;

        for (const auto& entry : fs::directory_iterator(config_.incomingDir)) {
            if (isJpg(entry.path()) && seenFiles_.find(entry.path()) == seenFiles_.end()) {
                // potential improvement: check if file is fully written (e.g., by size stability)
                newImages.push_back(entry.path());
            }
        }

        {
            std::lock_guard<std::mutex> lock(queueMutex_);

            if (!newImages.empty()) {
                imageBuffer_.insert(imageBuffer_.end(), newImages.begin(), newImages.end());
                for (const auto& imagePath : newImages) {
                    seenFiles_.insert(imagePath);
                }

                if (!bufferStartTime_.has_value()) {
                    bufferStartTime_ = std::chrono::steady_clock::now();
                }
            }

            auto now = std::chrono::steady_clock::now();

            // Condition 1: Batch size reached
            while (imageBuffer_.size() >= config_.batchSize) {
                std::vector<fs::path> batchImages(imageBuffer_.begin(), imageBuffer_.begin() + config_.batchSize);
                batchQueue_.push({batchImages, nextBatchId++}); // Push a BatchTask
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
                (now - *bufferStartTime_ >= std::chrono::seconds(config_.batchTimeoutSec)) && // fix timer bug
                !imageBuffer_.empty()) {

                batchQueue_.push({imageBuffer_, nextBatchId++});
                imageBuffer_.clear();
                bufferStartTime_ = std::nullopt;
                queueCV_.notify_one();
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    std::cout << "[INFO] FolderWatcher loop stopped." << std::endl;
}