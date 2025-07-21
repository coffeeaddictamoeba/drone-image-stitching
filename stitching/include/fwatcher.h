#ifndef FOLDER_WATCHER_H
#define FOLDER_WATCHER_H

#include "config.h"
#include "batchproc.h"

#include <filesystem>
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unordered_set>
#include <optional>
#include <chrono>

namespace fs = std::filesystem;

class FolderWatcher {
public:
    FolderWatcher(const Config& config_ref,
                  std::queue<BatchTask>& batch_queue,
                  std::mutex& queue_mutex,
                  std::condition_variable& queue_cv,
                  std::atomic<bool>& stop_signal);

    void watchFolderLoop();

private:
    const Config& config_;
    std::queue<BatchTask>& batchQueue_;
    std::mutex& queueMutex_;
    std::condition_variable& queueCV_;
    std::atomic<bool>& stopSignal_;

    std::unordered_set<fs::path> seenFiles_;
    std::vector<fs::path> imageBuffer_;
    std::optional<std::chrono::steady_clock::time_point> bufferStartTime_;

    inline bool isJpg(const fs::path& p);
};

#endif