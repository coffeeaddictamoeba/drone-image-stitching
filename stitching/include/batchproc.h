#ifndef BATCH_PROC_H
#define BATCH_PROC_H

#include "config.h"
#include "raii.h"
#include <filesystem>
#include <optional>
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;

enum class OdmRunResult {
    Success,
    CommandFailed,
    OrthophotoNotFound,
    OrthophotoZeroBytes,
    ValidationFailed
};

struct BatchTask {
    std::vector<fs::path> images;
    int batch_id;
};

class BatchProcessor {
public:
    BatchProcessor(const Config& config_ref,
                   std::queue<BatchTask>& batch_queue,
                   std::mutex& queue_mutex,
                   std::condition_variable& queue_cv,
                   std::atomic<bool>& stop_signal);

    void processBatchesLoop();

private:
    const Config& config_;
    std::queue<BatchTask>& batchQueue_;
    std::mutex& queueMutex_;
    std::condition_variable& queueCV_;
    std::atomic<bool>& stopSignal_;

    static constexpr const char* STITCHED_FILE = "stitched/final_orthophoto.tif";
    static constexpr const char* BATCHES_DIR = "batches";

    fs::path createBatchDirectory(const std::vector<fs::path>& images, int batch_id);
    int runCommand(const std::string& cmd);
    bool runOdmBatchInternal(const fs::path& batch_path);
    bool validateGeotiff(const fs::path& path);
    bool getRasterInfo(const fs::path& path, double gt[6], std::optional<std::string>& proj_wkt, int& width, int& height);
    void calculateUnionExtent(double gt1[6], int w1, int h1, double gt2[6], int w2, int h2,
                              double& union_minX, double& union_maxY, double& union_maxX, double& union_minY,
                              double& avg_resX, double& avg_resY);
    void mergeWithOTB(const fs::path& ortho_path);
    bool runOdmBatchSuccessful(const fs::path& batch_path, const fs::path& ortho_path);

    std::mutex mosaicMutex_;
};

#endif