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
    BatchProcessor(const Config& config,
                   std::queue<BatchTask>& batchQueue,
                   std::mutex& queueMutex,
                   std::condition_variable& queueCV,
                   std::atomic<bool>& stopSignal);

    void processBatchesLoop();

private:
    const Config& config_;
    std::queue<BatchTask>& batchQueue_;
    std::mutex& queueMutex_;
    std::condition_variable& queueCV_;
    std::atomic<bool>& stopSignal_;

    // batchproc.cpp
    fs::path createBatchDirectory(const std::vector<fs::path>& images, int batch_id);
    int runCommand(const std::string& cmd);
    bool runOdmBatchInternal(const fs::path& batchPath);
    void mergeWithOTB(const fs::path& orthoPath);
    bool runOdmBatchSuccessful(const fs::path& batchPath, const fs::path& orthoPath);
    void savePreviousOrthophoto(std::string &timestamp);
    void setProcessedStatusToImages(const fs::path& imagesDir);

    // tiffproc.cpp
    bool validateGeotiff(const fs::path& path);
    bool getRasterInfo(const fs::path& path, double geoTransform[6], std::optional<std::string>& projWkt, int& width, int& height);
    void calculateUnionExtent(double geoTransform1[6], int width1, int height1,
                              double geoTransform2[6], int width2, int height2,
                              double& unionMinX, double& unionMaxX, double& unionMinY, double& unionMaxY,
                              double& avgResX, double& avgResY);

    std::mutex mosaicMutex_;
};

#endif