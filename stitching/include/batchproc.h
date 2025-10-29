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

struct OrthoMosaic {
    fs::path orthoPath;
    double geoTransform[6];
    int width;
    int height;
    std::optional<std::string> mosaicProjWkt;
};

struct UnionExtent {
    double unionMinX;
    double unionMaxX;
    double unionMinY;
    double unionMaxY;
    double avgResX;
    double avgResY;
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
    std::mutex mosaicMutex_;

    // batchproc.cpp
    int runCommand(const std::string& cmd);

    fs::path createBatchDirectory(const std::vector<fs::path>& images, int batch_id);

    void savePreviousOrthophoto();
    void setProcessedStatusToImages(const fs::path& imagesDir);
    
    bool runBatch(const fs::path& batchPath);
    bool runBatchSuccessful(const fs::path& batchPath, const fs::path& orthoPath);

    std::optional<OrthoMosaic> fillOrthoMosaic(const fs::path& orthoPath);

    UnionExtent getUnionExtent(OrthoMosaic& oldm, OrthoMosaic& newm);

    std::optional<TemporaryPath> writeMosaicWKT(const OrthoMosaic& old);
    TemporaryPath initMosaicExpanded();
    TemporaryPath initMosaicRealigned();
    TemporaryPath initMosaicOptimized();

    // Mosaic processing
    void initMosaic(const fs::path& orthoPath);
    void expandMosaic(TemporaryPath& tempWktFile, TemporaryPath& tempWarpedExpanded, const OrthoMosaic &oldm, const OrthoMosaic &newm, const UnionExtent& u);
    void mergeMosaic(TemporaryPath& expanded, TemporaryPath& realigned);
    void optimizeMosaic(TemporaryPath& unoptimized, TemporaryPath& optimized);
    void merge(const fs::path& newOrthoPath);

    // tiffproc.cpp
    bool validateGeotiff(const char* path);
    bool getRasterInfo(const char* path, double geoTransform[6], std::optional<std::string>& projWkt, int& width, int& height);
    void calculateUnionExtent(double geoTransform1[6], int width1, int height1,
                              double geoTransform2[6], int width2, int height2,
                              double& unionMinX, double& unionMaxX, 
                              double& unionMinY, double& unionMaxY,
                              double& avgResX, double& avgResY);

    
};

#endif