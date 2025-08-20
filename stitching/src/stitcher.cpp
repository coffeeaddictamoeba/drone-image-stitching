#include "../include/stitcher.h"
#include <iostream>
#include <gdal/gdal_version.h>

#define RESET   "\033[0m"
#define RED     "\033[31m"      // Errors
#define YELLOW  "\033[33m"      // Warnings
#define GREEN   "\033[32m"      // Success

// Initialize static member
MosaicStitcher* MosaicStitcher::instance_ = nullptr;

MosaicStitcher::MosaicStitcher()
    : stopSignal_(false),
      watcher_(nullptr), // will be initialized in run()
      processor_(nullptr) // will be initialized in run()
{
    instance_ = this;
}

void MosaicStitcher::parseArgs(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--batch-size" && i + 1 < argc)
            config_.batchSize = std::stoi(argv[++i]);
        else if (arg == "--timeout" && i + 1 < argc)
            config_.batchTimeoutSec = std::stoi(argv[++i]);
        else if (arg == "--incoming" && i + 1 < argc)
            config_.incomingDir = argv[++i];
        else if (arg == "--no-bigtiff")
            config_.useBigTIFF = false;
        else if (arg == "--no-compress")
            config_.compress = false;
        else if (arg == "--no-retry")
            config_.retry = false;
        else if (arg == "--save-prev")
            config_.savePreviousOrthophoto = true;
        else if (arg == "--wait-batch-size")
            config_.waitForBatchSize = true;
        else if (arg == "--blocksize" && i + 1 < argc)
            config_.blockSize = std::stoi(argv[++i]);
        else if (arg == "--retry-amount" && i + 1 < argc)
            config_.retries = std::stoi(argv[++i]);
        else if (arg == "--retry-delay" && i + 1 < argc)
            config_.retryTimeoutSec = std::stoi(argv[++i]);
        else if (arg == "--rgb-threshold" && i + 1 < argc)
            config_.rgbValidationThreshold = std::stod(argv[++i]);
        else if (arg == "--alpha-threshold" && i + 1 < argc)
            config_.alphaValidationThreshold = std::stod(argv[++i]);
        else {
            std::cerr << RED << "Unknown or incomplete argument: " << arg << RESET << std::endl;
            std::exit(1);
        }
    }
}

void MosaicStitcher::initGDAL() {
    GDALAllRegister();
    std::cout << "[INFO] GDAL Version: " << GDALVersionInfo("--version") << std::endl;
}

void MosaicStitcher::cleanupGDAL() {
    GDALDestroyDriverManager();
    std::cout << "[INFO] GDAL cleanup complete." << std::endl;
}

void MosaicStitcher::signalHandler(int signum) {
    if (instance_) {
        std::cout << YELLOW << "\n[INFO] SIGINT (" << signum << ") received. Initiating graceful shutdown..." << RESET << std::endl;
        instance_->stopSignal_ = true;
        instance_->queueCV_.notify_all();
    }
}

void MosaicStitcher::run(int argc, char* argv[]) {
    parseArgs(argc, argv);
    initGDAL();

    std::signal(SIGINT, MosaicStitcher::signalHandler);

    watcher_ = std::make_unique<FolderWatcher>(config_, batchQueue_, queueMutex_, queueCV_, stopSignal_);
    processor_ = std::make_unique<BatchProcessor>(config_, batchQueue_, queueMutex_, queueCV_, stopSignal_);

    watcherThread_ = std::thread(&FolderWatcher::watchFolderLoop, watcher_.get());
    processorThread_ = std::thread(&BatchProcessor::processBatchesLoop, processor_.get());

    watcherThread_.join();
    processorThread_.join();

    cleanupGDAL();
}