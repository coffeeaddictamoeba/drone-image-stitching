#include "../include/stitcher.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <array>
#include <gdal_version.h>

#define RESET   "\033[0m"
#define RED     "\033[31m"      // Errors
#define YELLOW  "\033[33m"      // Warnings
#define GREEN   "\033[32m"      // Success

bool startApplication = true;

// Initialize static member
MosaicStitcher* MosaicStitcher::instance_ = nullptr;

MosaicStitcher::MosaicStitcher()
    : stopSignal_(false),
      watcher_(nullptr), // will be initialized in run()
      processor_(nullptr) // will be initialized in run()
{
    instance_ = this;
}

std::string execCommand(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "Error";
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    pclose(pipe);
    // Remove trailing newline
    if (!result.empty() && result.back() == '\n') result.pop_back();
    return result;
}

bool isCommandAvailable(const std::string& cmd) {
    return (system(("which " + cmd + " > /dev/null 2>&1").c_str()) == 0);
}

bool isDockerContainer() {
    std::ifstream cgroup("/proc/1/cgroup");
    std::string line;
    while (std::getline(cgroup, line)) {
        if (line.find("docker") != std::string::npos || line.find("containerd") != std::string::npos)
            return true;
    }
    std::ifstream dockerenv("/.dockerenv");
    return dockerenv.good();
}

bool isDockerImagePresent(const std::string& image) {
    std::string cmd = "docker image inspect " + image + " > /dev/null 2>&1";
    return (system(cmd.c_str()) == 0);
}

bool isDockerDaemonRunning() {
    return (system("docker info > /dev/null 2>&1") == 0);
}

void MosaicStitcher::isMySetupOkay() {
    // Timestamp
    std::time_t now = std::time(nullptr);
    std::cout << "=== Stitching Setup Check (" << std::asctime(std::localtime(&now)) << ") ===";

    // Docker container detection
    std::cout << "[Container] " << (isDockerContainer() ? "Running inside Docker" : "Not running inside Docker") << "\n";

    std::cout << "\n-- Dependency Status --\n";

    // GDAL
    if (isCommandAvailable("gdalinfo")) {
        std::cout << "[GDAL]          Found, Version: " << execCommand("gdalinfo --version") << "\n";
    } else {
        std::cout << "[GDAL]          Missing\n";
    }

    // ExifTool
    if (isCommandAvailable("exiftool")) {
        std::cout << "[ExifTool]      Found, Version: " << execCommand("exiftool -ver") << "\n";
    } else {
        std::cout << "[ExifTool]      Missing\n";
    }

    // Orfeo Toolbox
    if (isCommandAvailable("otbrun.sh")) {
        std::cout << "\nCheck Orfeo Toolbox availability: \n";
        execCommand("otbrun.sh otbcli_Mosaic -version");
        std::cout << "\n";
    } else {
        std::cout << "[Orfeo Toolbox] Missing (wrapper not found)\n";
    }

    // Docker
    if (isCommandAvailable("docker")) {
        std::cout << "[Docker]        Found, Version: " << execCommand("docker --version") << "\n";

        if (isDockerDaemonRunning()) {
            std::cout << "[Docker Daemon] Running\n";
            std::cout << "[ODM Image]     " 
                    << (isDockerImagePresent("opendronemap/odm") ? "Present" : "Not Found") 
                    << "\n";
        } else {
            std::cout << "[Docker Daemon] Not running or unreachable\n";
        }
    } else {
        std::cout << "[Docker]        Missing (cannot check ODM image)\n";
    }

    std::cout << "===========================================\n";
}

// Default arguments are in ../include/config.h
void MosaicStitcher::parseArgs(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Environment check (print if program is ready)
        if (arg == "--is-my-setup-ok") {
            isMySetupOkay();
            startApplication = false;
            break;
        }

        // Batch configs
        else if (arg == "--batch-size" && i + 1 < argc)
            config_.batchSize = std::stoi(argv[++i]);
        else if (arg == "--wait-batch-size")
            config_.waitForBatchSize = true;
        else if (arg == "--timeout" && i + 1 < argc)
            config_.batchTimeoutSec = std::stoi(argv[++i]);

        // Data (images) configs
        else if (arg == "--data" && i + 1 < argc) { // use this explicit option if running from docker
            config_.dataDir = argv[++i];
            config_.batchDir = config_.dataDir + '/' + config_.batchDir;
            config_.stitchedDir = config_.dataDir + '/' + config_.stitchedDir;
            config_.incomingDir = config_.dataDir + '/' + config_.incomingDir;
        }
        else if (arg == "--incoming" && i + 1 < argc)
            config_.incomingDir = argv[++i];
        else if (arg == "--processing" && i + 1 < argc)
            config_.batchDir = argv[++i];
        else if (arg == "--stitched" && i + 1 < argc)
            config_.stitchedDir = argv[++i];
        else if (arg == "--stitched-filename" && i + 1 < argc)
            config_.stitchedFileName = argv[++i];

        // Memory configs
        else if (arg == "--blocksize" && i + 1 < argc)
            config_.blockSize = std::stoi(argv[++i]);
        else if (arg == "--no-bigtiff")
            config_.useBigTIFF = false;
        else if (arg == "--no-compress")
            config_.compress = false;
        else if (arg == "--save-prev")
            config_.savePreviousOrthophoto = true;
        
        // Image verification configs
        else if (arg == "--no-retry")
            config_.retry = false;
        else if (arg == "--retry-amount" && i + 1 < argc)
            config_.retries = std::stoi(argv[++i]);
        else if (arg == "--retry-delay" && i + 1 < argc)
            config_.retryTimeoutSec = std::stoi(argv[++i]);
        else if (arg == "--rgb-threshold" && i + 1 < argc)
            config_.rgbValidationThreshold = std::stod(argv[++i]);
        else if (arg == "--alpha-threshold" && i + 1 < argc)
            config_.alphaValidationThreshold = std::stod(argv[++i]);

        // Invalid option
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
    if (startApplication) {
        initGDAL();

        std::signal(SIGINT, MosaicStitcher::signalHandler);
        std::signal(SIGTERM, MosaicStitcher::signalHandler);

        watcher_ = std::make_unique<FolderWatcher>(config_, batchQueue_, queueMutex_, queueCV_, stopSignal_);
        processor_ = std::make_unique<BatchProcessor>(config_, batchQueue_, queueMutex_, queueCV_, stopSignal_);

        watcherThread_ = std::thread(&FolderWatcher::watchFolderLoop, watcher_.get());
        processorThread_ = std::thread(&BatchProcessor::processBatchesLoop, processor_.get());

        watcherThread_.join();
        processorThread_.join();

        cleanupGDAL();
    }
}