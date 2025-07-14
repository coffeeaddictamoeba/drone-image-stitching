#include <csignal>
#include <cstddef>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <thread>
#include <mutex>
#include <chrono>
#include <unordered_set>
#include <vector>
#include <queue>
#include <condition_variable>
#include <future>
#include <sstream>
#include <iomanip>
#include <atomic>
#include <unistd.h>
#include <gdal/gdal_priv.h>
#include <gdal/ogr_spatialref.h>

namespace fs = std::filesystem;

struct Config {
    std::string incomingDir = "incoming";
    std::size_t blockSize = 256;
    std::size_t batchSize = 5;
    int batchTimeoutSec = 5;
    bool useBigTIFF = true;
    bool compress = true;
};

Config config;
std::mutex mosaicMutex;
std::atomic<bool> stopSignal = false;
std::mutex queueMutex;
std::condition_variable queueCV;
std::vector<fs::path> imageBuffer;
auto lastBatchTime = std::chrono::steady_clock::now();
std::queue<std::vector<fs::path>> batchQueue;

constexpr const char* STITCHED_FILE = "stitched/final_orthophoto.tif";
constexpr const char* VRT = "stitched/final.vrt";
constexpr const char* BATCHES = "batches";


inline bool is_jpg(const fs::path& p) {
    auto ext = p.extension().string();
    for (auto& c : ext) c = std::tolower(c);
    return ext == ".jpg";
}

void init_dirs() {
    fs::create_directories(config.incomingDir);
    fs::create_directories(BATCHES);
    fs::create_directories("stitched");
}

fs::path create_batch(const std::vector<fs::path>& images, int batch_id) {
    std::stringstream ss;
    ss << "batch_" << std::setw(3) << std::setfill('0') << batch_id;
    fs::path batch_path = fs::path(BATCHES) / ss.str() / "images";
    fs::create_directories(batch_path);
    for (const auto& img : images) {
        fs::rename(img, batch_path / img.filename());
    }
    return batch_path.parent_path();
}

int run_command(const std::string& cmd) {
    std::cout << "[CMD] Running: " << cmd << std::endl;
    return std::system(cmd.c_str());
}

bool run_odm_batch(const fs::path& batch_path) {
    // This is needed to create files from the user, not the root
    std::string uid = std::to_string(getuid());
    std::string gid = std::to_string(getgid());

    std::string abs_path = fs::absolute(batch_path).string();
    std::string cmd = "docker run --rm "
                  "--user " + uid + ":" + gid + " "
                  "-v \"" + abs_path + ":/datasets/project\" "
                  "-w /datasets/project "
                  "opendronemap/odm "
                  "--project-path /datasets project "
                  "--fast-orthophoto --skip-3dmodel";
    return run_command(cmd) == 0;
}

void merge_with_vrt(const fs::path& ortho_path) {
    std::lock_guard<std::mutex> lock(mosaicMutex);

    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");

    fs::path unique_path = fs::path("stitched") / ("ortho_" + timestamp.str() + ".tif");

    std::cout << "[DEBUG] Copying orthophoto to: " << unique_path << std::endl;

    try {
        fs::copy_file(ortho_path, unique_path, fs::copy_options::overwrite_existing);
    } catch (const fs::filesystem_error& e) {
        std::cerr << "[ERROR] Failed to copy orthophoto: " << e.what() << std::endl;
        return;
    }

    std::cout << "[DEBUG] Stitching all orthophotos using gdalwarp..." << std::endl;

    std::stringstream ss;
    ss << "gdalwarp -overwrite -multi "
       << "-r cubic "
       << "-of GTiff "
       << "-co TILED=YES ";

    if (config.compress) ss << "-co COMPRESS=DEFLATE ";
    if (config.useBigTIFF) ss << "-co BIGTIFF=YES ";

    ss << "-co BLOCKXSIZE=" << config.blockSize << " "
       << "-co BLOCKYSIZE=" << config.blockSize << " "
       << "stitched/*.tif "
       << STITCHED_FILE;

    int result = run_command(ss.str());
    if (result != 0) {
        std::cerr << "[ERROR] gdalwarp failed to generate stitched orthophoto." << std::endl;
    } else {
        std::cout << "[INFO] Successfully updated stitched orthophoto at: " << STITCHED_FILE << std::endl;
    }
}

void process_batches() {
    int batch_id = 1;
    while (!stopSignal) {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCV.wait(lock, [] { return !batchQueue.empty() || stopSignal.load(); });

        if (stopSignal && batchQueue.empty()) break;

        auto images = batchQueue.front();
        batchQueue.pop();
        lock.unlock();

        fs::path batch_path = create_batch(images, batch_id++);
        if (run_odm_batch(batch_path)) {
            fs::path ortho = batch_path / "odm_orthophoto" / "odm_orthophoto.tif";
            if (fs::exists(ortho)) {
                merge_with_vrt(ortho);
            }
        } else {
            std::cerr << "[ERROR] ODM batch failed: " << batch_path << std::endl;
        }
    }
}

void watch_folder() {
    std::unordered_set<fs::path> seen;

    while (!stopSignal) {
        std::vector<fs::path> new_images;

        for (const auto& f : fs::directory_iterator(config.incomingDir)) {
            if (is_jpg(f.path()) && seen.find(f.path()) == seen.end()) {
                new_images.push_back(f.path());
                seen.insert(f.path());
            }
        }

        {
            std::lock_guard<std::mutex> lock(queueMutex);
            imageBuffer.insert(imageBuffer.end(), new_images.begin(), new_images.end());

            auto now = std::chrono::steady_clock::now();

            while (imageBuffer.size() >= config.batchSize) {
                std::vector<fs::path> batch(imageBuffer.begin(), imageBuffer.begin() + config.batchSize);
                batchQueue.push(batch);
                imageBuffer.erase(imageBuffer.begin(), imageBuffer.begin() + config.batchSize);
                queueCV.notify_one();
                lastBatchTime = now;
            }

            if (!imageBuffer.empty() &&
                now - lastBatchTime > std::chrono::seconds(config.batchTimeoutSec)) {
                batchQueue.push(imageBuffer);
                imageBuffer.clear();
                queueCV.notify_one();
                lastBatchTime = now;
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    {
        std::lock_guard<std::mutex> lock(queueMutex);
        if (!imageBuffer.empty()) {
            batchQueue.push(imageBuffer);
            imageBuffer.clear();
            queueCV.notify_one();
        }
    }
}

void parse_args(int argc, char* argv[], Config& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--batch-size" && i + 1 < argc)
            config.batchSize = std::stoi(argv[++i]);
        else if (arg == "--timeout" && i + 1 < argc)
            config.batchTimeoutSec = std::stoi(argv[++i]);
        else if (arg == "--incoming" && i + 1 < argc)
            config.incomingDir = argv[++i];
        else if (arg == "--no-bigtiff")
            config.useBigTIFF = false;
        else if (arg == "--no-compress")
            config.compress = false;
        else if (arg == "--blocksize" && i + 1 < argc)
            config.blockSize = std::stoi(argv[++i]);
        else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            std::exit(1);
        }
    }
}

void signal_handler(int) {
    stopSignal = true;
    queueCV.notify_all();
}

int main(int argc, char* argv[]) {
    parse_args(argc, argv, config);

    init_dirs();
    std::signal(SIGINT, signal_handler);

    std::thread watcher(watch_folder);
    std::thread processor(process_batches);

    watcher.join();
    processor.join();

    return 0;
}