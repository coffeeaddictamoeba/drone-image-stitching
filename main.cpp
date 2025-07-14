#include <csignal>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <thread>
#include <mutex>
#include <chrono>
#include <set>
#include <vector>
#include <queue>
#include <condition_variable>
#include <future>
#include <sstream>
#include <iomanip>
#include <atomic>
#include <map>
#include <gdal/gdal_priv.h>
#include <gdal/ogr_spatialref.h>

namespace fs = std::filesystem;

constexpr const char* INCOMING = "incoming";
constexpr const char* BATCHES = "batches";
constexpr const char* STITCHED = "stitched/final_orthophoto.tif";
constexpr const char* VRT = "stitched/final.vrt";
constexpr int BATCH_SIZE = 5;
constexpr std::chrono::seconds BATCH_TIMEOUT(5);

std::mutex mosaicMutex;
std::atomic<bool> stopSignal = false;
std::mutex queueMutex;
std::condition_variable queueCV;

std::vector<fs::path> imageBuffer;
auto lastBatchTime = std::chrono::steady_clock::now();

std::queue<std::vector<fs::path>> batchQueue;

void init_dirs() {
    fs::create_directories(INCOMING);
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
    std::string abs_path = fs::absolute(batch_path).string();
    std::string cmd = "docker run --rm -v \"" + abs_path + ":/datasets/project\" "
                  "opendronemap/odm "
                  "--project-path /datasets project --fast-orthophoto --skip-3dmodel";
    return run_command(cmd) == 0;
}

void merge_with_vrt(const fs::path& ortho_path) {
    std::lock_guard<std::mutex> lock(mosaicMutex);

    if (!fs::exists(VRT)) {
        run_command("gdalbuildvrt " + std::string(VRT) + " " + ortho_path.string());
    } else {
        run_command("gdalbuildvrt -update " + std::string(VRT) + " " + ortho_path.string());
    }

    std::string translateCmd =
        "gdal_translate -of GTiff "
        "-co TILED=YES "
        "-co COMPRESS=DEFLATE "
        //"-co BIGTIFF=YES "
        "-co BLOCKXSIZE=256 -co BLOCKYSIZE=256 "
        + std::string(VRT) + " " + std::string(STITCHED);

    int result = run_command(translateCmd);
    if (result == 0) {
        std::cout << "[STITCH] Final orthophoto updated with tiling and compression.\n";
    } else {
        std::cerr << "[STITCH] Failed to create optimized GeoTIFF.\n";
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
    std::set<fs::path> seen;

    while (!stopSignal) {
        std::vector<fs::path> new_images;
        for (const auto& f : fs::directory_iterator(INCOMING)) {
            if ((f.path().extension() == ".jpg" || f.path().extension() == ".JPG") &&
                seen.find(f.path()) == seen.end()) {
                new_images.push_back(f.path());
                seen.insert(f.path());
            }
        }

        {
            std::lock_guard<std::mutex> lock(queueMutex);
            imageBuffer.insert(imageBuffer.end(), new_images.begin(), new_images.end());

            auto now = std::chrono::steady_clock::now();

            while (imageBuffer.size() >= BATCH_SIZE) {
                std::vector<fs::path> batch(imageBuffer.begin(), imageBuffer.begin() + BATCH_SIZE);
                batchQueue.push(batch);
                imageBuffer.erase(imageBuffer.begin(), imageBuffer.begin() + BATCH_SIZE);
                queueCV.notify_one();
                lastBatchTime = now;
            }

            if (!imageBuffer.empty() && now - lastBatchTime > BATCH_TIMEOUT) {
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

void signal_handler(int) {
    stopSignal = true;
    queueCV.notify_all();
}

int main() {
    init_dirs();
    std::signal(SIGINT, signal_handler);

    std::thread watcher(watch_folder);
    std::thread processor(process_batches);

    watcher.join();
    processor.join();

    return 0;
}