#include <csignal>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <thread>
#include <mutex>
#include <chrono>
#include <set>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <condition_variable>
#include <gdal/gdal_priv.h>
#include <gdal/ogr_spatialref.h>

namespace fs = std::filesystem;

std::mutex mosaicMutex;
std::condition_variable batchFinished;
bool stopSignal = false;

const std::string INCOMING = "incoming";
const std::string BATCHES = "batches";
const std::string STITCHED = "stitched/final_orthophoto.tif";
const int BATCH_SIZE = 5;

void init_dirs() {
    fs::create_directories(INCOMING);
    fs::create_directories(BATCHES);
    fs::create_directories("stitched");
}

fs::path create_batch(const std::vector<fs::path>& images, int batch_id) {
    std::stringstream ss;
    ss << "batch_" << std::setw(3) << std::setfill('0') << batch_id;
    fs::path batch_path = BATCHES + '/' + ss.str() + '/' + "images";
    fs::create_directories(batch_path);
    for (const auto& img : images) {
        fs::rename(img, batch_path / img.filename());
    }
    return batch_path.parent_path();
}

void run_odm_batch(const fs::path& batch_path) {
    std::string abs_path = fs::absolute(batch_path).string();
    std::string cmd = "docker run --rm -v \"" + abs_path +
                      "\":/datasets/project opendronemap/odm "
                      "--project-path /datasets project "
                      "--fast-orthophoto --skip-3dmodel";
    std::cout << "[ODM] Running: " << cmd << "\n";
    int result = std::system(cmd.c_str());
    if (result != 0) {
        std::cerr << "[ODM] Batch failed: " << batch_path << "\n";
    } else {
        std::cout << "[ODM] Batch complete: " << batch_path << "\n";
    }
}

void merge_orthophoto(const std::string& ortho_path, const std::string& mosaic_path) {
    std::lock_guard<std::mutex> lock(mosaicMutex);

    GDALAllRegister();
    GDALDataset* src = (GDALDataset*)GDALOpen(ortho_path.c_str(), GA_ReadOnly);
    if (!src) {
        std::cerr << "[GDAL] Failed to open: " << ortho_path << "\n";
        return;
    }

    if (!fs::exists(mosaic_path)) {
        GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
        char** papszOptions = NULL;
        GDALDataset* dst = driver->CreateCopy(mosaic_path.c_str(), src, FALSE, papszOptions, NULL, NULL);
        if (dst) {
            GDALClose(dst);
        } else {
            std::cerr << "[GDAL] Failed to create mosaic copy.\n";
        }

        std::cout << "[STITCH] Created new mosaic with: " << ortho_path << "\n";
    } else {
        // Merge with existing using gdalwarp
        std::string tmp_output = "stitched/tmp_mosaic.tif";
        std::string cmd = "gdalwarp -overwrite -r cubic " + mosaic_path + " " + ortho_path + " " + tmp_output;
        std::cout << "[GDAL] Merging with gdalwarp...\n";
        int result = std::system(cmd.c_str());
        if (result == 0) {
            fs::rename(tmp_output, mosaic_path);
            std::cout << "[STITCH] Updated mosaic.\n";
        } else {
            std::cerr << "[GDAL] Merge failed.\n";
        }
    }

    GDALClose(src);
}

void process_batch(int batch_id, std::vector<fs::path> images) {
    fs::path batch_path = create_batch(images, batch_id);
    run_odm_batch(batch_path);

    fs::path ortho = batch_path / "odm_orthophoto" / "odm_orthophoto.tif";
    if (fs::exists(ortho)) {
        merge_orthophoto(ortho.string(), STITCHED);
    } else {
        std::cerr << "[WARN] Orthophoto missing for batch: " << batch_path << "\n";
    }

    batchFinished.notify_all(); 
}

void watch_folder() {
    std::set<fs::path> processed;
    int batch_id = 1;

    while (!stopSignal) {
        std::vector<fs::path> new_images;
        for (const auto& f : fs::directory_iterator(INCOMING)) {
            if ((f.path().extension() == ".jpg" || f.path().extension() == ".JPG") &&
                processed.find(f.path()) == processed.end()) {
                new_images.push_back(f.path());
                processed.insert(f.path());
            }
            if (new_images.size() >= BATCH_SIZE)
                break;
        }

        if (new_images.size() >= BATCH_SIZE) {
            std::thread(process_batch, batch_id++, new_images).detach();
        }

        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

int main() {
    init_dirs();

    std::thread watcher(watch_folder);

    std::signal(SIGINT, [](int) {
        stopSignal = true;
    });

    watcher.join();
    return 0;
}
