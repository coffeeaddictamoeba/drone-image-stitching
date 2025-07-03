#include "../include/mosaic.h"
#include "../include/metadata.h"
#include <bits/types/timer_t.h>
#include <ctime>
#include <future>
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

constexpr int CROP_HEIGHT = 8192;
constexpr int CROP_WIDTH = 8192;
const std::string resultDir = "results";
const std::string tilesDir = "tiles";

// need to think about better alternative than just creating new mosaics when images do not align
// need better multithreading option. experiment with bs or tbb libs?
// think more about alignment, maybe add some pitch-roll-yaw metadata? (however it is not present on my test image set) ->
// -> so go to odm datasets and try to align its images
// MAKE AN OPTION FOR AUTOMATIC IMG LOADING

std::string makeMosaicDir(const std::string& baseDir, int idx) {
    std::string dir = baseDir + "/" + baseDir + "_" + std::to_string(idx);
    if (!fs::exists(dir)) fs::create_directory(dir);
    return dir;
}

void saveTilesAsMosaic(std::string outputDir, MosaicBuilder builder, int idx) {
    time_t timer;
    cv::Rect boundsCropped;
    cv::Mat mosaicCropped = builder.mosaicFromTiles(outputDir, boundsCropped, CROP_WIDTH, CROP_HEIGHT, OffsetOrigin::CENTER);
    if (mosaicCropped.empty()) {
        std::cerr << "Failed to reconstruct mosaic.\n";
        exit(1);
    }

    if (!fs::exists(resultDir)) fs::create_directory(resultDir);

    std::string resCropped = resultDir + "/result_cropped_" + std::to_string(idx) + "_" + std::to_string(std::time(&timer)) + ".png";
    std::cout << "Cropped mosaic bounds: " << boundsCropped << "\n";
    cv::imwrite(resCropped, mosaicCropped);
    std::cout << "Saved reconstructed cropped mosaic to: " << resCropped << "\n";
}

std::vector<std::string> findAllMosaics(const std::string& baseDir) {
    std::vector<std::string> mosaicDirs;
    for (const auto& entry : fs::directory_iterator(baseDir)) {
        if (entry.is_directory()) {
            mosaicDirs.push_back(entry.path().string());
        }
    }
    std::sort(mosaicDirs.begin(), mosaicDirs.end());
    return mosaicDirs;
}

// questionable but ok for now
bool addToMosaics(const std::string& imagePath, const std::vector<std::string>& mosaicDirs, ExifToolPipe& exiftool, int& imageIdx) {
    std::atomic<bool> found(false);
    std::promise<std::tuple<std::string, std::unique_ptr<MosaicBuilder>>> promise;

    for (const auto& dir : mosaicDirs) {
        std::async(std::launch::async, [&, dir]() {
            if (found.load()) return;

            TileManager tileManager(dir, exiftool);
            auto builder = std::make_unique<MosaicBuilder>(exiftool, tileManager);
            if (builder->addImageToMosaic(imagePath)) {
                if (!found.exchange(true)) {
                    promise.set_value({dir, std::move(builder)});
                }
            }
        });
    }

    auto future = promise.get_future();
    if (future.wait_for(std::chrono::seconds(5)) == std::future_status::ready) {
        auto [dir, builder] = future.get();
        saveTilesAsMosaic(dir, *builder, imageIdx++);
        return true;
    }

    return false;
}

void useExistingMosaic(const std::string& baseOutputDir, ExifToolPipe& exiftool) {
    auto mosaicDirs = findAllMosaics(baseOutputDir);
    if (mosaicDirs.empty()) {
        std::cerr << "No mosaics found in: " << baseOutputDir << "\n";
        return;
    }

    int imageIdx = 0;
    std::string queued1, queued2;

    while (true) {
        std::string newImagePath;
        std::cout << "Enter path to an image to add (or blank to quit): ";
        std::getline(std::cin, newImagePath);
        if (newImagePath.empty()) break;

        if (addToMosaics(newImagePath, mosaicDirs, exiftool, imageIdx)) continue;

        std::cout << "Image did not fit any mosaic.\n";
        if (queued1.empty()) {
            queued1 = newImagePath;
            std::cout << "Queued image for next mosaic: " << queued1 << "\n";
        } else {
            queued2 = newImagePath;

            if (mosaicDirs.size() >= 4) {
                std::cerr << "Maximum number of mosaics reached.\n";
                queued1.clear();
                queued2.clear();
                continue;
            }

            int newIdx = mosaicDirs.size();
            std::string newDir = makeMosaicDir(baseOutputDir, newIdx);
            TileManager newTileManager(newDir, exiftool);
            MosaicBuilder newBuilder(exiftool, newTileManager);

            if (!newBuilder.stitchToTiles(queued1, queued2)) {
                std::cerr << "Failed to create new mosaic with queued images.\n";
                queued1.clear();
                queued2.clear();
                continue;
            }

            std::cout << "Created new mosaic: " << newDir << "\n";
            saveTilesAsMosaic(newDir, newBuilder, imageIdx++);
            mosaicDirs.push_back(newDir);

            queued1.clear();
            queued2.clear();
        }
    }
}

void startNewMosaic(const std::string& refImage, const std::string& targetImage, const std::string& baseOutputDir, ExifToolPipe& exiftool) {
    std::string mosaicDir = makeMosaicDir(baseOutputDir, 0);
    TileManager tileManager(mosaicDir, exiftool);
    MosaicBuilder builder(exiftool, tileManager);

    if (!builder.stitchToTiles(refImage, targetImage)) {
        std::cerr << "Initial mosaic stitching failed.\n";
        exit(1);
    }
    
    std::cout << "Mosaic stitching completed successfully in " << mosaicDir << ".\n";
    saveTilesAsMosaic(mosaicDir, builder, 0);
}

int main(int argc, char** argv) {
    if (argc == 3) {
        std::string refImage = argv[1];
        std::string targetImage = argv[2];

        if (!fs::exists(tilesDir)) fs::create_directory(tilesDir);

        ExifToolPipe exiftool;
        startNewMosaic(refImage, targetImage, tilesDir, exiftool);
        useExistingMosaic(tilesDir, exiftool);
        return 0;
    } else if (argc == 1) {
        ExifToolPipe exiftool;
        useExistingMosaic(tilesDir, exiftool);
        return 0;
    } else {
        std::cerr << "Usage:\n"
                  << argv[0] << " <ref_image> <target_image>\n";
        return 1;
    }
}