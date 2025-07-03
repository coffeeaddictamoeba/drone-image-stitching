#include "../include/mosaic.h"
#include "../include/metadata.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <chrono>
#include <thread>
#include <optional>
#include <atomic>

namespace fs = std::filesystem;

constexpr int CROP_HEIGHT = 8192;
constexpr int CROP_WIDTH = 8192;
constexpr const char* resultDir = "results";
constexpr const char* tilesDir = "tiles";
constexpr const char* processedFile = "processed.txt";

class MosaicOrchestrator {
    std::string baseOutputDir;
    ExifToolPipe& exiftool;
    std::vector<std::string> mosaicDirs;
    std::unordered_set<std::string> processed;
    std::optional<std::string> queuedImage;
    int mosaicIdx = 0;
    int imageIdx = 0;

public:
    MosaicOrchestrator(const std::string& outputDir, ExifToolPipe& tool) : baseOutputDir(outputDir), exiftool(tool) {
        if (!fs::exists(baseOutputDir)) fs::create_directory(baseOutputDir);
        if (!fs::exists(resultDir)) fs::create_directory(resultDir);
        mosaicDirs = findAllMosaics(baseOutputDir);
        mosaicIdx = mosaicDirs.size();
        loadProcessed();
    }

    void loadProcessed() {
        std::ifstream in(processedFile);
        std::string line;
        while (std::getline(in, line)) {
            processed.insert(line);
        }
    }

    void markProcessed(const std::string& path) {
        processed.insert(path);
        std::ofstream out(processedFile, std::ios::app);
        out << path << "\n";
    }

    bool alreadyProcessed(const std::string& path) {
        return processed.count(path);
    }

    std::vector<std::string> findAllMosaics(const std::string& dir) {
        std::vector<std::string> dirs;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_directory()) dirs.push_back(entry.path().string());
        }
        std::sort(dirs.begin(), dirs.end());
        return dirs;
    }

    std::string makeMosaicDir(int idx) {
        std::string dir = baseOutputDir + "/" + baseOutputDir + "_" + std::to_string(idx);
        if (!fs::exists(dir)) fs::create_directory(dir);
        return dir;
    }

    void saveMosaic(const std::string& outputDir, MosaicBuilder& builder) {
        time_t timer;
        cv::Rect bounds;
        cv::Mat mosaic = builder.mosaicFromTiles(outputDir, bounds, CROP_WIDTH, CROP_HEIGHT, OffsetOrigin::CENTER);
        if (mosaic.empty()) {
            std::cerr << "Failed to reconstruct mosaic.\n";
            return;
        }

        std::string filename = std::string(resultDir) + "/result_cropped_" + std::to_string(imageIdx++) + "_" + std::to_string(std::time(&timer)) + ".png";
        cv::imwrite(filename, mosaic);
        std::cout << "Saved mosaic: " << filename << "\n";
    }

    void addImage(const std::string& imagePath) {
        if (alreadyProcessed(imagePath)) return;

        for (const auto& dir : mosaicDirs) {
            TileManager tm(dir, exiftool);
            MosaicBuilder builder(exiftool, tm);
            if (builder.addImageToMosaic(imagePath)) {
                saveMosaic(dir, builder);
                markProcessed(imagePath);
                return;
            }
        }

        if (!queuedImage.has_value()) {
            queuedImage = imagePath;
            std::cout << "Queued image for new mosaic: " << imagePath << "\n";
        } else {
            std::string newDir = makeMosaicDir(mosaicIdx++);
            TileManager tm(newDir, exiftool);
            MosaicBuilder builder(exiftool, tm);

            if (builder.stitchToTiles(queuedImage.value(), imagePath)) {
                std::cout << "Created new mosaic: " << newDir << "\n";
                saveMosaic(newDir, builder);
                mosaicDirs.push_back(newDir);
                markProcessed(queuedImage.value());
                markProcessed(imagePath);
            } else {
                std::cerr << "Failed to stitch new mosaic.\n";
            }
            queuedImage.reset();
        }
    }

    void interactive() {
        while (true) {
            std::string path;
            std::cout << "Enter image path (or blank to quit): ";
            std::getline(std::cin, path);
            if (path.empty()) break;
            addImage(path);
        }
    }

    void monitor(const std::string& dir) {
        std::cout << "Monitoring: " << dir << "\n";
        while (true) {
            for (const auto& entry : fs::directory_iterator(dir)) {
                if (!entry.is_regular_file()) continue;
                addImage(entry.path().string());
            }
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    }

    void startWithPair(const std::string& ref, const std::string& target) {
        std::string newDir = makeMosaicDir(mosaicIdx++);
        TileManager tm(newDir, exiftool);
        MosaicBuilder builder(exiftool, tm);

        if (!builder.stitchToTiles(ref, target)) {
            std::cerr << "Initial mosaic stitching failed.\n";
            exit(1);
        }

        saveMosaic(newDir, builder);
        mosaicDirs.push_back(newDir);
        markProcessed(ref);
        markProcessed(target);
    }
};

int main(int argc, char** argv) {
    ExifToolPipe exiftool;

    if (argc == 3 && std::string(argv[1]) != "-a") {
        std::string ref = argv[1];
        std::string target = argv[2];
        MosaicOrchestrator orchestrator(tilesDir, exiftool);
        orchestrator.startWithPair(ref, target);
        orchestrator.interactive();

    } else if (argc == 1) {
        MosaicOrchestrator orchestrator(tilesDir, exiftool);
        orchestrator.interactive();

    } else if (argc == 3 && std::string(argv[1]) == "-a") {
        std::string watchDir = argv[2];
        MosaicOrchestrator orchestrator(tilesDir, exiftool);
        orchestrator.monitor(watchDir);

    } else {
        std::cerr << "Usage:\n"
                  << argv[0] << " <ref_image> <target_image>\n"
                  << argv[0] << "\n"
                  << argv[0] << " -a <watch_directory>\n";
        return 1;
    }

    return 0;
}