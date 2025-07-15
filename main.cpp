#include <csignal>
#include <cstddef>
#include <iostream>
#include <algorithm> 
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
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
    std::size_t retries = 3;
    int batchTimeoutSec = 5;
    int retryTimeoutSec = 3;
    bool useBigTIFF = true;
    bool compress = true;
};

enum class OdmRunResult {
    Success,
    CommandFailed,
    OrthophotoNotFound,
    OrthophotoZeroBytes,
    ValidationFailed
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

std::optional<std::chrono::steady_clock::time_point> bufferStartTime = std::nullopt;

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

bool validate_geotiff(const fs::path& path) {
    GDALDataset* dataset = (GDALDataset*) GDALOpen(path.string().c_str(), GA_ReadOnly);
    if (!dataset) {
        std::cerr << "[ERROR] GDAL failed to open image: " << path << std::endl;
        return false;
    }

    int width = dataset->GetRasterXSize();
    int height = dataset->GetRasterYSize();
    if (width == 0 || height == 0) {
        std::cerr << "[ERROR] Invalid image dimensions: " << path << std::endl;
        GDALClose(dataset);
        return false;
    }

    int bandCount = dataset->GetRasterCount();
    if (bandCount < 3) {
        std::cerr << "[ERROR] Not enough bands (RGB expected): " << path << std::endl;
        GDALClose(dataset);
        return false;
    }

    int sampleSize = 500;
    int xOff = std::max(0, width / 2 - sampleSize / 2);
    int yOff = std::max(0, height / 2 - sampleSize / 2);
    int winX = std::min(sampleSize, width - xOff);
    int winY = std::min(sampleSize, height - yOff);

    std::vector<float> buffer(winX * winY);
    double totalStdDev = 0.0;
    for (int i = 1; i <= 3; ++i) {
        GDALRasterBand* band = dataset->GetRasterBand(i);
        if (band->RasterIO(GF_Read, xOff, yOff, winX, winY,
                           buffer.data(), winX, winY, GDT_Float32,
                           0, 0) != CE_None) {
            std::cerr << "[ERROR] Failed to read band " << i << " of image: " << path << std::endl;
            GDALClose(dataset);
            return false;
        }

        double sum = 0.0, sqSum = 0.0;
        for (float val : buffer) {
            sum += val;
            sqSum += val * val;
        }
        double n = buffer.size();
        double mean = sum / n;
        double stddev = std::sqrt((sqSum / n) - (mean * mean));
        totalStdDev += stddev;
    }

    bool rgbValid = totalStdDev >= 25.0;

    bool alphaValid = true;
    if (bandCount >= 4) {
        GDALRasterBand* alphaBand = dataset->GetRasterBand(4);
        if (alphaBand->RasterIO(GF_Read, xOff, yOff, winX, winY,
                                buffer.data(), winX, winY, GDT_Float32,
                                0, 0) != CE_None) {
            std::cerr << "[WARN] Could not read alpha band — skipping transparency check." << std::endl;
        } else {
            int visiblePixels = std::count_if(buffer.begin(), buffer.end(), [](float v) {
                return v > 10.0f;
            });
            double visibleFraction = (double)visiblePixels / buffer.size();
            if (visibleFraction < 0.1) {
                alphaValid = false;
            }
        }
    }

    GDALClose(dataset);

    if (!rgbValid && !alphaValid) {
        std::cerr << "[ERROR] Rejected: low RGB variance and mostly transparent: " << path << std::endl;
        return false;
    }

    if (!rgbValid) {
        std::cerr << "[WARN] Low RGB variance (stddev=" << totalStdDev << "): " << path << std::endl;
    }

    if (!alphaValid) {
        std::cerr << "[WARN] Alpha band mostly transparent in central region: " << path << std::endl;
    }

    return true;
}

void merge_with_otb(const fs::path& ortho_path) {
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

    std::ofstream fileList("stitched/file_list_otb.txt");
    for (const auto& entry : fs::directory_iterator("stitched")) {
        if (entry.path().extension() == ".tif") {
            fileList << entry.path().string() << std::endl;
        }
    }
    fileList.close();

    // otbcli_Mosaic 9.1.0
    std::stringstream otbMosaic;
    otbMosaic << "otbcli_Mosaic -il ";
    for (const auto& entry : fs::directory_iterator("stitched")) {
        if (entry.path().extension() == ".tif") {
            otbMosaic << "\"" << entry.path().string() << "\" ";
        }
    }
    otbMosaic << "-comp.feather slim "
          << "-comp.feather.slim.length 10 "
          << "-comp.feather.slim.exponent 1.0 "
          << "-interpolator bco "
          << "-interpolator.bco.radius 2 "
          << "-out \"" << STITCHED_FILE << "\" uint8";

    if (run_command(otbMosaic.str()) != 0) {
        std::cerr << "[ERROR] OTB Mosaic command failed." << std::endl;
        return;
    }

    std::cout << "[INFO] Successfully updated stitched orthophoto using OTB: " << STITCHED_FILE << std::endl;
}

void clean_after_odm(const fs::path& batch_path, const std::string& message = "") {
    std::cerr << message << std::endl;
    std::cerr << "Cleaning batch path: " << batch_path << std::endl;
    for (const auto& entry : fs::directory_iterator(batch_path)) {
        if (entry.path().filename() == "images") {
            continue;
        }
        try {
            fs::remove_all(entry.path());
        } catch (const fs::filesystem_error& e) {
            std::cerr << "[WARNING] Failed to remove " << entry.path() << ": " << e.what() << std::endl;
        }
    }
}

bool run_odm_batch_successful(const fs::path& batch_path, const fs::path& ortho) {
    std::size_t retryCount = 0;
    while (retryCount < config.retries) {
        std::cout << "[INFO] Processing batch (attempt #" << retryCount + 1 << " of " << config.retries << "): " << batch_path << std::endl;

        OdmRunResult result = OdmRunResult::CommandFailed;

        if (run_odm_batch(batch_path)) {
            if (!fs::exists(ortho)) {
                result = OdmRunResult::OrthophotoNotFound;
                std::cerr << "[ERROR] Orthophoto not found after ODM run: " << ortho << std::endl;
            } else {
                try {
                    auto file_size = fs::file_size(ortho);
                    if (file_size == 0) {
                        result = OdmRunResult::OrthophotoZeroBytes;
                        std::cerr << "[ERROR] Orthophoto is 0 bytes: " << ortho << std::endl;
                    } else {
                        if (!validate_geotiff(ortho)) {
                            result = OdmRunResult::ValidationFailed;
                            std::cerr << "[ERROR] Orthophoto failed GDAL validation: " << ortho << std::endl;
                        } else {
                            result = OdmRunResult::Success;
                        }
                    }
                } catch (const fs::filesystem_error& e) {
                    result = OdmRunResult::OrthophotoNotFound;
                    std::cerr << "[ERROR] Failed to get file size for " << ortho << ": " << e.what() << std::endl;
                }
            }
        } else {
            std::string err = "[ERROR] ODM command failed for batch: " + batch_path.string() + ". Retrying...";
            clean_after_odm(batch_path, err);
            ++retryCount;
            std::this_thread::sleep_for(std::chrono::seconds(config.retryTimeoutSec));
            continue;
        }

        if (result == OdmRunResult::Success) {
            std::cout << "[INFO] Orthophoto passed validation: " << ortho << std::endl;
            return true;
        } else {
            std::string err_message;
            switch (result) {
                case OdmRunResult::OrthophotoNotFound:
                    err_message = "[ERROR] Orthophoto not found after ODM run.";
                    break;
                case OdmRunResult::OrthophotoZeroBytes:
                    err_message = "[ERROR] Orthophoto is 0 bytes. Deleting and retrying...";
                    break;
                case OdmRunResult::ValidationFailed:
                    err_message = "[ERROR] Orthophoto failed GDAL validation. Deleting and retrying...";
                    break;
                default:
                    err_message = "[ERROR] Unhandled ODM processing error.";
                    break;
            }
            clean_after_odm(batch_path, err_message);
            ++retryCount;
            std::this_thread::sleep_for(std::chrono::seconds(config.retryTimeoutSec));
        }
    }
    return false;
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
        fs::path ortho = batch_path / "odm_orthophoto" / "odm_orthophoto.tif";

        if (run_odm_batch_successful(batch_path, ortho)) {
            merge_with_otb(ortho);
        } else {
            std::cerr << "[FATAL] Batch failed after " << config.retries << " retries: " << batch_path << std::endl;
        }
    }
}

void watch_folder() {
    std::unordered_set<fs::path> seen;
    std::optional<std::chrono::steady_clock::time_point> bufferStartTime = std::nullopt;

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

            if (!new_images.empty()) {
                imageBuffer.insert(imageBuffer.end(), new_images.begin(), new_images.end());

                if (!bufferStartTime.has_value()) {
                    bufferStartTime = std::chrono::steady_clock::now();
                }
            }

            auto now = std::chrono::steady_clock::now();

            while (imageBuffer.size() >= config.batchSize) {
                std::vector<fs::path> batch(imageBuffer.begin(), imageBuffer.begin() + config.batchSize);
                batchQueue.push(batch);
                imageBuffer.erase(imageBuffer.begin(), imageBuffer.begin() + config.batchSize);
                queueCV.notify_one();

                if (imageBuffer.empty()) {
                    bufferStartTime = std::nullopt;
                } else {
                    bufferStartTime = now;
                }
            }

            if (bufferStartTime &&
                (now - *bufferStartTime >= std::chrono::seconds(config.batchTimeoutSec)) &&
                !imageBuffer.empty()) {

                batchQueue.push(imageBuffer);
                imageBuffer.clear();
                bufferStartTime = std::nullopt;
                queueCV.notify_one();
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(2));
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
        else if (arg == "--retry-amount" && i + 1 < argc)
            config.retries = std::stoi(argv[++i]);
        else if (arg == "--retry-delay" && i + 1 < argc)
            config.retryTimeoutSec = std::stoi(argv[++i]);
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
    GDALAllRegister();
    parse_args(argc, argv, config);

    init_dirs();
    std::signal(SIGINT, signal_handler);

    std::thread watcher(watch_folder);
    std::thread processor(process_batches);

    watcher.join();
    processor.join();

    return 0;
}