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
    std::size_t batchSize = 10; // optimal
    std::size_t retries = 3;
    int batchTimeoutSec = 5;
    int retryTimeoutSec = 3;
    bool useBigTIFF = true;
    bool compress = true;
    bool retry = true;
    bool waitForBatchSize = false;
    bool savePreviousOrthophoto = false;
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

bool get_raster_info(const fs::path& path, double gt[6], char**& proj_wkt_ptr, int& width, int& height) {
    GDALDataset* poDataset = (GDALDataset*)GDALOpen(path.string().c_str(), GA_ReadOnly);
    if (poDataset == nullptr) {
        std::cerr << "[ERROR] Could not open raster: " << path << std::endl;
        return false;
    }

    if (poDataset->GetGeoTransform(gt) != CE_None) {
        std::cerr << "[ERROR] Could not get geotransform for: " << path << std::endl;
        GDALClose(poDataset);
        return false;
    }

    const char* pszWKT = poDataset->GetProjectionRef();
    if (pszWKT == nullptr || std::string(pszWKT).empty()) {
        std::cerr << "[WARNING] No projection found for: " << path << std::endl;
        proj_wkt_ptr = nullptr;
    } else {
        proj_wkt_ptr = (char**)CPLMalloc(sizeof(char*));
        *proj_wkt_ptr = CPLStrdup(pszWKT);
    }

    width = poDataset->GetRasterXSize();
    height = poDataset->GetRasterYSize();

    GDALClose(poDataset);
    return true;
}

void calculate_union_extent(double gt1[6], int w1, int h1, double gt2[6], int w2, int h2, double& union_minX, double& union_maxY, double& union_maxX, double& union_minY, double& avg_resX, double& avg_resY) {
    double x_min1 = gt1[0];
    double y_max1 = gt1[3];
    double x_max1 = gt1[0] + gt1[1] * w1 + gt1[2] * h1;
    double y_min1 = gt1[3] + gt1[4] * w1 + gt1[5] * h1;

    if (gt1[2] != 0.0 || gt1[4] != 0.0) {
        double x1_ul = gt1[0]; double y1_ul = gt1[3];
        double x1_ur = gt1[0] + gt1[1] * w1; double y1_ur = gt1[3] + gt1[4] * w1;
        double x1_ll = gt1[0] + gt1[2] * h1; double y1_ll = gt1[3] + gt1[5] * h1;
        double x1_lr = gt1[0] + gt1[1] * w1 + gt1[2] * h1; double y1_lr = gt1[3] + gt1[4] * w1 + gt1[5] * h1;

        x_min1 = std::min({x1_ul, x1_ur, x1_ll, x1_lr});
        x_max1 = std::max({x1_ul, x1_ur, x1_ll, x1_lr});
        y_min1 = std::min({y1_ul, y1_ur, y1_ll, y1_lr});
        y_max1 = std::max({y1_ul, y1_ur, y1_ll, y1_lr});
    }

    double x_min2 = gt2[0];
    double y_max2 = gt2[3];
    double x_max2 = gt2[0] + gt2[1] * w2 + gt2[2] * h2;
    double y_min2 = gt2[3] + gt2[4] * w2 + gt2[5] * h2;

    if (gt2[2] != 0.0 || gt2[4] != 0.0) {
        double x2_ul = gt2[0]; double y2_ul = gt2[3];
        double x2_ur = gt2[0] + gt2[1] * w2; double y2_ur = gt2[3] + gt2[4] * w2;
        double x2_ll = gt2[0] + gt2[2] * h2; double y2_ll = gt2[3] + gt2[5] * h2;
        double x2_lr = gt2[0] + gt2[1] * w2 + gt2[2] * h2; double y2_lr = gt2[3] + gt2[4] * w2 + gt2[5] * h2;

        x_min2 = std::min({x2_ul, x2_ur, x2_ll, x2_lr});
        x_max2 = std::max({x2_ul, x2_ur, x2_ll, x2_lr});
        y_min2 = std::min({y2_ul, y2_ur, y2_ll, y2_lr});
        y_max2 = std::max({y2_ul, y2_ur, y2_ll, y2_lr});
    }

    union_minX = std::min(x_min1, x_min2);
    union_maxY = std::max(y_max1, y_max2);
    union_maxX = std::max(x_max1, x_max2);
    union_minY = std::min(y_min1, y_min2);

    avg_resX = (std::abs(gt1[1]) + std::abs(gt2[1])) / 2.0;
    avg_resY = (std::abs(gt1[5]) + std::abs(gt2[5])) / 2.0;
}

void merge_with_otb(const fs::path& ortho_path) {
    std::lock_guard<std::mutex> lock(mosaicMutex);

    auto now   = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ts;
    ts << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");

    fs::create_directories("stitched");
    
    if (!fs::exists(STITCHED_FILE)) {
        try {
            fs::copy_file(ortho_path, STITCHED_FILE, fs::copy_options::overwrite_existing);
            std::cout << "[INFO] Mosaic initialized with: " << STITCHED_FILE << "\n";
            return;
        } catch (const fs::filesystem_error& e) {
            std::cerr << "[ERROR] Failed to initialize mosaic: " << e.what() << std::endl;
            return;
        }
    }

    if (config.savePreviousOrthophoto) {
        fs::path bak = fs::path("stitched") / ("prev_mosaic_" + ts.str() + ".tif");
        try {
            fs::copy_file(STITCHED_FILE, bak, fs::copy_options::overwrite_existing);
            std::cout << "[DEBUG] Previous mosaic backed up to: " << bak << std::endl;
        } catch (const fs::filesystem_error& e) {
            std::cerr << "[WARN] Failed to backup previous mosaic: " << e.what() << std::endl;
        }
    }

    double mosaic_gt[6];
    char** mosaic_proj_wkt_ptr = nullptr;
    int mosaic_width, mosaic_height;
    if (!get_raster_info(STITCHED_FILE, mosaic_gt, mosaic_proj_wkt_ptr, mosaic_width, mosaic_height)) {
        std::cerr << "[ERROR] Could not get info for existing mosaic: " << STITCHED_FILE << std::endl;
        return;
    }
    std::string mosaic_proj_wkt = (mosaic_proj_wkt_ptr && *mosaic_proj_wkt_ptr) ? *mosaic_proj_wkt_ptr : "";

    double new_ortho_gt[6];
    char** new_ortho_proj_wkt_ptr = nullptr;
    int new_ortho_width, new_ortho_height;
    if (!get_raster_info(ortho_path, new_ortho_gt, new_ortho_proj_wkt_ptr, new_ortho_width, new_ortho_height)) {
        std::cerr << "[ERROR] Could not get info for new orthophoto: " << ortho_path << std::endl;
        if (mosaic_proj_wkt_ptr) { CPLFree(*mosaic_proj_wkt_ptr); CPLFree(mosaic_proj_wkt_ptr); }
        return;
    }

    double union_minX, union_maxY, union_maxX, union_minY;
    double avg_resX, avg_resY;

    calculate_union_extent(
        mosaic_gt, mosaic_width, mosaic_height,
        new_ortho_gt, new_ortho_width, new_ortho_height,
        union_minX, union_maxY, union_maxX, union_minY,
        avg_resX, avg_resY
    );

    fs::path mosaic_wkt_tmp_file = fs::path("stitched") / ("mosaic_proj_" + ts.str() + ".wkt");
    if (!mosaic_proj_wkt.empty()) {
        std::ofstream wkt_file(mosaic_wkt_tmp_file);
        if (!wkt_file.is_open()) {
            std::cerr << "[ERROR] Could not create temporary WKT file: " << mosaic_wkt_tmp_file << std::endl;
            if (mosaic_proj_wkt_ptr) { CPLFree(*mosaic_proj_wkt_ptr); CPLFree(mosaic_proj_wkt_ptr); }
            if (new_ortho_proj_wkt_ptr) { CPLFree(*new_ortho_proj_wkt_ptr); CPLFree(new_ortho_proj_wkt_ptr); }
            return;
        }
        wkt_file << mosaic_proj_wkt;
        wkt_file.close();
    } else {
        std::cerr << "[ERROR] Mosaic has no projection, cannot proceed with gdalwarp for growth.\n";
        if (mosaic_proj_wkt_ptr) { CPLFree(*mosaic_proj_wkt_ptr); CPLFree(mosaic_proj_wkt_ptr); }
        if (new_ortho_proj_wkt_ptr) { CPLFree(*new_ortho_proj_wkt_ptr); CPLFree(new_ortho_proj_wkt_ptr); }
        return;
    }

    fs::path new_ortho_warped_expanded = fs::path("stitched") / ("new_ortho_warped_expanded_" + ts.str() + ".tif");
    std::stringstream gdalwarp_cmd;
    gdalwarp_cmd << "gdalwarp "
                 << "-t_srs \"" << mosaic_wkt_tmp_file.string() << "\" "
                 << "-te " << std::fixed << std::setprecision(10) << union_minX << " "
                 << std::fixed << std::setprecision(10) << union_minY << " "
                 << std::fixed << std::setprecision(10) << union_maxX << " "
                 << std::fixed << std::setprecision(10) << union_maxY << " "
                 << "-tr " << std::fixed << std::setprecision(10) << avg_resX << " "
                 << std::fixed << std::setprecision(10) << avg_resY << " "
                 << "-r bilinear "
                 << "-dstalpha "
                 << "-srcnodata 0 "
                 << "-dstnodata 0 "
                 << "\"" << ortho_path.string() << "\" "
                 << "\"" << new_ortho_warped_expanded.string() << "\" ";

    std::cout << "[CMD] Running: " << gdalwarp_cmd.str() << std::endl;
    if (run_command(gdalwarp_cmd.str()) != 0) {
        std::cerr << "[ERROR] gdalwarp failed to align and expand new orthophoto. Cannot grow mosaic.\n";
        if (mosaic_proj_wkt_ptr) { CPLFree(*mosaic_proj_wkt_ptr); CPLFree(mosaic_proj_wkt_ptr); }
        if (new_ortho_proj_wkt_ptr) { CPLFree(*new_ortho_proj_wkt_ptr); CPLFree(new_ortho_proj_wkt_ptr); }
        if (fs::exists(mosaic_wkt_tmp_file)) fs::remove(mosaic_wkt_tmp_file);
        return;
    }
    std::cout << "[INFO] New orthophoto warped and expanded to: " << new_ortho_warped_expanded << std::endl;

    fs::path tmp_mosaic_unoptimized = fs::path("stitched") / ("tmp_mosaic_unoptimized_" + ts.str() + ".tif");
    {
        std::stringstream mosaic_cmd;
        mosaic_cmd << "otbcli_Mosaic -il "
                   << "\"" << STITCHED_FILE << "\" "
                   << "\"" << new_ortho_warped_expanded.string() << "\" "
                   << "-comp.feather slim "
                   << "-comp.feather.slim.length 10 "
                   << "-comp.feather.slim.exponent 1.0 "
                   << "-harmo.method band "
                   << "-harmo.cost rmse "
                   << "-interpolator bco "
                   << "-interpolator.bco.radius 2 "
                   << "-out \"" << tmp_mosaic_unoptimized.string() << "\" uint8 ";

        std::cout << "[DEBUG] OTB Mosaic Command: " << mosaic_cmd.str() << std::endl;
        if (run_command(mosaic_cmd.str()) != 0) {
            std::cerr << "[ERROR] otbcli_Mosaic failed. Command: " << mosaic_cmd.str() << std::endl;
            if (mosaic_proj_wkt_ptr) { CPLFree(*mosaic_proj_wkt_ptr); CPLFree((void*)mosaic_proj_wkt_ptr); }
            if (new_ortho_proj_wkt_ptr) { CPLFree(*new_ortho_proj_wkt_ptr); CPLFree((void*)new_ortho_proj_wkt_ptr); }
            if (fs::exists(mosaic_wkt_tmp_file)) fs::remove(mosaic_wkt_tmp_file);
            fs::remove(new_ortho_warped_expanded);
            return;
        }
    }

    fs::path tmp_mosaic_optimized = fs::path("stitched") / ("tmp_mosaic_optimized_" + ts.str() + ".tif");
    std::stringstream gdal_translate_cmd;
    gdal_translate_cmd << "gdal_translate "
             << "\"" << tmp_mosaic_unoptimized.string() << "\" "
             << "\"" << tmp_mosaic_optimized.string() << "\" "
             << "-co \"TILED=YES\" ";

    if (config.compress) gdal_translate_cmd << "-co \"COMPRESS=DEFLATE\" ";
    if (config.useBigTIFF) gdal_translate_cmd << "-co \"BIGTIFF=YES\" ";
    gdal_translate_cmd << "-co \"BLOCKXSIZE=" << config.blockSize << "\" "
             << "-co \"BLOCKYSIZE=" << config.blockSize << "\" ";

    std::cout << "[CMD] Running: " << gdal_translate_cmd.str() << std::endl;
    if (run_command(gdal_translate_cmd.str()) != 0) {
        std::cerr << "[ERROR] gdal_translate failed to optimize output mosaic. Proceeding with unoptimized file.\n";
        try {
            fs::rename(tmp_mosaic_unoptimized, STITCHED_FILE);
            std::cout << "[INFO] Successfully updated stitched orthomosaic (unoptimized) at: " << STITCHED_FILE << std::endl;
        } catch (const fs::filesystem_error& e) {
            std::cerr << "[ERROR] Failed to replace stitched file with unoptimized version: " << e.what() << std::endl;
        }
        if (mosaic_proj_wkt_ptr) { CPLFree(*mosaic_proj_wkt_ptr); CPLFree(mosaic_proj_wkt_ptr); }
        if (new_ortho_proj_wkt_ptr) { CPLFree(*new_ortho_proj_wkt_ptr); CPLFree(new_ortho_proj_wkt_ptr); }
        if (fs::exists(mosaic_wkt_tmp_file)) fs::remove(mosaic_wkt_tmp_file);
        fs::remove(new_ortho_warped_expanded);
        return;
    }

    try {
        fs::rename(tmp_mosaic_optimized, STITCHED_FILE);
        std::cout << "[INFO] Successfully updated stitched orthomosaic at: " << STITCHED_FILE << std::endl;
    } catch (const fs::filesystem_error& e) {
        std::cerr << "[ERROR] Failed to replace stitched file: " << e.what() << std::endl;
        return;
    }

    try {
        if (mosaic_proj_wkt_ptr) { CPLFree(*mosaic_proj_wkt_ptr); CPLFree(mosaic_proj_wkt_ptr); }
        if (new_ortho_proj_wkt_ptr) { CPLFree(*new_ortho_proj_wkt_ptr); CPLFree(new_ortho_proj_wkt_ptr); }
        if (fs::exists(mosaic_wkt_tmp_file)) fs::remove(mosaic_wkt_tmp_file);
        fs::remove(new_ortho_warped_expanded);
        fs::remove(tmp_mosaic_unoptimized);
        fs::remove(tmp_mosaic_optimized);
    } catch (const fs::filesystem_error& e) {
        std::cerr << "[WARN] Failed to clean up temporary files: " << e.what() << std::endl;
    }
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

    if (!config.retry) config.retries = 1;

    while (retryCount < config.retries && !stopSignal) {
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
            if (stopSignal) return false;
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
            if (stopSignal) return false;
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

        if (stopSignal) break;     

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

            if (!config.waitForBatchSize && bufferStartTime &&
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
        else if (arg == "--no-retry")
            config.retry = false;
        else if (arg == "--save-prev")
            config.savePreviousOrthophoto = true;
        else if (arg == "--wait-batch-size")
            config.waitForBatchSize = true;
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