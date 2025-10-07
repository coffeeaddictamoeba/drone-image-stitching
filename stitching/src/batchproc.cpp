#include "../include/batchproc.h"

#ifndef _WIN32
    #include <sys/types.h>
    #include <sys/wait.h>
    #include <unistd.h>
    #include <signal.h>
#endif

#include <thread>
#include <algorithm>
#include <cmath>
#include <fstream>

#define RESET   "\033[0m"
#define RED     "\033[31m"      // Errors
#define YELLOW  "\033[33m"      // Warnings
#define GREEN   "\033[32m"      // Success

BatchProcessor::BatchProcessor(const Config& config_ref, std::queue<BatchTask>& batch_queue, std::mutex& queue_mutex, std::condition_variable& queue_cv, std::atomic<bool>& stop_signal)
    : config_(config_ref), batchQueue_(batch_queue), queueMutex_(queue_mutex), queueCV_(queue_cv), stopSignal_(stop_signal) {
    fs::create_directories(config_.batchDir);
    fs::create_directories(fs::path(config_.stitchedFile).parent_path());
}

void BatchProcessor::processBatchesLoop() {
    while (!stopSignal_.load()) {
        std::unique_lock<std::mutex> lock(queueMutex_);
        queueCV_.wait(lock, [this] { return !batchQueue_.empty() || stopSignal_.load(); });

        if (stopSignal_.load()) {
            // no need to re-lock, we already own it here
            while (!batchQueue_.empty()) batchQueue_.pop();
            break;
        }

        if (batchQueue_.empty()) continue;

        BatchTask task = batchQueue_.front();
        batchQueue_.pop();
        lock.unlock();

        if (stopSignal_.load()) break;

        fs::path batch_path = createBatchDirectory(task.images, task.batch_id);
        fs::path ortho_path = batch_path / "odm_orthophoto" / "odm_orthophoto.tif";

        if (runOdmBatchSuccessful(batch_path, ortho_path)) {
            mergeWithOTB(ortho_path);
        } else {
            std::cerr << RED << "[FATAL] Batch failed after " << config_.retries << " retries: " << batch_path << RESET << std::endl;
        }

        // cleanup
        for (const auto& entry : fs::directory_iterator(batch_path)) {
            if (entry.path().filename() == "images") continue;
            try {
                fs::remove_all(entry.path());
            } catch (const fs::filesystem_error& e) {
                std::cerr << RED << "[WARNING] Failed to remove " << entry.path() << ": " << e.what() << RESET << std::endl;
            }
        }
    }

    std::cout << "[INFO] BatchProcessor loop stopped." << std::endl;
}

fs::path BatchProcessor::createBatchDirectory(const std::vector<fs::path>& images, int batch_id) {
    std::stringstream ss;
    ss << "batch_" << std::setw(3) << std::setfill('0') << batch_id;
    fs::path batch_root_path = fs::path(config_.batchDir) / ss.str();
    fs::path images_path = batch_root_path / "images";
    
    fs::create_directories(images_path);
    
    for (const auto& img : images) {
        try {
            fs::rename(img, images_path / img.filename());
        } catch (const fs::filesystem_error& e) {
            std::cerr << RED << "[ERROR] Failed to move image " << img << " to batch " << images_path << ": " << e.what() << RESET << std::endl;
        }
    }

    return batch_root_path;
}

int BatchProcessor::runCommand(const std::string& cmd) {
    if (stopSignal_.load()) return 1;

    #ifndef _WIN32
        pid_t pid = fork();
        if (pid == 0) {
            // Child
            execl("/bin/sh", "sh", "-c", cmd.c_str(), nullptr);
            _exit(127);
        }

        int status = 0;
        while (true) {
            pid_t w = waitpid(pid, &status, WNOHANG);
            if (w == pid) break; // process finished

            if (stopSignal_.load()) {
                std::cerr << "[INFO] Stopping command: " << cmd << std::endl;
                kill(pid, SIGTERM);                 // try graceful stop first
                std::this_thread::sleep_for(std::chrono::seconds(1));
                kill(pid, SIGKILL);                 // force kill if still alive
                waitpid(pid, &status, 0);
                return -1;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    #else
        return std::system(cmd.c_str());
    #endif
}

fs::path BatchProcessor::mapBatchPathForOdm(const fs::path& batch_path) {
    fs::path abs_path = fs::absolute(batch_path);

    // Case 1: running in a container linked to DinD
    const char* docker_host_env = std::getenv("DOCKER_HOST");
    if (docker_host_env) {
        // detect mount root dynamically (try HOST_PROJECT_ROOT)
        const char* host_root_env = std::getenv("HOST_PROJECT_ROOT");
        std::string container_root = "/prototype"; // default container root
        if (host_root_env) {
            fs::path host_root = host_root_env;
            if (abs_path.string().find(container_root) == 0) {
                return host_root / abs_path.string().substr(container_root.size());
            }
        }
    }

    // Case 2: running directly on host or inside container without DinD
    return abs_path;
}

bool BatchProcessor::runOdmBatchInternal(const fs::path& batch_path) {
    #ifdef _WIN32 // probably will be removed
        std::string abs_path = fs::absolute(batch_path).string();

        std::string docker_host_path = abs_path;
        std::replace(docker_host_path.begin(), docker_host_path.end(), '\\', '/');

        std::string cmd = "docker run --rm "
                        "-v \"" + docker_host_path + ":/datasets/project\" "
                        "-w /datasets/project "
                        "opendronemap/odm "
                        "--project-path /datasets project "
                        "--fast-orthophoto --skip-3dmodel";
    #else
        std::string uid = std::to_string(getuid());
        std::string gid = std::to_string(getgid());

        std::string host_abs_path = mapBatchPathForOdm(batch_path).string();
        std::string cmd = "docker run --rm "
                    "--user " + uid + ":" + gid + " "
                    "-v \"" + host_abs_path + ":/datasets/project\" "
                    "-w /datasets/project "
                    "opendronemap/odm "
                    "--project-path /datasets project "
                    "--fast-orthophoto --skip-3dmodel";
    #endif
    return runCommand(cmd) == 0;
}

bool BatchProcessor::validateGeotiff(const fs::path& path) {
    GDALDatasetRAII dataset((GDALDataset*)GDALOpen(path.string().c_str(), GA_ReadOnly));
    if (!dataset) {
        std::cerr << RED << "[ERROR] GDAL failed to open image: " << path << RESET << std::endl;
        return false;
    }

    int width = dataset->GetRasterXSize();
    int height = dataset->GetRasterYSize();
    if (width == 0 || height == 0) {
        std::cerr << RED << "[ERROR] Invalid image dimensions: " << path << RESET << std::endl;
        return false;
    }

    int bandCount = dataset->GetRasterCount();
    if (bandCount < 3) {
        std::cerr << RED << "[ERROR] Not enough bands (RGB expected): " << path << RESET << std::endl;
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
        if (!band || band->RasterIO(GF_Read, xOff, yOff, winX, winY,
                           buffer.data(), winX, winY, GDT_Float32,
                           0, 0) != CE_None) {
            std::cerr << RED << "[ERROR] Failed to read band " << i << " of image: " << path << RESET << std::endl;
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

    bool rgbValid = totalStdDev >= config_.rgbValidationThreshold;

    bool alphaValid = true;
    if (bandCount >= 4) {
        GDALRasterBand* alphaBand = dataset->GetRasterBand(4);
        if (!alphaBand || alphaBand->RasterIO(GF_Read, xOff, yOff, winX, winY,
                                buffer.data(), winX, winY, GDT_Float32,
                                0, 0) != CE_None) {
            std::cerr << RED << "[WARN] Could not read alpha band — skipping transparency check." << RESET << std::endl;
        } else {
            int visiblePixels = std::count_if(buffer.begin(), buffer.end(), [](float v) {
                return v > 10.0f; // Check if more than 10% of image is transparent
            });
            double visibleFraction = (double)visiblePixels / buffer.size();
            if (visibleFraction < config_.alphaValidationThreshold) {
                alphaValid = false;
            }
        }
    }

    if (!rgbValid && !alphaValid) {
        std::cerr << RED << "[ERROR] Rejected: low RGB variance and mostly transparent: " << path << RESET << std::endl;
        return false;
    }

    if (!rgbValid) {
        std::cerr << YELLOW << "[WARN] Low RGB variance (stddev=" << totalStdDev << "): " << path << RESET << std::endl;
    }

    if (!alphaValid) {
        std::cerr << YELLOW << "[WARN] Alpha band mostly transparent in central region: " << path << RESET << std::endl;
    }

    return true;
}

bool BatchProcessor::getRasterInfo(const fs::path& path, double gt[6], std::optional<std::string>& proj_wkt, int& width, int& height) {
    GDALDatasetRAII dataset((GDALDataset*)GDALOpen(path.string().c_str(), GA_ReadOnly));
    if (!dataset) {
        std::cerr << RED << "[ERROR] Could not open raster: " << path << RESET << std::endl;
        return false;
    }

    if (dataset->GetGeoTransform(gt) != CE_None) {
        std::cerr << RED << "[ERROR] Could not get geotransform for: " << path << RESET << std::endl;
        return false;
    }

    const char* pszWKT = dataset->GetProjectionRef();
    if (pszWKT == nullptr || std::string(pszWKT).empty()) {
        std::cerr << RED << "[WARNING] No projection found for: " << path << RESET << std::endl;
        proj_wkt = std::nullopt;
    } else {
        proj_wkt = std::string(pszWKT); // Directly convert to std::string
    }

    width = dataset->GetRasterXSize();
    height = dataset->GetRasterYSize();

    return true;
}

void BatchProcessor::calculateUnionExtent(double gt1[6], int w1, int h1, double gt2[6], int w2, int h2,
                                          double& union_minX, double& union_maxY, double& union_maxX, double& union_minY,
                                          double& avg_resX, double& avg_resY) {
    auto get_corners = [](double gt[6], int w, int h) -> std::vector<std::pair<double, double>> {
        std::vector<std::pair<double, double>> corners;
        corners.push_back({gt[0], gt[3]}); // Ul: (0,0)
        corners.push_back({gt[0] + gt[1] * w, gt[3] + gt[4] * w}); // Ur: (w,0)
        corners.push_back({gt[0] + gt[1] * w + gt[2] * h, gt[3] + gt[4] * w + gt[5] * h}); // Lr: (w,h)
        corners.push_back({gt[0] + gt[2] * h, gt[3] + gt[5] * h}); // Ll: (0,h)
        return corners;
    };

    std::vector<std::pair<double, double>> corners1 = get_corners(gt1, w1, h1);
    std::vector<std::pair<double, double>> corners2 = get_corners(gt2, w2, h2);

    std::vector<double> all_xs, all_ys;
    for (const auto& p : corners1) { all_xs.push_back(p.first); all_ys.push_back(p.second); }
    for (const auto& p : corners2) { all_xs.push_back(p.first); all_ys.push_back(p.second); }

    union_minX = *std::min_element(all_xs.begin(), all_xs.end());
    union_maxX = *std::max_element(all_xs.begin(), all_xs.end());
    union_minY = *std::min_element(all_ys.begin(), all_ys.end());
    union_maxY = *std::max_element(all_ys.begin(), all_ys.end());

    avg_resX = (std::abs(gt1[1]) + std::abs(gt2[1])) / 2.0;
    avg_resY = (std::abs(gt1[5]) + std::abs(gt2[5])) / 2.0;
}

void BatchProcessor::mergeWithOTB(const fs::path& ortho_path) {
    std::lock_guard<std::mutex> lock(mosaicMutex_);

    auto now   = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ts_ss;
    ts_ss << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
    std::string timestamp = ts_ss.str();

    fs::path stitched_parent_dir = fs::path(config_.stitchedFile).parent_path();
    fs::create_directories(stitched_parent_dir);
    
    if (!fs::exists(config_.stitchedFile)) {
        try {
            fs::copy_file(ortho_path, config_.stitchedFile, fs::copy_options::overwrite_existing);
            std::cout << "[INFO] Mosaic initialized with: " << config_.stitchedFile << "\n";
            return;
        } catch (const fs::filesystem_error& e) {
            std::cerr << RED << "[ERROR] Failed to initialize mosaic: " << e.what() << RESET << std::endl;
            return;
        }
    }

    if (config_.savePreviousOrthophoto) {
        fs::path bak_path = stitched_parent_dir / ("prev_mosaic_" + timestamp + ".tif");
        try {
            fs::copy_file(config_.stitchedFile, bak_path, fs::copy_options::overwrite_existing);
            std::cout << "[DEBUG] Previous mosaic backed up to: " << bak_path << std::endl;
        } catch (const fs::filesystem_error& e) {
            std::cerr << RED << "[WARN] Failed to backup previous mosaic: " << e.what() << RESET << std::endl;
        }
    }

    double mosaic_gt[6];
    std::optional<std::string> mosaic_proj_wkt_opt;
    int mosaic_width, mosaic_height;
    if (!getRasterInfo(config_.stitchedFile, mosaic_gt, mosaic_proj_wkt_opt, mosaic_width, mosaic_height)) {
        std::cerr << RED << "[ERROR] Could not get info for existing mosaic: " << config_.stitchedFile << RESET << std::endl;
        return;
    }

    double new_ortho_gt[6];
    std::optional<std::string> new_ortho_proj_wkt_opt;
    int new_ortho_width, new_ortho_height;
    if (!getRasterInfo(ortho_path, new_ortho_gt, new_ortho_proj_wkt_opt, new_ortho_width, new_ortho_height)) {
        std::cerr << RED << "[ERROR] Could not get info for new orthophoto: " << ortho_path << RESET << std::endl;
        return;
    }

    if (!mosaic_proj_wkt_opt.has_value() || mosaic_proj_wkt_opt->empty()) {
        std::cerr << RED << "[ERROR] Mosaic has no projection, cannot proceed with gdalwarp for growth." << RESET << "\n";
        return;
    }

    double union_minX, union_maxY, union_maxX, union_minY;
    double avg_resX, avg_resY;
    calculateUnionExtent(
        mosaic_gt, mosaic_width, mosaic_height,
        new_ortho_gt, new_ortho_width, new_ortho_height,
        union_minX, union_maxY, union_maxX, union_minY,
        avg_resX, avg_resY
    );

    fs::path mosaic_wkt_tmp_file_path = stitched_parent_dir / ("mosaic_proj_" + timestamp + ".wkt");
    TemporaryPath mosaic_wkt_tmp_file(mosaic_wkt_tmp_file_path);
    {
        std::ofstream wkt_file(mosaic_wkt_tmp_file.get_path());
        if (!wkt_file.is_open()) {
            std::cerr << RED << "[ERROR] Could not create temporary WKT file: " << mosaic_wkt_tmp_file.get_path() << RESET << std::endl;
            return;
        }
        wkt_file << *mosaic_proj_wkt_opt;
        wkt_file.close();
    }

    fs::path new_ortho_warped_expanded_path = stitched_parent_dir / ("new_ortho_warped_expanded_" + timestamp + ".tif");
    TemporaryPath new_ortho_warped_expanded(new_ortho_warped_expanded_path);
    {
        std::stringstream gdalwarp_cmd;
        gdalwarp_cmd << "gdalwarp "
                     << "-t_srs \"" << mosaic_wkt_tmp_file.get_path().string() << "\" "
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
                     << "\"" << new_ortho_warped_expanded.get_path().string() << "\" ";

        if (runCommand(gdalwarp_cmd.str()) != 0) {
            std::cerr << RED << "[ERROR] gdalwarp failed to align and expand new orthophoto. Cannot grow mosaic." << RESET << "\n";
            return;
        }
        std::cout << "[INFO] New orthophoto warped and expanded to: " << new_ortho_warped_expanded.get_path() << std::endl;
    }

    fs::path tmp_mosaic_unoptimized_path = stitched_parent_dir / ("tmp_mosaic_unoptimized_" + timestamp + ".tif");
    TemporaryPath tmp_mosaic_unoptimized(tmp_mosaic_unoptimized_path);
    {
        std::stringstream mosaic_cmd;
        // Use the wrapper so that OTB libraries are isolated
        mosaic_cmd << "otbrun.sh otbcli_Mosaic -il "
                   << "\"" << config_.stitchedFile << "\" "
                   << "\"" << new_ortho_warped_expanded.get_path().string() << "\" "
                   << "-comp.feather slim "
                   << "-comp.feather.slim.length 10 "
                   << "-comp.feather.slim.exponent 1.0 "
                   << "-harmo.method band "
                   << "-harmo.cost rmse "
                   << "-interpolator bco "
                   << "-interpolator.bco.radius 2 "
                   << "-out \"" << tmp_mosaic_unoptimized.get_path().string() << "\" uint8 ";

        std::cout << "[DEBUG] OTB Mosaic Command: " << mosaic_cmd.str() << std::endl;
        if (runCommand(mosaic_cmd.str()) != 0) {
            std::cerr << RED << "[ERROR] otbcli_Mosaic failed. Command: " << mosaic_cmd.str() << RESET << std::endl;
            return;
        }
    }

    fs::path tmp_mosaic_optimized_path = stitched_parent_dir / ("tmp_mosaic_optimized_" + timestamp + ".tif");
    TemporaryPath tmp_mosaic_optimized(tmp_mosaic_optimized_path);
    {
        std::stringstream gdal_translate_cmd;
        gdal_translate_cmd << "gdal_translate "
                 << "\"" << tmp_mosaic_unoptimized.get_path().string() << "\" "
                 << "\"" << tmp_mosaic_optimized.get_path().string() << "\" "
                 << "-co \"TILED=YES\" ";

        if (config_.compress) gdal_translate_cmd << "-co \"COMPRESS=DEFLATE\" ";
        if (config_.useBigTIFF) gdal_translate_cmd << "-co \"BIGTIFF=YES\" ";
        gdal_translate_cmd << "-co \"BLOCKXSIZE=" << config_.blockSize << "\" "
                 << "-co \"BLOCKYSIZE=" << config_.blockSize << "\" ";

        if (runCommand(gdal_translate_cmd.str()) != 0) {
            std::cerr << RED << "[ERROR] gdal_translate failed to optimize output mosaic. Proceeding with unoptimized file." << RESET << "\n";
            try { // If optimization fails, try to use the unoptimized file
                fs::rename(tmp_mosaic_unoptimized.release(), config_.stitchedFile);
                std::cout << "[INFO] Successfully updated stitched orthomosaic (unoptimized) at: " << config_.stitchedFile << std::endl;
            } catch (const fs::filesystem_error& e) {
                std::cerr << RED << "[ERROR] Failed to replace stitched file with unoptimized version: " << e.what() << RESET << std::endl;
            }
            return;
        }
    }

    try {
        fs::rename(tmp_mosaic_optimized.release(), config_.stitchedFile);
        std::cout << GREEN << "[INFO] Successfully updated stitched orthomosaic at: " << config_.stitchedFile << RESET << std::endl;
    } catch (const fs::filesystem_error& e) {
        std::cerr << RED <<"[ERROR] Failed to replace stitched file: " << e.what() << RESET << std::endl;
        return;
    }
}

bool BatchProcessor::runOdmBatchSuccessful(const fs::path& batch_path, const fs::path& ortho_path) {
    std::size_t retryCount = 0;
    std::size_t effectiveRetries = config_.retry ? config_.retries : 1;

    while (retryCount < effectiveRetries && !stopSignal_) {
        std::cout << "[INFO] Processing batch (attempt #" << retryCount + 1 << " of " << effectiveRetries << "): " << batch_path << std::endl;

        if (stopSignal_.load()) break;

        OdmRunResult result = OdmRunResult::CommandFailed;

        if (runOdmBatchInternal(batch_path)) {
            if (!fs::exists(ortho_path)) {
                result = OdmRunResult::OrthophotoNotFound;
                std::cerr << RED << "[ERROR] Orthophoto not found after ODM run: " << ortho_path << RESET << std::endl;
            } else {
                try {
                    auto file_size = fs::file_size(ortho_path);
                    if (file_size == 0) {
                        result = OdmRunResult::OrthophotoZeroBytes;
                        std::cerr << RED << "[ERROR] Orthophoto is 0 bytes: " << ortho_path << RESET << std::endl;
                    } else {
                        if (!validateGeotiff(ortho_path)) {
                            result = OdmRunResult::ValidationFailed;
                            std::cerr << RED << "[ERROR] Orthophoto failed GDAL validation: " << ortho_path << RESET << std::endl;
                        } else {
                            result = OdmRunResult::Success;
                        }
                    }
                } catch (const fs::filesystem_error& e) {
                    result = OdmRunResult::OrthophotoNotFound;
                    std::cerr << RED << "[ERROR] Failed to get file size for " << ortho_path << ": " << e.what() << RESET << std::endl;
                }
            }
        } else {
            // ODM command failed already handled by runOdmBatchInternal return
        }

        if (result == OdmRunResult::Success) {
            std::cout << GREEN << "[INFO] Orthophoto passed validation: " << ortho_path << RESET << std::endl;
            return true;
        } else {
            std::string err_message;
            switch (result) {
                case OdmRunResult::OrthophotoNotFound:
                    err_message = "[ERROR] Orthophoto not found after ODM run. Retrying...";
                    break;
                case OdmRunResult::OrthophotoZeroBytes:
                    err_message = "[ERROR] Orthophoto is 0 bytes. Deleting and retrying...";
                    break;
                case OdmRunResult::ValidationFailed:
                    err_message = "[ERROR] Orthophoto failed GDAL validation. Deleting and retrying...";
                    break;
                case OdmRunResult::CommandFailed:
                    err_message = "[ERROR] ODM command failed for batch: " + batch_path.string() + ". Retrying...";
                    break;
                default:
                    err_message = "[ERROR] Unhandled ODM processing error. Retrying...";
                    break;
            }
            
            try {
                if (fs::exists(batch_path / "odm_orthophoto")) {
                    fs::remove_all(batch_path / "odm_orthophoto");
                    std::cout << "[DEBUG] Cleaned partial ODM output for retry: " << (batch_path / "odm_orthophoto") << std::endl;
                }
            } catch (const fs::filesystem_error& e) {
                std::cerr << RED << "[ERROR] Failed to clean partial ODM output for retry: " << e.what() << RESET << std::endl;
            }

            ++retryCount;
            if (stopSignal_) return false;
            if (retryCount < effectiveRetries) {
                 std::this_thread::sleep_for(std::chrono::seconds(config_.retryTimeoutSec));
            }
        }
    }
    return false;
}