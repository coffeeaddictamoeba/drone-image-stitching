#include "../include/batchproc.h"
#include <string>

#ifndef _WIN32
    #include <sys/types.h>
    #include <sys/wait.h>
    #include <unistd.h>
    #include <signal.h>
#endif

#include <thread>
#include <fstream>

#define RESET   "\033[0m"
#define RED     "\033[31m"      // Errors
#define YELLOW  "\033[33m"      // Warnings
#define GREEN   "\033[32m"      // Success

BatchProcessor::BatchProcessor(const Config& config, std::queue<BatchTask>& batchQueue, std::mutex& queueMutex, std::condition_variable& queueCV, std::atomic<bool>& stopSignal)
    : config_(config), 
    batchQueue_(batchQueue), 
    queueMutex_(queueMutex), 
    queueCV_(queueCV), 
    stopSignal_(stopSignal) {
    fs::create_directories(config_.batchDir);
    fs::create_directories(config_.stitchedDir);
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

        fs::path batchPath = createBatchDirectory(task.images, task.batch_id);
        fs::path orthoPath = batchPath / "odm_orthophoto" / "odm_orthophoto.tif";

        if (runOdmBatchSuccessful(batchPath, orthoPath)) {
            mergeWithOTB(orthoPath);
        } else {
            std::cerr << RED << "[FATAL] Batch failed after " << config_.retries << " retries: " << batchPath << RESET << std::endl;
        }

        // cleanup
        for (const auto& entry : fs::directory_iterator(batchPath)) {
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

    fs::path batchRootPath = fs::path(config_.batchDir) / ss.str();
    fs::path batchImagesPath = batchRootPath / "images";
    fs::create_directories(batchImagesPath);
    
    for (const auto& img : images) {
        try {
            fs::rename(img, batchImagesPath / img.filename()); // + add/check metadata tag of being processed?
        } catch (const fs::filesystem_error& e) {
            std::cerr << RED << "[ERROR] Failed to move image " << img << " to batch " << batchImagesPath << ": " << e.what() << RESET << std::endl;
        }
    }

    return batchRootPath;
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

bool BatchProcessor::runOdmBatchInternal(const fs::path& batchPath) {
    #ifdef _WIN32 // probably will be removed
        std::string absoluteBatchPath = fs::absolute(batchPath).string();

        std::string dockerHostPath = absoluteBatchPath;
        std::replace(dockerHostPath.begin(), dockerHostPath.end(), '\\', '/');

        std::string cmd = "docker run --rm "
                        "-v \"" + dockerHostPath + ":/datasets/project\" "
                        "-w /datasets/project "
                        "opendronemap/odm "
                        "--project-path /datasets project "
                        "--fast-orthophoto --skip-3dmodel";
    #else
        std::string uid = std::to_string(getuid());
        std::string gid = std::to_string(getgid());

        std::string absoluteBatchPath = fs::absolute(batchPath).string();

        std::string cmd = "docker run --rm "
                    "--user " + uid + ":" + gid + " "
                    "-v \"" + absoluteBatchPath + ":/datasets/project\" "
                    "-w /datasets/project "
                    "opendronemap/odm "
                    "--project-path /datasets project "
                    "--fast-orthophoto --skip-3dmodel";
    #endif
    return runCommand(cmd) == 0;
}

void BatchProcessor::savePreviousOrthophoto(std::string &timestamp) {
    fs::path previousMosaicPath = config_.stitchedDir + '/' + ("prev_mosaic_" + timestamp + ".tif");
    try {
        fs::copy_file(config_.stitchedFile, previousMosaicPath, fs::copy_options::overwrite_existing);
        std::cout << "[DEBUG] Previous mosaic saved to: " << previousMosaicPath << std::endl;
    } catch (const fs::filesystem_error& e) {
        std::cerr << RED << "[WARN] Failed to backup previous mosaic: " << e.what() << RESET << std::endl;
    }
}

void BatchProcessor::mergeWithOTB(const fs::path& orthoPath) {
    std::lock_guard<std::mutex> lock(mosaicMutex_);

    // Set timestamp
    std::stringstream tss;
    auto now   = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    tss << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");
    std::string timestamp = tss.str();

    fs::create_directories(config_.stitchedDir);
    
    // Initialize mosaic
    if (!fs::exists(config_.stitchedFile)) {
        try {
            fs::copy_file(orthoPath, config_.stitchedFile, fs::copy_options::overwrite_existing);
            std::cout << "[INFO] Mosaic initialized with: " << config_.stitchedFile << "\n";
            return;
        } catch (const fs::filesystem_error& e) {
            std::cerr << RED << "[ERROR] Failed to initialize mosaic: " << e.what() << RESET << std::endl;
            return;
        }
    }

    if (config_.savePreviousOrthophoto) savePreviousOrthophoto(timestamp);

    double mosaicGeoTransform[6];
    int mosaicWidth, mosaicHeight;
    std::optional<std::string> mosaicProjWkt;
    if (!getRasterInfo(config_.stitchedFile, mosaicGeoTransform, mosaicProjWkt, mosaicWidth, mosaicHeight)) {
        std::cerr << RED << "[ERROR] Could not get info for existing mosaic: " << config_.stitchedFile << RESET << std::endl;
        return;
    }

    if (!mosaicProjWkt.has_value() || mosaicProjWkt->empty()) {
        std::cerr << RED << "[ERROR] Mosaic has no projection, cannot proceed with gdalwarp for growth." << RESET << "\n";
        return;
    }

    double newMosaicGeoTransform[6];
    int newMosaicWidth, newMosaicHeight;
    std::optional<std::string> newMosaicProjWkt;
    if (!getRasterInfo(orthoPath, newMosaicGeoTransform, newMosaicProjWkt, newMosaicWidth, newMosaicHeight)) {
        std::cerr << RED << "[ERROR] Could not get info for new orthophoto: " << orthoPath << RESET << std::endl;
        return;
    }

    double unionMinX, unionMaxX, unionMinY, unionMaxY;
    double avgResX, avgResY;
    calculateUnionExtent(
        mosaicGeoTransform, mosaicWidth, mosaicHeight,
        newMosaicGeoTransform, newMosaicWidth, newMosaicHeight,
        unionMinX, unionMaxX,  
        unionMinY, unionMaxY,
        avgResX, avgResY
    );

    fs::path mosaicWktTempPath = config_.stitchedDir + '/' + ("mosaic_proj_" + timestamp + ".wkt");
    TemporaryPath mosaicWktTempFile(mosaicWktTempPath);
    {
        std::ofstream wktfile(mosaicWktTempFile.get_path());
        if (!wktfile.is_open()) {
            std::cerr << RED << "[ERROR] Could not create temporary WKT file: " << mosaicWktTempFile.get_path() << RESET << std::endl;
            return;
        }
        wktfile << *mosaicProjWkt;
        wktfile.close();
    }

    fs::path newMosaicWarpedExpandedPath = config_.stitchedDir + '/' + ("new_ortho_warped_expanded_" + timestamp + ".tif");
    TemporaryPath newMosaicWarpedExpandedFile(newMosaicWarpedExpandedPath);
    {
        std::stringstream gdalwarp_cmd;
        gdalwarp_cmd << "gdalwarp "
                     << "-t_srs \"" << mosaicWktTempFile.get_path().string() << "\" "
                     << "-te " << std::fixed << std::setprecision(10) << unionMinX << " "
                     << std::fixed << std::setprecision(10) << unionMinY << " "
                     << std::fixed << std::setprecision(10) << unionMaxX << " "
                     << std::fixed << std::setprecision(10) << unionMaxY << " "
                     << "-tr " << std::fixed << std::setprecision(10) << avgResX << " "
                     << std::fixed << std::setprecision(10) << avgResY << " "
                     << "-r bilinear "
                     << "-dstalpha "
                     << "-srcnodata 0 "
                     << "-dstnodata 0 "
                     << "\"" << orthoPath.string() << "\" "
                     << "\"" << newMosaicWarpedExpandedFile.get_path().string() << "\" ";

        if (runCommand(gdalwarp_cmd.str()) != 0) {
            std::cerr << RED << "[ERROR] gdalwarp failed to align and expand new orthophoto. Cannot grow mosaic." << RESET << "\n";
            return;
        }
        std::cout << "[INFO] New orthophoto warped and expanded to: " << newMosaicWarpedExpandedFile.get_path() << std::endl;
    }

    fs::path mosaicTempUnoptimizedPath = config_.stitchedDir + '/' + ("tmp_mosaic_unoptimized_" + timestamp + ".tif");
    TemporaryPath mosaicTempUnoptimizedFile(mosaicTempUnoptimizedPath);
    {
        std::stringstream mosaic_cmd;
        mosaic_cmd << "otbrun.sh otbcli_Mosaic -il " // Use the wrapper so that OTB libraries are isolated
                   << "\"" << config_.stitchedFile << "\" "
                   << "\"" << newMosaicWarpedExpandedFile.get_path().string() << "\" "
                   << "-comp.feather slim "
                   << "-comp.feather.slim.length 10 "
                   << "-comp.feather.slim.exponent 1.0 "
                   << "-harmo.method band "
                   << "-harmo.cost rmse "
                   << "-interpolator bco "
                   << "-interpolator.bco.radius 2 "
                   << "-out \"" << mosaicTempUnoptimizedFile.get_path().string() << "\" uint8 ";

        std::cout << "[DEBUG] OTB Mosaic Command: " << mosaic_cmd.str() << std::endl;
        if (runCommand(mosaic_cmd.str()) != 0) {
            std::cerr << RED << "[ERROR] otbcli_Mosaic failed. Command: " << mosaic_cmd.str() << RESET << std::endl;
            return;
        }
    }

    fs::path mosaicTempOptimizedPath = config_.stitchedDir + '/' + ("tmp_mosaic_optimized_" + timestamp + ".tif");
    TemporaryPath mosaicTempOptimizedFile(mosaicTempOptimizedPath);
    {
        std::stringstream gdaltranslate_cmd;
        gdaltranslate_cmd << "gdal_translate "
                 << "\"" << mosaicTempUnoptimizedFile.get_path().string() << "\" "
                 << "\"" << mosaicTempOptimizedFile.get_path().string() << "\" "
                 << "-co \"TILED=YES\" ";

        if (config_.compress) gdaltranslate_cmd << "-co \"COMPRESS=DEFLATE\" ";
        if (config_.useBigTIFF) gdaltranslate_cmd << "-co \"BIGTIFF=YES\" ";

        gdaltranslate_cmd << "-co \"BLOCKXSIZE=" << config_.blockSize << "\" "
                 << "-co \"BLOCKYSIZE=" << config_.blockSize << "\" ";

        if (runCommand(gdaltranslate_cmd.str()) != 0) {
            std::cerr << RED << "[ERROR] gdal_translate failed to optimize output mosaic. Proceeding with unoptimized file." << RESET << "\n";
            try { // If optimization fails, try to use the unoptimized file
                fs::rename(mosaicTempUnoptimizedFile.release(), config_.stitchedFile);
                std::cout << "[INFO] Successfully updated stitched orthomosaic (unoptimized) at: " << config_.stitchedFile << std::endl;
            } catch (const fs::filesystem_error& e) {
                std::cerr << RED << "[ERROR] Failed to replace stitched file with unoptimized version: " << e.what() << RESET << std::endl;
            }
            return;
        }
    }

    try {
        fs::rename(mosaicTempOptimizedFile.release(), config_.stitchedFile);
        std::cout << GREEN << "[INFO] Successfully updated stitched orthomosaic at: " << config_.stitchedFile << RESET << std::endl;
    } catch (const fs::filesystem_error& e) {
        std::cerr << RED <<"[ERROR] Failed to replace stitched file: " << e.what() << RESET << std::endl;
        return;
    }
}

bool BatchProcessor::runOdmBatchSuccessful(const fs::path& batchPath, const fs::path& orthoPath) {
    std::size_t retryCount = 0;
    std::size_t effectiveRetries = config_.retry ? config_.retries : 1;

    while (retryCount < effectiveRetries && !stopSignal_) {
        std::cout << "[INFO] Processing batch (attempt #" << retryCount + 1 << " of " << effectiveRetries << "): " << batchPath << std::endl;

        if (stopSignal_.load()) break;

        OdmRunResult result = OdmRunResult::CommandFailed;

        if (runOdmBatchInternal(batchPath)) {
            if (!fs::exists(orthoPath)) {
                result = OdmRunResult::OrthophotoNotFound;
                std::cerr << RED << "[ERROR] Orthophoto not found after ODM run: " << orthoPath << RESET << std::endl;
            } else {
                try {
                    auto fsize = fs::file_size(orthoPath);
                    if (fsize == 0) {
                        result = OdmRunResult::OrthophotoZeroBytes;
                        std::cerr << RED << "[ERROR] Orthophoto is 0 bytes: " << orthoPath << RESET << std::endl;
                    } else {
                        if (!validateGeotiff(orthoPath)) {
                            result = OdmRunResult::ValidationFailed;
                            std::cerr << RED << "[ERROR] Orthophoto failed GDAL validation: " << orthoPath << RESET << std::endl;
                        } else {
                            result = OdmRunResult::Success;
                        }
                    }
                } catch (const fs::filesystem_error& e) {
                    result = OdmRunResult::OrthophotoNotFound;
                    std::cerr << RED << "[ERROR] Failed to get file size for " << orthoPath << ": " << e.what() << RESET << std::endl;
                }
            }
        } else {
            // ODM command failed already handled by runOdmBatchInternal return
        }

        if (result == OdmRunResult::Success) {
            std::cout << GREEN << "[INFO] Orthophoto passed validation: " << orthoPath << RESET << std::endl;
            return true;
        } else {
            std::string err;
            switch (result) {
                case OdmRunResult::OrthophotoNotFound:
                    err = "[ERROR] Orthophoto not found after ODM run. Retrying...";
                    break;
                case OdmRunResult::OrthophotoZeroBytes:
                    err = "[ERROR] Orthophoto is 0 bytes. Deleting and retrying...";
                    break;
                case OdmRunResult::ValidationFailed:
                    err = "[ERROR] Orthophoto failed GDAL validation. Deleting and retrying...";
                    break;
                case OdmRunResult::CommandFailed:
                    err = "[ERROR] ODM command failed for batch: " + batchPath.string() + ". Retrying...";
                    break;
                default:
                    err = "[ERROR] Unhandled ODM processing error. Retrying...";
                    break;
            }
            
            try {
                if (fs::exists(batchPath / "odm_orthophoto")) {
                    fs::remove_all(batchPath / "odm_orthophoto");
                    std::cout << "[DEBUG] Cleaned partial ODM output for retry: " << (batchPath / "odm_orthophoto") << std::endl;
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