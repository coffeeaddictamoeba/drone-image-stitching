#include "../include/batchproc.h"
#include "helpers.h"

#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>

#ifndef _WIN32
    #include <sys/types.h>
    #include <sys/wait.h>
    #include <unistd.h>
    #include <signal.h>
#endif

#include <thread>
#include <fstream>
#include <sstream>

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

// add checking image status by this later to avoid processing the same image twice
void BatchProcessor::setProcessedStatusToImages(const fs::path& imagesDir) {
    for (const auto& img : fs::directory_iterator(imagesDir)) {
        try {
            std::string cmd = "exiftool -overwrite_original -" + config_.exifTagProcessed + "=\"Processed\" " + img.path().string() + " >/dev/null 2>&1 ";
            runCommand(cmd);
        } catch (const std::exception& e) {
            fprintf(
                stderr,
                RED "[ERROR] Failed to tag image %s: %s \r\n" RESET, img.path().c_str(), e.what()
            );
        }
    }
}

void BatchProcessor::processBatchesLoop() {
    while (!stopSignal_.load()) {
        std::unique_lock<std::mutex> lock(queueMutex_);

        queueCV_.wait(lock, [this] { 
            return !batchQueue_.empty() || stopSignal_.load(); 
        });

        if (stopSignal_.load()) {
            while (!batchQueue_.empty()) 
                batchQueue_.pop();
            break;
        }

        if (batchQueue_.empty()) continue;

        BatchTask task = batchQueue_.front();
        batchQueue_.pop();
        lock.unlock();

        if (stopSignal_.load()) break;

        fs::path batchPath = createBatchDirectory(task.images, task.batch_id);
        fs::path orthoPath = batchPath / "odm_orthophoto" / "odm_orthophoto.tif";

        bool isBatchSuccessful = runBatchSuccessful(batchPath, orthoPath);
        if (isBatchSuccessful) merge(orthoPath);
        else {
            fprintf(
                stderr,
                RED "[ERROR] Batch failed after %zu retries: %s \r\n" RESET, config_.retries, batchPath.c_str()
            );
        }

        // cleanup
        for (const auto& entry : fs::directory_iterator(batchPath)) {
            if (entry.path().filename() == "images") {
                if (isBatchSuccessful) { // assign "processed" tag to images only after successful processing
                    setProcessedStatusToImages(entry.path());
                }
                continue;
            }

            try {
                fs::remove_all(entry.path());
            } catch (const fs::filesystem_error& e) {
                fprintf(
                    stderr,
                    RED "[ERROR] Failed to remove %s: %s \r\n" RESET, entry.path().c_str(), e.what()
                );
            }
        }
    }

    fprintf(stdout, "[INFO] BatchProcessor loop stopped. \r\n");
}

fs::path BatchProcessor::createBatchDirectory(const std::vector<fs::path>& images, int batch_id) {
    std::stringstream ss;
    ss << "batch_" << std::setw(3) << std::setfill('0') << batch_id;

    fs::path batchRootPath = fs::path(config_.batchDir) / ss.str();
    fs::path batchImagesPath = batchRootPath / "images";
    fs::create_directories(batchImagesPath);
    
    for (const auto& img : images) {
        try {
            fs::rename(img, batchImagesPath / img.filename());
        } catch (const fs::filesystem_error& e) {
            fprintf(
                stderr,
                RED "[ERROR] Failed to move image %s to batch %s: %s \r\n" RESET, img.c_str(), batchImagesPath.c_str(), e.what()
            );
        }
    }

    return batchRootPath;
}

int BatchProcessor::runCommand(const std::string& cmd) {
    if (stopSignal_.load()) return 1;

    #ifndef _WIN32
        pid_t pid = fork();
        if (pid == 0) { // Child
            execl("/bin/sh", "sh", "-c", cmd.c_str(), nullptr);
            _exit(127);
        }

        int status = 0;
        while (true) {
            pid_t w = waitpid(pid, &status, WNOHANG);

            if (w == pid) break; // process finished

            if (stopSignal_.load()) {
                fprintf(
                    stdout,
                    "[INFO] Stopping command: %s \r\n" RESET, cmd.c_str()
                );

                // try graceful stop first
                kill(pid, SIGTERM);
                std::this_thread::sleep_for(std::chrono::seconds(1));

                // force kill if still alive
                kill(pid, SIGKILL);
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

bool BatchProcessor::runBatch(const fs::path& batchPath) {
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
                    "--fast-orthophoto "
                    "--skip-3dmodel ";
    #endif
    return runCommand(cmd) == 0;
}

void BatchProcessor::initMosaic(const fs::path& orthoPath) {
    try {
        fs::create_directories(config_.stitchedDir);
        fs::copy_file(
            orthoPath, 
            config_.stitchedFile, 
            fs::copy_options::overwrite_existing
        );
        fprintf(stdout, "[INFO] Mosaic initialized with: %s \r\n", config_.stitchedFile.c_str());
    } catch (const fs::filesystem_error& e) {
        fprintf(
            stderr,
            RED "[ERROR] Failed to initialize mosaic: %s \r\n" RESET, e.what()
        );
        return;
    }
}

void BatchProcessor::savePreviousOrthophoto() {
    std::string previousMosaicPath = config_.stitchedDir + '/' + ("prev_mosaic_" + getTimestamp() + ".tif");
    try {
        fs::copy_file(
            config_.stitchedFile, 
            previousMosaicPath, 
            fs::copy_options::overwrite_existing
        );
        fprintf(stdout, "[INFO] Previous mosaic saved to: %s \r\n", previousMosaicPath.c_str());
    } catch (const fs::filesystem_error& e) {
        fprintf(
            stderr,
            RED "[ERROR] Failed to backup previous mosaic: %s \r\n" RESET, e.what()
        );
    }
}

std::optional<OrthoMosaic> BatchProcessor::fillOrthoMosaic(const fs::path& orthoPath) {
    OrthoMosaic mosaic;
    mosaic.orthoPath = orthoPath;

    if (!getRasterInfo(
        orthoPath.c_str(), 
        mosaic.geoTransform, 
        mosaic.mosaicProjWkt, 
        mosaic.width, 
        mosaic.height)) {
        fprintf(
            stderr,
            RED "[ERROR] Could not get info for mosaic: %s \r\n" RESET, config_.stitchedFile.c_str()
        );
        return std::nullopt;
    }

    if (!mosaic.mosaicProjWkt.has_value() || mosaic.mosaicProjWkt->empty()) {
        fprintf(
            stderr,
            RED "[ERROR] Mosaic has no projection, cannot proceed with gdalwarp for growth. \r\n" RESET
        );
        return std::nullopt;
    }

    return mosaic;
}

UnionExtent BatchProcessor::getUnionExtent(OrthoMosaic& oldm, OrthoMosaic& newm) {
    UnionExtent u;
    calculateUnionExtent(
        oldm.geoTransform, oldm.width, oldm.height,
        newm.geoTransform, newm.width, newm.height,
        u.unionMinX, u.unionMaxX, 
        u.unionMinY, u.unionMaxY,
        u.avgResX, u.avgResY
    );
    return u;
}

std::optional<TemporaryPath> BatchProcessor::writeMosaicWKT(const OrthoMosaic& old) {
    const fs::path mosaicWktTempPath = config_.stitchedDir + '/' + "mosaic_proj.wkt";
    TemporaryPath mosaicWktTempFile(mosaicWktTempPath);
    {
        std::ofstream wktfile(mosaicWktTempPath);
        if (!wktfile.is_open()) {
            fprintf(
                stderr,
                RED "[ERROR] Could not create temporary WKT file: %s \r\n" RESET, mosaicWktTempPath.c_str()
            );
            return std::nullopt;
        }
        wktfile << *old.mosaicProjWkt;
        wktfile.close();
    }
    return mosaicWktTempFile;
}

TemporaryPath BatchProcessor::initMosaicExpanded() {
    const fs::path expandedPath = config_.stitchedDir + '/' + "new_ortho_warped_expanded.tif";
    TemporaryPath expanded(expandedPath);
    return expanded;
}

TemporaryPath BatchProcessor::initMosaicRealigned() {
    fs::path realignedPath = config_.stitchedDir + '/' + "tmp_mosaic_unoptimized.tif";
    TemporaryPath realigned(realignedPath);
    return realigned;
}

TemporaryPath BatchProcessor::initMosaicOptimized() {
    fs::path optimizedPath = config_.stitchedDir + '/' + "tmp_mosaic_optimized.tif";
    TemporaryPath optimized(optimizedPath);
    return optimized;
}

void BatchProcessor::expandMosaic(TemporaryPath& tempWktFile, TemporaryPath& tempWarpedExpanded, const OrthoMosaic &oldm, const OrthoMosaic &newm, const UnionExtent& u) {
    std::stringstream cmd;
    cmd << "gdalwarp "
        << "-t_srs \"" << tempWktFile.get_path().string() << "\" "
        << "-te " << std::fixed << std::setprecision(10) << u.unionMinX << " "
        << std::fixed << std::setprecision(10) << u.unionMinY << " "
        << std::fixed << std::setprecision(10) << u.unionMaxX << " "
        << std::fixed << std::setprecision(10) << u.unionMaxY << " "
        << "-tr " << std::fixed << std::setprecision(10) << u.avgResX << " "
        << std::fixed << std::setprecision(10) << u.avgResY << " "
        << "-r bilinear "
        << "-dstalpha "
        << "-srcnodata 0 "
        << "-dstnodata 0 "
        << "\"" << newm.orthoPath.string() << "\" "
        << "\"" << tempWarpedExpanded.get_path().string() << "\" ";

    if (runCommand(cmd.str()) != 0) {
        fprintf(
            stderr,
            RED "[ERROR] gdalwarp failed to align and expand new orthophoto. Cannot grow mosaic. \r\n" RESET
        );
        return;
    }

    fprintf(stdout, "[INFO] New orthophoto warped and expanded to: %s \r\n" RESET, tempWarpedExpanded.get_path().c_str());
}

void BatchProcessor::mergeMosaic(TemporaryPath& expanded, TemporaryPath& realigned) {
    std::stringstream cmd;
    cmd << "otbcli_Mosaic -il "
        << "\"" << config_.stitchedFile << "\" "
        << "\"" << expanded.get_path().string() << "\" "
        << "-comp.feather slim "
        << "-comp.feather.slim.length 10 "
        << "-comp.feather.slim.exponent 1.0 "
        << "-harmo.method band "
        << "-harmo.cost rmse "
        << "-interpolator bco "
        << "-interpolator.bco.radius 2 "
        << "-out \"" << realigned.get_path().string() << "\" uint8 ";

    if (runCommand(cmd.str()) != 0) {
        fprintf(
            stderr,
            RED "[ERROR] otbcli_Mosaic failed. Command: %s \r\n" RESET, cmd.str().c_str()
        );
        return;
    }
}

void BatchProcessor::optimizeMosaic(TemporaryPath& unoptimized, TemporaryPath& optimized) {
    std::stringstream cmd;
    cmd << "gdal_translate "
        << "\"" << unoptimized.get_path().string() << "\" "
        << "\"" << optimized.get_path().string() << "\" "
        << "-co \"TILED=YES\" ";

    if (config_.compress) cmd << "-co \"COMPRESS=DEFLATE\" ";
    if (config_.useBigTIFF) cmd << "-co \"BIGTIFF=YES\" ";

    cmd << "-co \"BLOCKXSIZE=" << config_.blockSize << "\" "
        << "-co \"BLOCKYSIZE=" << config_.blockSize << "\" ";

    if (runCommand(cmd.str()) != 0) { // If optimization fails, try to use the unoptimized file
        fprintf(
            stderr,
            RED "[ERROR] gdal_translate failed to optimize output mosaic. Proceeding with unoptimized file. \r\n" RESET
        );
        try {
            fs::rename(unoptimized.release(), config_.stitchedFile);
            fprintf(
                stdout,
                "[INFO] Successfully updated stitched orthomosaic (unoptimized) at: %s \r\n", config_.stitchedFile.c_str()
            );
        } catch (const fs::filesystem_error& e) {
            fprintf(
                stderr,
                RED "[ERROR] Failed to replace stitched file with unoptimized version: %s \r\n" RESET, e.what()
            );
        }
        return;
    }

    try {
        fs::rename(optimized.release(), config_.stitchedFile);
        fprintf(
            stdout,
            GREEN "[INFO] Successfully updated stitched orthomosaic at: %s \r\n" RESET, config_.stitchedFile.c_str()
        );
    } catch (const fs::filesystem_error& e) {
        fprintf(
            stderr,
            RED "[ERROR] Failed to replace stitched file: %s \r\n" RESET, e.what()
        );
        return;
    }
}

void BatchProcessor::merge(const fs::path& newOrthoPath) {
    std::lock_guard<std::mutex> lock(mosaicMutex_);

    // Initialize mosaic
    if (!fs::exists(config_.stitchedFile)) {
        initMosaic(newOrthoPath);
        return;
    }

    if (config_.savePreviousOrthophoto) savePreviousOrthophoto();

    std::optional<OrthoMosaic> stitched = fillOrthoMosaic(config_.stitchedFile);
    std::optional<OrthoMosaic> newOrtho = fillOrthoMosaic(newOrthoPath);

    // Calculate union extent
    UnionExtent u = getUnionExtent(stitched.value(), newOrtho.value());

    // Expand mosaic using gdalwarp
    std::optional<TemporaryPath> tempWKT = writeMosaicWKT(stitched.value());       // Write mosaic WKT
    TemporaryPath expanded = initMosaicExpanded();                                     // Warp new orthophoto to mosaic CRS & bounds
    expandMosaic(tempWKT.value(), expanded, stitched.value(), newOrtho.value(), u);

    // Blend realigned image with mosaic
    TemporaryPath realigned = initMosaicRealigned();
    mergeMosaic(expanded, realigned);

    // Optimize final mosaic
    TemporaryPath optimized = initMosaicOptimized();
    optimizeMosaic(realigned, optimized);
}

bool BatchProcessor::runBatchSuccessful(const fs::path& batchPath, const fs::path& orthoPath) {
    int retryCount = 0;
    int effectiveRetries = config_.retry ? config_.retries : 1;

    while (retryCount <= effectiveRetries && !stopSignal_) {
        fprintf(stdout, "[INFO] Processing batch (attempt #%d of %d): %s \r\n", retryCount+1, effectiveRetries+1, batchPath.c_str());

        if (stopSignal_.load()) break;

        OdmRunResult result = OdmRunResult::CommandFailed;

        if (runBatch(batchPath)) {
            const char* ortho = orthoPath.c_str();
            if (!fs::exists(orthoPath)) {
                result = OdmRunResult::OrthophotoNotFound;
                fprintf(
                    stderr,
                    RED "[ERROR] Orthophoto not found after ODM run: %s \r\n" RESET, ortho
                );
            } else {
                try {
                    auto fsize = fs::file_size(orthoPath);
                    if (fsize == 0) {
                        result = OdmRunResult::OrthophotoZeroBytes;
                        fprintf(
                            stderr,
                            RED "[ERROR] Orthophoto is 0 bytes: %s \r\n" RESET, ortho
                        );
                    } else {
                        if (!validateGeotiff(ortho)) {
                            result = OdmRunResult::ValidationFailed;
                            fprintf(
                                stderr,
                                RED "[ERROR] Orthophoto failed GDAL validation: %s \r\n" RESET, ortho
                            );
                        } else {
                            result = OdmRunResult::Success;
                        }
                    }
                } catch (const fs::filesystem_error& e) {
                    result = OdmRunResult::OrthophotoNotFound;
                    fprintf(
                        stderr,
                        RED "[ERROR] Failed to get file size for %s: %s \r\n" RESET, ortho, e.what()
                    );
                }
            }
        } else {
            // ODM command failed already handled by runBatch return
        }

        if (result == OdmRunResult::Success) {
            fprintf(stdout, GREEN "[INFO] Orthophoto passed validation: %s \r\n" RESET, orthoPath.c_str());
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
            
            // cleanup
            try {
                if (fs::exists(batchPath / "odm_orthophoto")) {
                    for (const auto& entry : fs::directory_iterator(batchPath)) {
                        if (entry.path().filename() == "images") continue;
                        try {
                            fs::remove_all(entry.path());
                        } catch (const fs::filesystem_error& e) {
                            fprintf(
                                stderr,
                                RED "[ERROR] Failed to remove %s: %s \r\n" RESET, entry.path().c_str(), e.what()
                            );
                        }
                    }
                }
            } catch (const fs::filesystem_error& e) {
                fprintf(
                    stderr,
                    RED "[ERROR] Failed to clean ODM output for retry: %s \r\n" RESET, e.what()
                );
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