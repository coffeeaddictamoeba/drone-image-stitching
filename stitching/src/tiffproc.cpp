#include "../include/batchproc.h"

// use opencv to validate more accurately?
#include <algorithm>
#include <cmath>

#define RESET   "\033[0m"
#define RED     "\033[31m"      // Errors
#define YELLOW  "\033[33m"      // Warnings
#define GREEN   "\033[32m"      // Success

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

bool BatchProcessor::getRasterInfo(const fs::path& path, double geoTransform[6], std::optional<std::string>& projWkt, int& width, int& height) {
    GDALDatasetRAII dataset((GDALDataset*)GDALOpen(path.string().c_str(), GA_ReadOnly));
    if (!dataset) {
        std::cerr << RED << "[ERROR] Could not open raster: " << path << RESET << std::endl;
        return false;
    }

    if (dataset->GetGeoTransform(geoTransform) != CE_None) {
        std::cerr << RED << "[ERROR] Could not get geotransform for: " << path << RESET << std::endl;
        return false;
    }

    const char* pszWKT = dataset->GetProjectionRef();
    if (pszWKT == nullptr || std::string(pszWKT).empty()) {
        std::cerr << RED << "[WARNING] No projection found for: " << path << RESET << std::endl;
        projWkt = std::nullopt;
    } else {
        projWkt = std::string(pszWKT); // Directly convert to std::string
    }

    width = dataset->GetRasterXSize();
    height = dataset->GetRasterYSize();

    return true;
}

void BatchProcessor::calculateUnionExtent(double geoTransform1[6], int width1, int height1, 
                                          double geoTransform2[6], int width2, int height2,
                                          double& unionMinX, double& unionMaxX, double& unionMinY, double& unionMaxY,
                                          double& avgResX, double& avgResY) {
    auto getCorners = [](double geoTransform[6], int width, int height) -> std::vector<std::pair<double, double>> {
        std::vector<std::pair<double, double>> corners;
        corners.push_back({geoTransform[0], geoTransform[3]}); // Ul: (0,0)
        corners.push_back({geoTransform[0] + geoTransform[1] * width, geoTransform[3] + geoTransform[4] * width}); // Ur: (w,0)
        corners.push_back({geoTransform[0] + geoTransform[1] * width + geoTransform[2] * height, geoTransform[3] + geoTransform[4] * width + geoTransform[5] * height}); // Lr: (w,h)
        corners.push_back({geoTransform[0] + geoTransform[2] * height, geoTransform[3] + geoTransform[5] * height}); // Ll: (0,h)
        return corners;
    };

    std::vector<std::pair<double, double>> corners1 = getCorners(geoTransform1, width1, height1);
    std::vector<std::pair<double, double>> corners2 = getCorners(geoTransform2, width2, height2);

    std::vector<double> all_xs, all_ys;
    for (const auto& p : corners1) { all_xs.push_back(p.first); all_ys.push_back(p.second); }
    for (const auto& p : corners2) { all_xs.push_back(p.first); all_ys.push_back(p.second); }

    unionMinX = *std::min_element(all_xs.begin(), all_xs.end());
    unionMaxX = *std::max_element(all_xs.begin(), all_xs.end());
    unionMinY = *std::min_element(all_ys.begin(), all_ys.end());
    unionMaxY = *std::max_element(all_ys.begin(), all_ys.end());

    avgResX = (std::abs(geoTransform1[1]) + std::abs(geoTransform2[1])) / 2.0;
    avgResY = (std::abs(geoTransform1[5]) + std::abs(geoTransform2[5])) / 2.0;
}