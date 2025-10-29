#include "../include/batchproc.h"

// use opencv to validate more accurately?
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>

#define RESET   "\033[0m"
#define RED     "\033[31m"      // Errors
#define YELLOW  "\033[33m"      // Warnings
#define GREEN   "\033[32m"      // Success

bool BatchProcessor::validateGeotiff(const char* path) {
    GDALDatasetRAII dataset((GDALDataset*)GDALOpen(path, GA_ReadOnly));
    if (!dataset) {
        fprintf(
            stderr,
            RED "[ERROR] GDAL failed to open image: %s\r\n" RESET, path
        );
        return false;
    }

    int width = dataset->GetRasterXSize();
    int height = dataset->GetRasterYSize();

    if (width == 0 || height == 0) {
        fprintf(
            stderr,
            RED "[ERROR] Invalid image dimensions: %s\r\n" RESET, path
        );
        return false;
    }

    int bandCount = dataset->GetRasterCount();

    if (bandCount < 3) {
        fprintf(
            stderr,
            RED "[ERROR] Not enough bands (RGB expected): %s\r\n" RESET, path
        );
        return false;
    }

    constexpr int sampleSize = 500;
    int offsetX = std::max(0, width/2 - sampleSize/2);
    int offsetY = std::max(0, height/2 - sampleSize/2);
    int winX = std::min(sampleSize, width - offsetX);
    int winY = std::min(sampleSize, height - offsetY);
    double stddev = 0.0;

    const size_t bufsize = winX * winY;
    float buffer[bufsize];
    for (int i = 1; i <= 3; ++i) { // Get RGB channel
        GDALRasterBand* band = dataset->GetRasterBand(i);
        if (!band || band->RasterIO(GF_Read, 
                                      offsetX, 
                                      offsetY, 
                                      winX, 
                                      winY,
                                      buffer, 
                                      winX, 
                                      winY, 
                                      GDT_Float32,
                                      0, 
                                      0) != CE_None) {
            fprintf(
            stderr,
            RED "[ERROR] Failed to read band %d: %s\r\n" RESET, i, path
            );
            return false;
        }

        // Calculate presence of each color in image
        double sum = 0.0;
        double sqSum = 0.0;
        for (int i = 0; i < bufsize; i++) {
            sum += buffer[i];
            sqSum += buffer[i]*buffer[i];
        }

        double mean = sum / bufsize;
        stddev = std::sqrt((sqSum/buffer[i])-(mean*mean));
    }

    bool rgbValid = stddev >= config_.rgbValidationThreshold;
    if (!rgbValid) {
        fprintf(
        stderr,
        YELLOW "[WARN] Low RGB variance detected: %f: %s\r\n" RESET, stddev, path
        );
    }

    bool alphaValid = true;
    if (bandCount >= 4) {
        GDALRasterBand* alphaBand = dataset->GetRasterBand(4);
        if (!alphaBand || alphaBand->RasterIO(GF_Read, 
                                                offsetX, 
                                                offsetY, 
                                                winX, 
                                                winY,
                                                buffer, 
                                                winX, 
                                                winY,
                                                GDT_Float32,
                                                0, 
                                                0) != CE_None) {
            fprintf(
            stderr,
            RED "[WARN] Could not read alpha band — skipping transparency check." RESET
            );
        } else {
            int visiblePixels = std::count_if(buffer, buffer+(bufsize-1), [](float v) {
                return v > 10.0f; // Check if more than 10% of image is transparent
            });
            double visibleFraction = (double)visiblePixels / bufsize;
            if (visibleFraction < config_.alphaValidationThreshold) {
                alphaValid = false;
            }
        }
    }

    if (!rgbValid && !alphaValid) {
        fprintf(
            stderr,
            RED "[ERROR] Rejected: low RGB variance and mostly transparent:  %s \r\n" RESET, path
        );
        return false;
    }

    if (!alphaValid) {
        fprintf(
            stderr,
            YELLOW "[WARN] Alpha band mostly transparent in central region: %s \r\n" RESET, path
        );
    }

    return true;
}

bool BatchProcessor::getRasterInfo(const char* path, double geoTransform[6], std::optional<std::string>& projWkt, int& width, int& height) {
    GDALDatasetRAII dataset((GDALDataset*)GDALOpen(path, GA_ReadOnly));
    if (!dataset) {
        fprintf(
            stderr, 
            "[ERROR] Could not open raster: %s \r\n" RESET, path
        );
        return false;
    }

    if (dataset->GetGeoTransform(geoTransform) != CE_None) {
        fprintf(
            stderr,
            "[ERROR] Could not get geotransform for: %s \r\n" RESET, path
        );
        return false;
    }

    const char* pszWKT = dataset->GetProjectionRef();
    if (pszWKT == nullptr || std::string(pszWKT).empty()) {
        fprintf(
            stderr,
            "[WARN] No projection found for: %s \r\n" RESET, path
        );
        projWkt = std::nullopt;
    } else {
        projWkt = std::string(pszWKT);
    }

    width = dataset->GetRasterXSize();
    height = dataset->GetRasterYSize();

    return true;
}

void BatchProcessor::calculateUnionExtent(double geoTransform1[6], int width1, int height1, 
                                          double geoTransform2[6], int width2, int height2,
                                          double& unionMinX, double& unionMaxX, double& unionMinY, double& unionMaxY,
                                          double& avgResX, double& avgResY) {
    auto getCorners = [](double geoTransform[6], int width, int height, double corners[4][2]) -> void {
        corners[0][0] = geoTransform[0];
        corners[0][1] = geoTransform[3];

        corners[1][0] = geoTransform[0] + geoTransform[1] * width;
        corners[1][1] = geoTransform[3] + geoTransform[4] * width;

        corners[2][0] = geoTransform[0] + geoTransform[1] * width + geoTransform[2] * height;
        corners[2][1] = geoTransform[3] + geoTransform[4] * width + geoTransform[5] * height;

        corners[3][0] = geoTransform[0] + geoTransform[2] * height;
        corners[3][1] = geoTransform[3] + geoTransform[5] * height;
    };

    double corners1[4][2];
    double corners2[4][2];
    getCorners(geoTransform1, width1, height1, corners1);
    getCorners(geoTransform2, width2, height2, corners2);

    constexpr int s = 8;
    int cntx = 0;
    int cnty = 0;
    double xs[s];
    double ys[s];
    for (int i = 0; i < 4; ++i) {
        // Xs
        xs[cntx++] = corners1[i][0];
        xs[cntx++] = corners2[i][0];
        // Ys
        ys[cnty++] = corners1[i][1];
        ys[cnty++] = corners2[i][1];
    }

    unionMinX = *std::min_element(xs, xs+s);
    unionMaxX = *std::max_element(xs, xs+s);
    unionMinY = *std::min_element(ys, ys+s);
    unionMaxY = *std::max_element(ys, ys+s);

    avgResX = (std::abs(geoTransform1[1]) + std::abs(geoTransform2[1])) / 2.0;
    avgResY = (std::abs(geoTransform1[5]) + std::abs(geoTransform2[5])) / 2.0;
}