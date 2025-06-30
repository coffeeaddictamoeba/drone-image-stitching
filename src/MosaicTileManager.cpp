#include "../include/mosaic.h"
#include "../include/fmatch.h"
#include <cmath>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;

TileManager::TileManager(const std::string& outputDir, ExifToolPipe& tool): outputDirectory_(outputDir), exiftool_(tool) {
    std::filesystem::create_directories(outputDirectory_);
}

TileKey TileManager::getTileKeyForPoint(int x, int y) const {
    return { x / TILE_SIZE, y / TILE_SIZE };
}

std::string TileManager::getOutputDirectory() const {
    return outputDirectory_;
}

std::string TileManager::getTilePath(const TileKey& key) const {
    return outputDirectory_ + "/tile_" + std::to_string(key.y) + "_" + std::to_string(key.x) + ".png";
}

void TileManager::loadGlobalMetadata(){
    std::string path = outputDirectory_ + "/" + COORDS_METADATA;
    std::ifstream in(path);
    if (!in) return;

    std::string key;
    int value;
    while (in >> key >> value) {
        if (key == "min_x:") globalMinX_ = value;
        else if (key == "min_y:") globalMinY_ = value;
        else if (key == "max_x:") globalMaxX_ = value;
        else if (key == "max_y:") globalMaxY_ = value;
    }
}

void TileManager::saveGlobalMetadata() const {
    std::string path = outputDirectory_ + "/" + COORDS_METADATA;
    std::ofstream out(path);
    out << "min_x: " << globalMinX_ << "\n"
        << "min_y: " << globalMinY_ << "\n"
        << "max_x: " << globalMaxX_ << "\n"
        << "max_y: " << globalMaxY_ << "\n";
}

void TileManager::updateGlobalBounds(const TileKey& key) {
    globalMinX_ = std::min(globalMinX_, key.x);
    globalMinY_ = std::min(globalMinY_, key.y);
    globalMaxX_ = std::max(globalMaxX_, key.x);
    globalMaxY_ = std::max(globalMaxY_, key.y);
    saveGlobalMetadata();
}

TileKey TileManager::getGlobalCenterTileKey() const {
    return {
        (globalMinX_ + globalMaxX_) / 2,
        (globalMinY_ + globalMaxY_) / 2
    };
}

std::pair<double, double> TileManager::calculateTileGPS(const TileKey& tileKey, const TileKey& centerTile, double centerLat, double centerLon, double gsd) const {
    int dx = tileKey.x - centerTile.x;
    int dy = tileKey.y - centerTile.y;

    double offsetX_m = dx * TILE_SIZE * gsd;
    double offsetY_m = dy * TILE_SIZE * gsd;
    
    double metersPerDegLon = M_PER_DEGREE_LATITUDE * std::cos(centerLat * M_PI / 180.0);

    double lat = centerLat - (offsetY_m / M_PER_DEGREE_LATITUDE);
    double lon = centerLon + (offsetX_m / metersPerDegLon);

    return {lat, lon};
}

cv::Mat TileManager::loadTile(const TileKey& key) const {
    std::string path = getTilePath(key);
    if (fs::exists(path)) {
        cv::Mat tile = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (!tile.empty()) return tile;
    }
    return cv::Mat(TILE_SIZE, TILE_SIZE, CV_8UC4, cv::Scalar(0, 0, 0, 0));
}

void TileManager::assignMetadata(const std::string imagePath, const double lat, const double lon, const double alt, const double flen) const {
    std::ostringstream tagStream;
    tagStream << "-n\n";
    tagStream << "-" << IMG_GPS_LAT << "=" << lat << "\n";
    tagStream << "-" << IMG_GPS_LON << "=" << lon << "\n";
    tagStream << "-" << IMG_GPS_ALT << "=" << alt << "\n";
    tagStream << "-" << IMG_FOCAL_LEN_TAG << "=" << flen << "\n";

    exiftool_.setExifTag(imagePath, tagStream.str());
}

void TileManager::saveTile(const TileKey& key, const cv::Mat& tile, const double lat, const double lon, const std::map<std::string, double>& exif) const {
    std::string path = getTilePath(key);
    cv::imwrite(path, tile);

    assignMetadata(path, lat, lon, exif.at(IMG_GPS_ALT), exif.at(IMG_FOCAL_LEN_TAG));
}

cv::Mat TileManager::warpTileRegion(const cv::Mat& input, const cv::Mat& H, const cv::Rect& tileRect) const {
    // 1. Create an empty Mat for the output tile.
    //    It should have the same type as the input image (e.g., CV_8UC4 for RGBA images).
    cv::Mat tileOutput = cv::Mat::zeros(TILE_SIZE, TILE_SIZE, input.type());

    // 2. Define the transformation matrix that maps directly from the input image's
    //    coordinate system to the *local* (0,0) to (TILE_SIZE, TILE_SIZE) coordinate
    //    system of the `tileOutput` matrix.
    //
    //    The `H` matrix maps points from `input` image pixels to `global mosaic pixels`.
    //    We need to combine `H` with a translation that shifts the `tileRect`'s global
    //    position `(tileRect.x, tileRect.y)` to `(0,0)` in the `tileOutput` matrix.
    //
    //    The combined transformation `M` is:
    //    M = Translation_Matrix * H
    //    where Translation_Matrix shifts coordinates by (-tileRect.x, -tileRect.y).

    cv::Mat translationMatrix = cv::Mat::eye(3, 3, CV_64F); // Identity matrix
    translationMatrix.at<double>(0, 2) = -tileRect.x; // Shift X by -tileRect.x
    translationMatrix.at<double>(1, 2) = -tileRect.y; // Shift Y by -tileRect.y

    // The order of multiplication is crucial: apply the homography first, then the translation.
    // Alternatively, if H maps image -> global, and we want image -> tile_local:
    // tile_local_pt = Translation(-tileRect.x, -tileRect.y) * global_pt
    // global_pt = H * image_pt
    // So, tile_local_pt = Translation(-tileRect.x, -tileRect.y) * H * image_pt
    // Thus, the combined matrix `M_img_to_tile_local` is `translationMatrix * H`.
    cv::Mat M_img_to_tile_local = translationMatrix * H;

    // 3. Apply the perspective warp.
    //    'input' is the source image.
    //    'tileOutput' is the destination for the warped pixels.
    //    'M_img_to_tile_local' is the transformation matrix.
    //    'tileOutput.size()' is the size of the destination.
    //    'INTER_LINEAR' for smooth interpolation.
    //    'BORDER_CONSTANT' with Scalar(0,0,0,0) for transparent borders.
    cv::warpPerspective(input, tileOutput, M_img_to_tile_local, tileOutput.size(),
                        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0,0));

    // 4. Ensure the output has an alpha channel if the input image was 3-channel (BGR).
    //    This is important for proper blending in applyImagePerTile.
    if (tileOutput.channels() == 3) {
        cv::cvtColor(tileOutput, tileOutput, cv::COLOR_BGR2BGRA);
    }

    return tileOutput;
}

cv::Mat TileManager::computeGlobalHomography(const TileKey& localOriginKey, const cv::Mat& localHomography) {
    double offsetX = localOriginKey.x * TILE_SIZE;
    double offsetY = localOriginKey.y * TILE_SIZE;

    cv::Mat offsetMat = (cv::Mat_<double>(3, 3) <<
        1, 0, offsetX,
        0, 1, offsetY,
        0, 0, 1);

    return offsetMat * localHomography;
}

void TileManager::applyImageWarpOnce(const cv::Mat& img, const cv::Mat& homography, double imageLat, double imageLon, double gsd, std::map<std::string, double> exif) {
    std::vector<cv::Point2f> corners = {
        {0, 0},
        {(float)img.cols, 0},
        {(float)img.cols, (float)img.rows},
        {0, (float)img.rows}
    };
    cv::perspectiveTransform(corners, corners, homography);

    float minX = FLT_MAX, minY = FLT_MAX, maxX = -FLT_MAX, maxY = -FLT_MAX;
    for (const auto& p : corners) {
        minX = std::min(minX, p.x);
        minY = std::min(minY, p.y);
        maxX = std::max(maxX, p.x);
        maxY = std::max(maxY, p.y);
    }

    int width = std::ceil(maxX - minX);
    int height = std::ceil(maxY - minY);

    cv::Mat warpCanvas(height, width, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    cv::Mat T = (cv::Mat_<double>(3, 3) <<
        1, 0, -minX,
        0, 1, -minY,
        0, 0, 1);

    cv::Mat fullHomography = T * homography;

    cv::warpPerspective(img, warpCanvas, fullHomography, warpCanvas.size(),
                        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));

    int tx0 = std::floor(minX / TILE_SIZE), ty0 = std::floor(minY / TILE_SIZE);
    int tx1 = std::floor(maxX / TILE_SIZE), ty1 = std::floor(maxY / TILE_SIZE);

    TileKey globalCenterKey = getGlobalCenterTileKey();

    std::string centerPath = getTilePath(globalCenterKey);
    std::vector<std::string> sharedTags = {IMG_GPS_LAT, IMG_GPS_LON, IMG_GPS_ALT, IMG_FOCAL_LEN_TAG, IMG_WIDTH_TAG};
    auto rawExif = exiftool_.getExifTags(centerPath, sharedTags);
    exif = exiftool_.parseExifValuesToNumbers(rawExif);

    double centerLat = exif.at(IMG_GPS_LAT);
    double centerLon = exif.at(IMG_GPS_LON);

    gsd = (3.674 * exif.at(IMG_GPS_ALT)) / (exif.at(IMG_FOCAL_LEN_TAG) * exif.at(IMG_WIDTH_TAG));

    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            TileKey key{tx, ty};
            cv::Rect globalTileRect(tx * TILE_SIZE, ty * TILE_SIZE, TILE_SIZE, TILE_SIZE);
            cv::Rect localTileRect(globalTileRect.x - minX, globalTileRect.y - minY, TILE_SIZE, TILE_SIZE);

            if (localTileRect.x < 0 || localTileRect.y < 0 ||
                localTileRect.x + TILE_SIZE > warpCanvas.cols ||
                localTileRect.y + TILE_SIZE > warpCanvas.rows) {
                continue;
            }

            cv::Mat tile = loadTile(key);
            cv::Mat patch = warpCanvas(localTileRect);

            std::vector<cv::Mat> channels;
            cv::split(patch, channels);
            cv::Mat mask = channels[3] > 0;
            patch.copyTo(tile, mask);

            auto [tileLat, tileLon] = calculateTileGPS(key, globalCenterKey, centerLat, centerLon, gsd);
            saveTile(key, tile, tileLat, tileLon, exif);
            updateGlobalBounds(key);
        }
    }
}

void TileManager::applyImagePerTile(const cv::Mat& img, const cv::Mat& homography, double lat, double lon, double gsd, std::map<std::string, double> exif) {
    std::vector<cv::Point2f> corners = {
        {0, 0},
        {(float)img.cols, 0},
        {(float)img.cols, (float)img.rows},
        {0, (float)img.rows}
    };
    cv::perspectiveTransform(corners, corners, homography);

    // --- ADDED DEBUGGING HERE ---
    std::cout << "DEBUG: Original Image Corners:\n";
    std::cout << "  (0, 0)\n";
    std::cout << "  (" << img.cols << ", 0)\n";
    std::cout << "  (" << img.cols << ", " << img.rows << ")\n";
    std::cout << "  (0, " << img.rows << ")\n";

    std::cout << "DEBUG: Warped Image Corners (Global Mosaic Pixels):\n";
    for (const auto& p : corners) {
        std::cout << "  (" << std::fixed << std::setprecision(10) << p.x << ", "
                  << std::fixed << std::setprecision(10) << p.y << ")\n";
    }
    // --- END ADDED DEBUGGING ---

    float minX = FLT_MAX, minY = FLT_MAX, maxX = -FLT_MAX, maxY = -FLT_MAX;
    for (const auto& p : corners) {
        minX = std::min(minX, p.x);
        minY = std::min(minY, p.y);
        maxX = std::max(maxX, p.x);
        maxY = std::max(maxY, p.y);
    }

    // --- ADDED PRECISION DEBUGGING HERE (Crucial for floor/tile issue) ---
    std::cout << "DEBUG: Warped Image Pixel Bounds (minX, minY, maxX, maxY): "
              << std::fixed << std::setprecision(10) << minX << ", "
              << std::fixed << std::setprecision(10) << minY << ", "
              << std::fixed << std::setprecision(10) << maxX << ", "
              << std::fixed << std::setprecision(10) << maxY << "\n";
    std::cout << "DEBUG: minX / TILE_SIZE: " << std::fixed << std::setprecision(10) << (minX / TILE_SIZE) << "\n";
    std::cout << "DEBUG: minY / TILE_SIZE: " << std::fixed << std::setprecision(10) << (minY / TILE_SIZE) << "\n";
    std::cout << "DEBUG: maxX / TILE_SIZE: " << std::fixed << std::setprecision(10) << (maxX / TILE_SIZE) << "\n";
    std::cout << "DEBUG: maxY / TILE_SIZE: " << std::fixed << std::setprecision(10) << (maxY / TILE_SIZE) << "\n";
    // --- END ADDED PRECISION DEBUGGING ---

    int tx0 = static_cast<int>(std::floor(minX / TILE_SIZE));
    int ty0 = static_cast<int>(std::floor(minY / TILE_SIZE));
    int tx1 = static_cast<int>(std::floor(maxX / TILE_SIZE));
    int ty1 = static_cast<int>(std::floor(maxY / TILE_SIZE));

    // DEBUG: Print calculated tile range
    std::cout << "DEBUG: Calculated Tile Range (tx0, ty0) to (tx1, ty1): "
              << "(" << tx0 << ", " << ty0 << ") to (" << tx1 << ", " << ty1 << ")\n";

    // --- ADDED DEBUG FOR IMAGE CENTER AND ITS TILE KEY ---
    cv::Point2f imageCenterPx = {(float)img.cols / 2.0f, (float)img.rows / 2.0f};
    std::vector<cv::Point2f> projectedImageCenter;
    std::vector<cv::Point2f> imageCenterVec = {imageCenterPx};
    cv::perspectiveTransform(imageCenterVec, projectedImageCenter, homography);
    double imageCenterMosaicX = projectedImageCenter[0].x;
    double imageCenterMosaicY = projectedImageCenter[0].y;
    TileKey imageCenterTileKey = {
        static_cast<int>(std::floor(imageCenterMosaicX / TILE_SIZE)),
        static_cast<int>(std::floor(imageCenterMosaicY / TILE_SIZE))
    };
    std::cout << "DEBUG: Image Center (orig px): (" << std::fixed << std::setprecision(10) << imageCenterPx.x << ", " << std::fixed << std::setprecision(10) << imageCenterPx.y << ")\n";
    std::cout << "DEBUG: Image Center (mosaic px): (" << std::fixed << std::setprecision(10) << imageCenterMosaicX << ", " << std::fixed << std::setprecision(10) << imageCenterMosaicY << ")\n";
    std::cout << "DEBUG: Image Center Tile Key: (" << imageCenterTileKey.x << ", " << imageCenterTileKey.y << ")\n";
    // --- END ADDED DEBUG ---

    // The rest of applyImagePerTile remains the same, using imageCenterTileKey
    // ...
    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            TileKey key{tx, ty};
            cv::Rect tileRect(tx * TILE_SIZE, ty * TILE_SIZE, TILE_SIZE, TILE_SIZE);
            cv::Mat tile = loadTile(key);

            // Assuming you have applied the corrected warpTileRegion from my previous response.
            cv::Mat patch = warpTileRegion(img, homography, tileRect);
            if (patch.empty()) continue;

            // Apply alpha blending
            std::vector<cv::Mat> channels;
            cv::split(patch, channels);
            cv::Mat mask = channels[3] > 0;
            patch.copyTo(tile, mask);

            auto [tileLat, tileLon] = calculateTileGPS(key, imageCenterTileKey, lat, lon, gsd);

            saveTile(key, tile, tileLat, tileLon, exif);
            std::cout << "The tile " << getTilePath(key) << " coordinates are:" << "lat: " << tileLat << " lon: " << tileLon << "\n";
            updateGlobalBounds(key);
        }
    }
}

void TileManager::applyImage(const std::string& imagePath, const cv::Mat& homography, bool warpOnce) {
    loadGlobalMetadata();

    std::vector<std::string> sharedTags = {IMG_GPS_ALT, IMG_FOCAL_LEN_TAG, IMG_WIDTH_TAG};

    auto rawExif = exiftool_.getExifTags(imagePath, sharedTags);
    auto exif = exiftool_.parseExifValuesToNumbers(rawExif);

    double imageLat = exiftool_.parseExifGPS(exiftool_.getExifTag(imagePath, IMG_GPS_LAT));
    double imageLon = exiftool_.parseExifGPS(exiftool_.getExifTag(imagePath, IMG_GPS_LON));
    double gsd = (3.674 * exif.at(IMG_GPS_ALT)) / (exif.at(IMG_FOCAL_LEN_TAG) * exif.at(IMG_WIDTH_TAG));

    std::cout << "DEBUG: Image path: " << imagePath << "\n";
    std::cout << "DEBUG: Parsed Image Lat: " << std::fixed << std::setprecision(10) << imageLat << "\n";
    std::cout << "DEBUG: Parsed Image Lon: " << std::fixed << std::setprecision(10) << imageLon << "\n";
    std::cout << "DEBUG: Parsed GSD: " << std::fixed << std::setprecision(10) << gsd << "\n";

    cv::Mat img = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << imagePath << "\n";
        return;
    }

    if (img.channels() == 3) {
        cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);
    }

    std::cout << "min x  | min y | max x | max y \n" << globalMinX_ << "   " << globalMinY_ << "   " << globalMaxX_ << "   " << globalMaxY_ << "\n";

    if (warpOnce) {
        applyImageWarpOnce(img, homography, imageLat, imageLon, gsd, exif);
    } else {
        applyImagePerTile(img, homography, imageLat, imageLon, gsd, exif);
    }
}