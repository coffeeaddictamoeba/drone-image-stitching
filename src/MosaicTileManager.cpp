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

void TileManager::loadGlobalMetadata() {
    const std::string path = outputDirectory_ + "/" + COORDS_METADATA;
    std::ifstream in(path);
    
    if (!in) return;

    std::string key;
    double double_value;
    int int_value;

    while (in >> key) {
        if (key == "min_x:") {in >> int_value; globalMinX_ = int_value;}
        else if (key == "min_y:") {in >> int_value; globalMinY_ = int_value;}
        else if (key == "max_x:") {in >> int_value; globalMaxX_ = int_value;}
        else if (key == "max_y:") {in >> int_value; globalMaxY_ = int_value;}
        else if (key == "heading:") {in >> double_value; globalHeading_ = double_value;}
        else if (key == "origin_lat:") { in >> double_value; mosaicOriginLat_ = double_value; }
        else if (key == "origin_lon:") { in >> double_value; mosaicOriginLon_ = double_value; }
    }

    #ifdef DEBUG
    std::cout << "DEBUG: Loaded global metadata: min_x=" << globalMinX_ << ", min_y=" << globalMinY_
              << ", max_x=" << globalMaxX_ << ", max_y=" << globalMaxY_ << ", heading=" << globalHeading_
              << ", origin_lat=" << std::fixed << std::setprecision(10) << mosaicOriginLat_
              << ", origin_lon=" << std::fixed << std::setprecision(10) << mosaicOriginLon_ << "\n";
    #endif
}

void TileManager::saveGlobalMetadata() const {
    const std::string path = outputDirectory_ + "/" + COORDS_METADATA;
    std::ofstream out(path);
    out << "min_x: " << globalMinX_ << "\n"
        << "min_y: " << globalMinY_ << "\n"
        << "max_x: " << globalMaxX_ << "\n"
        << "max_y: " << globalMaxY_ << "\n"
        << "heading: " << globalHeading_ << "\n"
        << "origin_lat: " << std::fixed << std::setprecision(10) << mosaicOriginLat_ << "\n"
        << "origin_lon: " << std::fixed << std::setprecision(10) << mosaicOriginLon_ << "\n";
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

// std::pair<double, double> TileManager::calculateTileGPS(const TileKey& tileKey, const TileKey& centerTile, double centerLat, double centerLon, double gsd) const {
//     int dx = tileKey.x - centerTile.x;
//     int dy = tileKey.y - centerTile.y;

//     double offsetX_m = static_cast<double>(dx) * TILE_SIZE * gsd;
//     double offsetY_m = static_cast<double>(dy) * TILE_SIZE * gsd;

//     double hrad = globalHeading_ * M_PI / 180.0;

//     double angleX = (globalHeading_ + 90.0) * M_PI / 180.0;
//     double angleY = (globalHeading_ + 180.0) * M_PI / 180.0;

//     double eastOffset = (offsetX_m * std::sin(angleX)) + (offsetY_m * std::sin(angleY));
//     double northOffset = (offsetX_m * std::cos(angleX)) + (offsetY_m * std::cos(angleY));
    
//     double metersPerDegLon = M_PER_DEGREE_LATITUDE * std::cos(centerLat * M_PI / 180.0);

//     double lat = centerLat - (northOffset / M_PER_DEGREE_LATITUDE);
//     double lon = centerLon + (eastOffset / metersPerDegLon);

//     return {lat, lon};
// }

std::pair<double, double> TileManager::calculateTileGPS(const TileKey& tileKey, const TileKey& startTile, double gsd) const {
    int dx = tileKey.x - startTile.x;
    int dy = tileKey.y - startTile.y;

    double offsetX_m = static_cast<double>(dx) * TILE_SIZE * gsd;
    double offsetY_m = static_cast<double>(dy) * TILE_SIZE * gsd;
    
    double heading_rad = globalHeading_ * M_PI / 180.0;

    double cos_heading = std::cos(heading_rad);
    double sin_heading = std::sin(heading_rad);

    double true_east_offset_m = (offsetX_m * cos_heading) + (-offsetY_m * sin_heading);
    double true_north_offset_m = (-offsetX_m * sin_heading) + (-offsetY_m * cos_heading);
    
    double metersPerDegLon = M_PER_DEGREE_LATITUDE * std::cos(mosaicOriginLat_ * M_PI / 180.0);

    double lat = mosaicOriginLat_ + (true_north_offset_m / M_PER_DEGREE_LATITUDE);
    double lon = mosaicOriginLon_ + (true_east_offset_m / metersPerDegLon);

    #ifdef DEBUG
    std::cout << std::fixed << std::setprecision(10); // Set precision for all subsequent double outputs
    std::cout << "DEBUG: calculateTileGPS for TileKey (" << tileKey.x << "," << tileKey.y << "):\n";
    std::cout << "  centerLat=" << mosaicOriginLat_ << ", centerLon=" << mosaicOriginLon_ << ", gsd=" << gsd << ", globalHeading_=" << globalHeading_ << "\n";
    std::cout << "  offsetX_m (mosaic)=" << offsetX_m << ", offsetY_m (mosaic)=" << offsetY_m << "\n";
    std::cout << "  true_east_offset_m=" << true_east_offset_m << ", true_north_offset_m=" << true_north_offset_m << "\n";
    std::cout << "  Calculated Lat=" << lat << ", Lon=" << lon << "\n";
    #endif

    return {lat, lon};
}

double TileManager::estimateGSD(const std::map<std::string, double> exif) const {
    if (exif.count(EXIFTAGS::FOCAL_LENGTH_TAG) && exif.count(EXIFTAGS::IMAGE_WIDTH_TAG) && exif.at(EXIFTAGS::FOCAL_LENGTH_TAG) != 0.0 && exif.at(EXIFTAGS::IMAGE_WIDTH_TAG) != 0.0) {
        double alt = exif.count(EXIFTAGS::GPS_ALTITUDE_TAG) ? exif.at(EXIFTAGS::GPS_ALTITUDE_TAG) : 0.0;
        double flen = exif.count(EXIFTAGS::FOCAL_LENGTH_TAG) ? exif.at(EXIFTAGS::FOCAL_LENGTH_TAG) : 0.0;
        double width_px = exif.count(EXIFTAGS::IMAGE_WIDTH_TAG) ? exif.at(EXIFTAGS::IMAGE_WIDTH_TAG) : 0.0;
        return (3.674 * alt) / (flen * width_px);
    }
    return 0.0;
}

cv::Mat TileManager::loadTile(const TileKey& key) const {
    std::string path = getTilePath(key);
    if (fs::exists(path)) {
        cv::Mat tile = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (!tile.empty()) return tile;
    }
    return cv::Mat(TILE_SIZE, TILE_SIZE, CV_8UC4, cv::Scalar(0, 0, 0, 0));
}

void TileManager::assignMetadata(const std::string imagePath, const double lat, const double lon, const double alt, const double flen, const double gpsDir) const {
    std::ostringstream tagStream;
    tagStream << "-n\n";
    tagStream << "-" << EXIFTAGS::GPS_LATITUDE_TAG << "=" << lat << "\n";
    tagStream << "-" << EXIFTAGS::GPS_LONGITUDE_TAG << "=" << lon << "\n";
    tagStream << "-" << EXIFTAGS::GPS_ALTITUDE_TAG << "=" << alt << "\n";
    tagStream << "-" << EXIFTAGS::FOCAL_LENGTH_TAG << "=" << flen << "\n";
    tagStream << "-" << EXIFTAGS::GPS_IMG_DIRECTION_TAG << "=" << gpsDir << "\n";

    exiftool_.setExifTag(imagePath, tagStream.str());
}

void TileManager::saveTile(const TileKey& key, const cv::Mat& tile, const double lat, const double lon, const std::map<std::string, double>& exif) const {
    std::string path = getTilePath(key);
    cv::imwrite(path, tile);

    assignMetadata(path, 
        lat, 
        lon, 
        exif.at(EXIFTAGS::GPS_ALTITUDE_TAG), 
        exif.at(EXIFTAGS::FOCAL_LENGTH_TAG), 
        exif.at(EXIFTAGS::GPS_IMG_DIRECTION_TAG) // it is not the same as globalHeading_. this is made for tile to repeat the source image metadata
    );
}

cv::Mat TileManager::warpTileRegion(const cv::Mat& input, const cv::Mat& H, const cv::Rect& tileRect) const {
    cv::Mat tileOutput = cv::Mat::zeros(TILE_SIZE, TILE_SIZE, input.type());

    cv::Mat translationMatrix = cv::Mat::eye(3, 3, CV_64F);
    translationMatrix.at<double>(0, 2) = -tileRect.x;
    translationMatrix.at<double>(1, 2) = -tileRect.y;

    cv::Mat M_img_to_tile_local = translationMatrix * H;

    cv::warpPerspective(input, tileOutput, M_img_to_tile_local, tileOutput.size(),
                        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0,0));

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

void TileManager::applyImageWarpOnce(const cv::Mat& img, const cv::Mat& homography, double gsd, std::map<std::string, double> exif) {
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

            auto [tileLat, tileLon] = calculateTileGPS(key, TileKey{0,0}, gsd);
            saveTile(key, tile, tileLat, tileLon, exif);
            updateGlobalBounds(key);
        }
    }
}

void TileManager::applyImagePerTile(const cv::Mat& img, const cv::Mat& homography, double gsd, std::map<std::string, double> exif) {
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

    int tx0 = static_cast<int>(std::floor(minX / TILE_SIZE));
    int ty0 = static_cast<int>(std::floor(minY / TILE_SIZE));
    int tx1 = static_cast<int>(std::floor(maxX / TILE_SIZE));
    int ty1 = static_cast<int>(std::floor(maxY / TILE_SIZE));

    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            TileKey key{tx, ty};
            cv::Rect tileRect(tx * TILE_SIZE, ty * TILE_SIZE, TILE_SIZE, TILE_SIZE);
            cv::Mat tile = loadTile(key);

            cv::Mat patch = warpTileRegion(img, homography, tileRect);
            if (patch.empty()) continue;

            std::vector<cv::Mat> channels;
            cv::split(patch, channels);
            cv::Mat mask = channels[3] > 0;
            patch.copyTo(tile, mask);

            auto [tileLat, tileLon] = calculateTileGPS(key, TileKey{0, 0}, gsd);
            saveTile(key, tile, tileLat, tileLon, exif);
            updateGlobalBounds(key);
        }
    }
}

void TileManager::applyImage(const std::string& imagePath, const cv::Mat& homography, bool warpOnce) {
    loadGlobalMetadata();

    std::vector<std::string> sharedTags = {
        EXIFTAGS::GPS_ALTITUDE_TAG, 
        EXIFTAGS::FOCAL_LENGTH_TAG, 
        EXIFTAGS::IMAGE_WIDTH_TAG,
        EXIFTAGS::GPS_IMG_DIRECTION_TAG,
        EXIFTAGS::GPS_LATITUDE_TAG,
        EXIFTAGS::GPS_LONGITUDE_TAG
    };

    auto rawExifMap = exiftool_.getExifTags(imagePath, sharedTags);
    auto parsedExif = exiftool_.parseExifValuesToNumbers(rawExifMap);

    std::cout << "DEBUG: Image path: " << imagePath << "\n";
    std::cout << "  GPS_ALTITUDE_TAG: " << std::fixed << std::setprecision(10) << (parsedExif.count(EXIFTAGS::GPS_ALTITUDE_TAG) ? parsedExif.at(EXIFTAGS::GPS_ALTITUDE_TAG) : 0.0) << "\n";
    std::cout << "  FOCAL_LENGTH_TAG: " << std::fixed << std::setprecision(10) << (parsedExif.count(EXIFTAGS::FOCAL_LENGTH_TAG) ? parsedExif.at(EXIFTAGS::FOCAL_LENGTH_TAG) : 0.0) << "\n";
    std::cout << "  IMAGE_WIDTH_TAG: " << std::fixed << std::setprecision(10) << (parsedExif.count(EXIFTAGS::IMAGE_WIDTH_TAG) ? parsedExif.at(EXIFTAGS::IMAGE_WIDTH_TAG) : 0.0) << "\n";
    std::cout << "DEBUG: Parsed Image Lat: " << std::fixed << std::setprecision(10) << (parsedExif.count(EXIFTAGS::GPS_LATITUDE_TAG) ? parsedExif.at(EXIFTAGS::GPS_LATITUDE_TAG) : 0.0) << "\n";
    std::cout << "DEBUG: Parsed Image Lon: " << std::fixed << std::setprecision(10) << (parsedExif.count(EXIFTAGS::GPS_LONGITUDE_TAG) ? parsedExif.at(EXIFTAGS::GPS_LONGITUDE_TAG) : 0.0) << "\n";
    std::cout << "DEBUG: Parsed GPSImgDirection: " << std::fixed << std::setprecision(2) << (parsedExif.count(EXIFTAGS::GPS_IMG_DIRECTION_TAG) ? parsedExif.at(EXIFTAGS::GPS_IMG_DIRECTION_TAG) : 0.0) << "\n";
    std::cout << "DEBUG: Current globalHeading_ (before potential update): " << std::fixed << std::setprecision(2) << globalHeading_ << "\n";

    if (globalMinX_ == 0 && globalMinY_ == 0 && globalMaxX_ == 0 && globalMaxY_ == 0) {
        if (parsedExif.count(EXIFTAGS::GPS_IMG_DIRECTION_TAG)) {
            globalHeading_ = parsedExif.at(EXIFTAGS::GPS_IMG_DIRECTION_TAG);
            mosaicOriginLat_ = parsedExif.at(EXIFTAGS::GPS_LATITUDE_TAG);
            mosaicOriginLon_ = parsedExif.at(EXIFTAGS::GPS_LONGITUDE_TAG);
            #ifdef DEBUG
            std::cout << "DEBUG: First image detected. Setting globalHeading_ to " << globalHeading_ << " from current image.\n";
            #endif
        } else {
            std::cerr << "Warning: GPSImgDirection tag not found for first image. Defaulting mosaic heading to 0.\n";
            globalHeading_ = 0.0;
        }
    }

    double gsd = estimateGSD(parsedExif);

    cv::Mat img = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << imagePath << "\n";
        return;
    }

    if (img.channels() == 3) {
        cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);
    }

    if (warpOnce) {
        applyImageWarpOnce(img, homography, gsd, parsedExif);
    } else {
        applyImagePerTile(img, homography, gsd, parsedExif);
    }
}