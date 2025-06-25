#include "../include/mosaic.h"
#include "../include/fmatch.h"
#include <cmath>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>
#include <optional>
#include <string>

namespace fs = std::filesystem;

MosaicTileManager::MosaicTileManager(const std::string& outputDir, ExifToolPipe& tool)
    : outputDirectory_(outputDir), exiftool_(tool)
{
    std::filesystem::create_directories(outputDirectory_);
}

TileKey MosaicTileManager::getTileKeyForPoint(int x, int y) const {
    return { x / TILE_SIZE, y / TILE_SIZE };
}

std::string MosaicTileManager::getOutputDirectory() const {
    return outputDirectory_;
}

std::string MosaicTileManager::getTilePath(const TileKey& key) const {
    return outputDirectory_ + "/tile_" + std::to_string(key.y) + "_" + std::to_string(key.x) + ".png";
}

// TODO: make more flexible
std::pair<double, double> MosaicTileManager::extractGPS(const std::string& imagePath) const {
    auto parseCoordinate = [](const std::string& value) -> double {
        std::string str = std::regex_replace(value, std::regex("^ +| +$|( ) +"), "$1");

        try {
            return std::stod(str);
        } catch (...) {

        }

        std::smatch match;
        std::regex dmsRegex(R"((\d+)[^\d]+(\d+)[^\d]+([\d.]+))"); // DMS format: 54 deg 54' 19.67"
        if (std::regex_search(str, match, dmsRegex) && match.size() == 4) {
            double degrees = std::stod(match[1]);
            double minutes = std::stod(match[2]);
            double seconds = std::stod(match[3]);
            return degrees + minutes / 60.0 + seconds / 3600.0;
        }

        throw std::runtime_error("Failed to parse GPS coordinate: " + str);
    };

    std::string latStr = exiftool_.inExifTag(imagePath, "GPSLatitude");
    std::string lonStr = exiftool_.inExifTag(imagePath, "GPSLongitude");

    if (latStr.empty() || lonStr.empty()) {
        throw std::runtime_error("Missing GPS metadata in image: " + imagePath);
    }

    double lat = parseCoordinate(latStr);
    double lon = parseCoordinate(lonStr);

    return { lat, lon };
}

// Make tags as constants?
double MosaicTileManager::estimateGSD(const std::string& imagePath) const {
    // Constants for Pi Camera V2
    const double sensorWidth = 3.674; // mm

    std::string altitudeStr = exiftool_.inExifTag(imagePath, "GPSAltitude");
    std::string focalStr = exiftool_.inExifTag(imagePath, "FocalLength");
    std::string widthStr = exiftool_.inExifTag(imagePath, "ImageWidth");

    if (altitudeStr.empty() || focalStr.empty() || widthStr.empty()) {
        throw std::runtime_error("Missing required EXIF tags for GSD computation.");
    }

    double altitude = exiftool_.parseExifNumber(altitudeStr); // meters
    double focalLength = exiftool_.parseExifNumber(focalStr); // mm
    int imageWidth = std::stoi(widthStr);                       // px

    if (focalLength <= 0 || imageWidth <= 0) {
        throw std::runtime_error("Invalid focal length or image width for GSD computation.");
    }
    return (sensorWidth * altitude) / (focalLength * imageWidth); // m/px
}

std::pair<double, double> MosaicTileManager::calculateTileGPS(
    const TileKey& tileKey,
    const TileKey& centerTile,
    double centerLat,
    double centerLon,
    double gsd) const
{
    int dx = tileKey.x - centerTile.x;
    int dy = tileKey.y - centerTile.y;

    double offsetX_m = dx * TILE_SIZE * gsd;
    double offsetY_m = dy * TILE_SIZE * gsd;
    
    double metersPerDegLon = M_PER_DEGREE_LATITUDE * std::cos(centerLat * M_PI / 180.0);

    double lat = centerLat - (offsetY_m / M_PER_DEGREE_LATITUDE);
    double lon = centerLon + (offsetX_m / metersPerDegLon);

    return {lat, lon};
}

cv::Mat MosaicTileManager::loadTile(const TileKey& key) const {
    std::string path = getTilePath(key);
    if (fs::exists(path)) {
        cv::Mat tile = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (!tile.empty()) return tile;
    }
    return cv::Mat(TILE_SIZE, TILE_SIZE, CV_8UC4, cv::Scalar(0, 0, 0, 0));
}

void MosaicTileManager::assignMetadata(const std::string imagePath, const double lat, const double lon, const double alt, const double flen) const {
    std::ostringstream tagStream;
    tagStream << "-n\n";
    tagStream << "-GPSLatitude=" << lat << "\n";
    tagStream << "-GPSLongitude=" << lon << "\n";
    tagStream << "-GPSAltitude=" << alt << "\n";
    tagStream << "-FocalLength=" << flen << "\n";

    exiftool_.setExifTag(imagePath, tagStream.str());
}

void MosaicTileManager::saveTile(const TileKey& key, const cv::Mat& tile, const double lat, const double lon, const std::string imagePath) const {
    std::string path = getTilePath(key);
    cv::imwrite(path, tile);
    
    double alt = exiftool_.parseExifNumber(exiftool_.inExifTag(imagePath, "GPSAltitude"));
    double flen = exiftool_.parseExifNumber(exiftool_.inExifTag(imagePath, "FocalLength"));
    assignMetadata(path, lat, lon, alt, flen);
}

cv::Mat MosaicTileManager::warpTileRegion(const cv::Mat& input,
                                          const cv::Mat& H,
                                          const cv::Rect& tileRect) const
{
    std::vector<cv::Point2f> tileCorners = {
        cv::Point2f(tileRect.x, tileRect.y),
        cv::Point2f(tileRect.x + TILE_SIZE, tileRect.y),
        cv::Point2f(tileRect.x + TILE_SIZE, tileRect.y + TILE_SIZE),
        cv::Point2f(tileRect.x, tileRect.y + TILE_SIZE)
    };

    cv::Mat H_inv = H.inv();
    std::vector<cv::Point2f> srcQuad;
    cv::perspectiveTransform(tileCorners, srcQuad, H_inv);

    cv::Rect srcBoundingBox = cv::boundingRect(srcQuad);
    if ((srcBoundingBox & cv::Rect(0, 0, input.cols, input.rows)) != srcBoundingBox)
        return cv::Mat();

    cv::Mat croppedInput = input(srcBoundingBox).clone(); // think about a better alternative (memory-inefficient)

    for (auto& pt : srcQuad) pt -= cv::Point2f(srcBoundingBox.x, srcBoundingBox.y);

    std::vector<cv::Point2f> dstQuad = {
        cv::Point2f(0, 0),
        cv::Point2f(TILE_SIZE, 0),
        cv::Point2f(TILE_SIZE, TILE_SIZE),
        cv::Point2f(0, TILE_SIZE)
    };

    cv::Mat tileOutput;
    cv::Mat tileH = cv::getPerspectiveTransform(srcQuad, dstQuad);
    cv::warpPerspective(croppedInput, tileOutput, tileH,
                        cv::Size(TILE_SIZE, TILE_SIZE),
                        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0,0));

    return tileOutput;
}

cv::Mat MosaicTileManager::computeTileHomography(const TileKey& tileKey, const cv::Mat& homography) {
    double offsetX = tileKey.x * TILE_SIZE;
    double offsetY = tileKey.y * TILE_SIZE;

    cv::Mat T_offset = (cv::Mat_<double>(3, 3) <<
        1, 0, offsetX,
        0, 1, offsetY,
        0, 0, 1);

    cv::Mat adjustedHomography = homography * T_offset;

    return adjustedHomography;
}

cv::Mat MosaicTileManager::computeGlobalHomography(
    const TileKey& localOriginKey,
    const cv::Mat& localHomography)
{
    double offsetX = localOriginKey.x * TILE_SIZE;
    double offsetY = localOriginKey.y * TILE_SIZE;

    cv::Mat offsetMat = (cv::Mat_<double>(3, 3) <<
        1, 0, offsetX,
        0, 1, offsetY,
        0, 0, 1);

    return offsetMat * localHomography;
}

void MosaicTileManager::applyImageWarpOnce(const std::string& imagePath,
                                           const cv::Mat& homography)
{
    auto [lat, lon] = extractGPS(imagePath);
    double gsd = estimateGSD(imagePath);

    cv::Mat img = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << imagePath << "\n";
        return;
    }
    if (img.channels() == 3) {
        cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);
    }

    // Warp image onto canvas
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

    TileKey centerKey{ (tx0 + tx1) / 2, (ty0 + ty1) / 2 };

    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            TileKey key{tx, ty};
            cv::Rect globalTileRect(tx * TILE_SIZE, ty * TILE_SIZE, TILE_SIZE, TILE_SIZE);
            cv::Rect localTileRect(globalTileRect.x - minX, globalTileRect.y - minY, TILE_SIZE, TILE_SIZE);

            cv::Mat tile = loadTile(key);

            if (localTileRect.x < 0 || localTileRect.y < 0 ||
                localTileRect.x + TILE_SIZE > warpCanvas.cols ||
                localTileRect.y + TILE_SIZE > warpCanvas.rows) {
                continue;
            }

            cv::Mat patch = warpCanvas(localTileRect);
            for (int y = 0; y < TILE_SIZE; ++y) {
                for (int x = 0; x < TILE_SIZE; ++x) {
                    cv::Vec4b p = patch.at<cv::Vec4b>(y, x);
                    if (p[3]) {
                        tile.at<cv::Vec4b>(y, x) = p;
                    }
                }
            }

            auto [tileLat, tileLon] = calculateTileGPS(key, centerKey, lat, lon, gsd);
            saveTile(key, tile, tileLat, tileLon, imagePath);
        }
    }
}

void MosaicTileManager::applyImagePerTile(const std::string& imagePath,
                                          const cv::Mat& homography)
{
    auto [lat, lon] = extractGPS(imagePath);

    cv::Mat img = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << imagePath << "\n";
        return;
    }
    if (img.channels() == 3) {
        cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);
    }

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

    int tx0 = std::floor(minX / TILE_SIZE), ty0 = std::floor(minY / TILE_SIZE);
    int tx1 = std::floor(maxX / TILE_SIZE), ty1 = std::floor(maxY / TILE_SIZE);

    TileKey centerKey{ (tx0 + tx1) / 2, (ty0 + ty1) / 2 };
    double gsd = estimateGSD(imagePath);

    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            TileKey key{tx, ty};
            cv::Rect tileRect(tx * TILE_SIZE, ty * TILE_SIZE, TILE_SIZE, TILE_SIZE);
            cv::Mat tile = loadTile(key);
            cv::Mat patch = warpTileRegion(img, homography, tileRect);
            if (patch.empty()) continue;

            for (int y = 0; y < TILE_SIZE; ++y) {
                for (int x = 0; x < TILE_SIZE; ++x) {
                    cv::Vec4b p = patch.at<cv::Vec4b>(y, x);
                    if (p[3]) {
                        tile.at<cv::Vec4b>(y, x) = p;
                    }
                }
            }

            auto [tileLat, tileLon] = calculateTileGPS(key, centerKey, lat, lon, gsd);
            saveTile(key, tile, tileLat, tileLon, imagePath);
        }
    }
}

void MosaicTileManager::applyImage(const std::string& imagePath,
                                   const cv::Mat& homography,
                                   bool warpOnce)
{
    if (warpOnce) {
        applyImageWarpOnce(imagePath, homography);
    } else {
        applyImagePerTile(imagePath, homography);
    }
}