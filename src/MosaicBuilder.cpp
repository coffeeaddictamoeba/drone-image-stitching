#include "../include/mosaic.h"
#include "../include/fmatch.h"
#include <cmath>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>
#include <optional>
#include <string>

namespace fs = std::filesystem;

MosaicBuilder::MosaicBuilder(const std::string& refImagePath,
    const std::string& targetImagePath,
    ExifToolPipe& tool,
    MosaicTileManager& tileManager)
    : refImagePath_(refImagePath),
      targetImagePath_(targetImagePath),
      exiftool_(tool),
      tiles_(tileManager) {}

ImageMatrix MosaicBuilder::toImageMatrix(std::string imagePath) {
    cv::Mat matrix = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    if (matrix.empty()) {
        std::cerr << "Failed to load the image.\n";
        exit(1);
    }
    return ImageMatrix{matrix, imagePath};
}

bool MosaicBuilder::loadImages() {
    ref_ = toImageMatrix(refImagePath_);
    target_ = toImageMatrix(targetImagePath_);
    return !ref_.imageMatrix.empty() && !target_.imageMatrix.empty();
}

bool MosaicBuilder::loadImages(std::string refImagePath, std::string targetImagePath) {
    ref_ = toImageMatrix(refImagePath);
    target_ = toImageMatrix(targetImagePath);
    return !ref_.imageMatrix.empty() && !target_.imageMatrix.empty();
}

bool MosaicBuilder::alignImages(const ImageMatrix& src,
                                const ImageMatrix& dst,
                                cv::Mat& H)
{
    FeatureMatcher matcher(exiftool_, "SIFT");
    std::vector<cv::DMatch> matches;
    if (!matcher.computeHomography(src, dst, H, matches)) {
        std::cerr << "Feature matching failed.\n";
        return false;
    }
    return true;
}

bool MosaicBuilder::stitchToTiles() {
    if (!loadImages()) {
        std::cerr << "Failed to load input images.\n";
        return false;
    }

    tiles_.applyImage(refImagePath_, cv::Mat::eye(3, 3, CV_64F), false);    

    if (!alignImages(ref_, target_, homography_)) { return false; }

    tiles_.applyImage(targetImagePath_, homography_, false);
    return true;
}

bool MosaicBuilder::stitchToTiles(std::string refImagePath, std::string targetImagePath) {
    if (!loadImages(refImagePath, targetImagePath)) {
        std::cerr << "Failed to load input images.\n";
        return false;
    }

    tiles_.applyImage(refImagePath, cv::Mat::eye(3, 3, CV_64F), false);    

    if (!alignImages(ref_, target_, homography_)) { return false; }

    tiles_.applyImage(targetImagePath, homography_, false);
    return true;
}

// Mosaic from tiles: fixed size
cv::Mat MosaicBuilder::mosaicFromTiles(
    const std::string& tileDir, 
    cv::Rect& mosaicBounds, 
    int startX, 
    int startY, 
    int endX,
    int endY) {
    std::map<std::pair<int, int>, std::string> tileMap;

    for (const auto& entry : fs::directory_iterator(tileDir)) {
        if (!entry.is_regular_file()) continue;
        std::string filename = entry.path().filename().string();

        std::smatch match;
        if (std::regex_match(filename, match, TILE_REGEX)) {
            int ty = std::stoi(match[1]);
            int tx = std::stoi(match[2]);
            tileMap[{tx, ty}] = entry.path().string();

            startX = std::min(startX, tx);
            startY = std::min(startY, ty);
            endX = std::max(endX, tx);
            endY = std::max(endY, ty);
        }
    }

    if (tileMap.empty()) {
        std::cerr << "No tiles found in: " << tileDir << "\n";
        return {};
    }

    int mosaicWidth = (endX - startX + 1) * TILE_SIZE;
    int mosaicHeight = (endY - startY + 1) * TILE_SIZE;
    mosaicBounds = cv::Rect(startX * TILE_SIZE, startY * TILE_SIZE, mosaicWidth, mosaicHeight);

    cv::Mat mosaic(mosaicHeight, mosaicWidth, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    for (const auto& [key, path] : tileMap) {
        int tx = key.first, ty = key.second;
        cv::Mat tile = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (tile.empty()) {
            std::cerr << "Failed to read tile: " << path << "\n";
            continue;
        }
        int x = (tx - startX) * TILE_SIZE;
        int y = (ty - startY) * TILE_SIZE;

        tile.copyTo(mosaic(cv::Rect(x, y, TILE_SIZE, TILE_SIZE)));
    }
    return mosaic;
}

// Mosaic from tiles: full size
cv::Mat MosaicBuilder::mosaicFromTiles(const std::string& tileDir, cv::Rect& mosaicBounds) {
    std::map<std::pair<int, int>, std::string> tileMap;

    int minX = INT_MAX, minY = INT_MAX, maxX = INT_MIN, maxY = INT_MIN;

    for (const auto& entry : fs::directory_iterator(tileDir)) {
        if (!entry.is_regular_file()) continue;
        std::string filename = entry.path().filename().string();

        std::smatch match;
        if (std::regex_match(filename, match, TILE_REGEX)) {
            int ty = std::stoi(match[1]);
            int tx = std::stoi(match[2]);
            tileMap[{tx, ty}] = entry.path().string();

            minX = std::min(minX, tx);
            minY = std::min(minY, ty);
            maxX = std::max(maxX, tx);
            maxY = std::max(maxY, ty);
        }
    }

    if (tileMap.empty()) {
        std::cerr << "No tiles found in: " << tileDir << "\n";
        return {};
    }

    int mosaicWidth = (maxX - minX + 1) * TILE_SIZE;
    int mosaicHeight = (maxY - minY + 1) * TILE_SIZE;
    mosaicBounds = cv::Rect(minX * TILE_SIZE, minY * TILE_SIZE, mosaicWidth, mosaicHeight);

    cv::Mat mosaic(mosaicHeight, mosaicWidth, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    for (const auto& [key, path] : tileMap) {
        int tx = key.first, ty = key.second;
        cv::Mat tile = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (tile.empty()) {
            std::cerr << "Failed to read tile: " << path << "\n";
            continue;
        }
        int x = (tx - minX) * TILE_SIZE;
        int y = (ty - minY) * TILE_SIZE;

        tile.copyTo(mosaic(cv::Rect(x, y, TILE_SIZE, TILE_SIZE)));
    }
    return mosaic;
}

bool MosaicBuilder::isValidTile(std::string tilePath) {
    cv::Mat img = cv::imread(tilePath, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cout << "Failed to load image: " << tilePath << "\n";
        return false;
    }

    if (img.channels() == 4) {
        std::vector<cv::Mat> channels;
        cv::split(img, channels);
        double minAlpha, maxAlpha;
        cv::minMaxLoc(channels[3], &minAlpha, &maxAlpha);
        if (maxAlpha < 1) {
            std::cout << "Skipping fully transparent tile: " << tilePath << "\n";
            return false;
        }
    }

    cv::Mat gray;
    if (img.channels() == 4 || img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img;

    double minVal, maxVal;
    cv::minMaxLoc(gray, &minVal, &maxVal);
    if (std::abs(maxVal - minVal) < 1e-3) {
        std::cout << "Skipping visually uniform tile: " << tilePath << "\n";
        return false;
    }

    return true;
}

double MosaicBuilder::findTileDistance(std::string tilePath, double latToCompare, double lonToCompare) {
    auto [lat, lon] = tiles_.extractGPS(tilePath);

    double dLat = (lat - latToCompare) * DEG_TO_RAD;
    double dLon = (lon - lonToCompare) * DEG_TO_RAD;

    // Haversine formula
    double a = std::sin(dLat/2) * std::sin(dLat/2) + 
               std::cos(latToCompare * DEG_TO_RAD) * std::cos(lat * DEG_TO_RAD) * 
               std::sin(dLon/2) * std::sin(dLon/2);

    double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1-a)); // central angle

    return R_M * c; // distance in meters
}

std::optional<TileKey> MosaicBuilder::findClosestTile(const std::string& imagePath) {
    auto [lat1, lon1] = tiles_.extractGPS(imagePath);

    double bestDist = DBL_MAX;
    std::optional<TileKey> bestKey;

    for (const auto& entry : fs::directory_iterator(tiles_.getOutputDirectory())) {
        std::string filename = entry.path().filename().string();
        std::smatch match;
        if (!std::regex_match(filename, match, TILE_REGEX)) continue;

        int ty = std::stoi(match[1]);
        int tx = std::stoi(match[2]);
        std::string tilePath = entry.path().string();

        if (!isValidTile(tilePath)) continue;

        double dist = findTileDistance(tilePath, lat1, lon1);

        if (dist < bestDist || (std::abs(dist - bestDist) < 1e-6 &&
            (tx < bestKey->x || (tx == bestKey->x && ty < bestKey->y)))) {
            bestDist = dist;
            bestKey = TileKey{tx, ty};
        }
    }
    TileKey key = *bestKey;
    std::cout << "The best matching tile: " << tiles_.getTilePath(key) << '\n';
    return bestKey;
}

// Get mosaic around central tile 3x3 size
cv::Mat MosaicBuilder::getMosaicAroundTile(TileKey center, int radius, cv::Rect& outBounds) {
    int minX = center.x - radius;
    int maxX = center.x + radius;
    int minY = center.y - radius;
    int maxY = center.y + radius;

    int width = (maxX - minX + 1) * TILE_SIZE;
    int height = (maxY - minY + 1) * TILE_SIZE;

    outBounds = cv::Rect(minX * TILE_SIZE, minY * TILE_SIZE, width, height);
    cv::Mat mosaic(height, width, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    for (int tx = minX; tx <= maxX; ++tx) {
        for (int ty = minY; ty <= maxY; ++ty) {
            TileKey key{tx, ty};
            std::string path = tiles_.getTilePath(key);
            if (!fs::exists(path) || !isValidTile(path)) continue;

            cv::Mat tile = cv::imread(path, cv::IMREAD_UNCHANGED);
            if (tile.empty()) continue;

            int x = (tx - minX) * TILE_SIZE;
            int y = (ty - minY) * TILE_SIZE;
            tile.copyTo(mosaic(cv::Rect(x, y, TILE_SIZE, TILE_SIZE)));
        }
    }
    return mosaic;
}

bool MosaicBuilder::addImageToMosaic(const std::string& newImagePath) {
    auto bestTileKeyOpt = findClosestTile(newImagePath);
    if (!bestTileKeyOpt) return false;

    int height = exiftool_.parseExifNumber(exiftool_.inExifTag(newImagePath, "ImageHeight"));
    int width = exiftool_.parseExifNumber(exiftool_.inExifTag(newImagePath, "ImageWidth"));
    int tileRadius = std::ceil(std::max(height, width) / float(TILE_SIZE)) / 2;
    
    TileKey bestKey = *bestTileKeyOpt;
    TileKey localOriginKey{ bestKey.x - tileRadius, bestKey.y - tileRadius };
    
    cv::Rect localBounds;
    cv::Mat localMosaic = getMosaicAroundTile(bestKey, tileRadius, localBounds);
    if (localMosaic.empty()) {
        std::cerr << "Failed to build local mosaic.\n";
        return false;
    }
    
    ImageMatrix mosaicMatrix{localMosaic, "local mosaic"};
    ImageMatrix newImageMatrix = toImageMatrix(newImagePath);
    
    cv::Mat H;
    if (!alignImages(newImageMatrix, mosaicMatrix, H)) return false;

    H = H.inv();
    cv::Mat withOffset = tiles_.computeGlobalHomography(localOriginKey, H);
    
    tiles_.applyImage(newImagePath, withOffset, true);
    return true;    
}