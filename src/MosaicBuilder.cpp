#include "../include/mosaic.h"
#include "../include/fmatch.h"
#include "../external/ctre.hpp"
#include <cmath>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <future>

namespace fs = std::filesystem;

MosaicBuilder::MosaicBuilder(ExifToolPipe& tool, TileManager& tileManager) : exiftool_(tool), tiles_(tileManager) {}

ImageMatrix MosaicBuilder::toImageMatrix(std::string imagePath) const {
    cv::Mat matrix = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    if (matrix.empty()) {
        std::cerr << "Failed to load the image.\n";
        exit(1);
    }
    return ImageMatrix{matrix, imagePath};
}

bool MosaicBuilder::loadImages(std::string refImagePath, std::string targetImagePath) {
    ref_ = toImageMatrix(refImagePath);
    target_ = toImageMatrix(targetImagePath);
    return !ref_.imageMatrix.empty() && !target_.imageMatrix.empty();
}

bool MosaicBuilder::alignImages(const ImageMatrix& src, const ImageMatrix& dst, cv::Mat& H) {
    FeatureMatcher matcher(exiftool_, "SIFT");
    std::vector<cv::DMatch> matches;
    if (!matcher.computeHomography(src, dst, H, matches)) {
        std::cerr << "Feature matching failed.\n";
        return false;
    }
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
cv::Mat MosaicBuilder::mosaicFromTiles(const std::string& tileDir, cv::Rect& mosaicBounds, int startX, int startY, int endX, int endY) {
    std::vector<std::tuple<int, int, std::string>> tiles;

    for (const auto& entry : fs::directory_iterator(tileDir)) {
        if (!entry.is_regular_file()) continue;
        std::string filename = entry.path().filename().string();
        if (auto m = ctre::match<R"(tile_(-?\d+)_(-?\d+)\.png)">(filename)) {
            auto ty = std::stoi(std::string{m.get<1>().to_view()});
            auto tx = std::stoi(std::string{m.get<2>().to_view()});
            tiles.emplace_back(tx, ty, entry.path().string());

            startX = std::min(startX, tx);
            startY = std::min(startY, ty);
            endX = std::max(endX, tx);
            endY = std::max(endY, ty);
        }
    }

    if (tiles.empty()) {
        std::cerr << "No tiles found in: " << tileDir << "\n";
        return {};
    }

    int mosaicWidth = (endX - startX + 1) * TILE_SIZE;
    int mosaicHeight = (endY - startY + 1) * TILE_SIZE;
    mosaicBounds = cv::Rect(startX * TILE_SIZE, startY * TILE_SIZE, mosaicWidth, mosaicHeight);

    cv::Mat mosaic(mosaicHeight, mosaicWidth, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    std::mutex mosaicMutex;

    const unsigned numThreads = std::thread::hardware_concurrency();
    size_t tilesPerThread = (tiles.size() + numThreads - 1) / numThreads;

    std::vector<std::future<void>> futures;

    for (unsigned i = 0; i < numThreads; ++i) {
        size_t begin = i * tilesPerThread;
        size_t end = std::min(begin + tilesPerThread, tiles.size());
        if (begin >= end) break;

        futures.emplace_back(std::async(std::launch::async, [&, begin, end]() {
            for (size_t j = begin; j < end; ++j) {
                auto [tx, ty, path] = tiles[j];
                cv::Mat tile = cv::imread(path, cv::IMREAD_UNCHANGED);
                if (tile.empty()) {
                    std::cerr << "Failed to read tile: " << path << "\n";
                    continue;
                }

                int x = (tx - startX) * TILE_SIZE;
                int y = (ty - startY) * TILE_SIZE;

                std::lock_guard<std::mutex> lock(mosaicMutex);
                tile.copyTo(mosaic(cv::Rect(x, y, TILE_SIZE, TILE_SIZE)));
            }
        }));
    }

    for (auto& fut : futures) { fut.get(); }

    return mosaic;
}

// Mosaic from tiles: full size
cv::Mat MosaicBuilder::mosaicFromTiles(const std::string& tileDir, cv::Rect& mosaicBounds) {
    std::map<std::pair<int, int>, std::string> tileMap;

    int minX = INT_MAX, minY = INT_MAX, maxX = INT_MIN, maxY = INT_MIN;

    for (const auto& entry : fs::directory_iterator(tileDir)) {
        if (!entry.is_regular_file()) continue;
        std::string filename = entry.path().filename().string();

        if (auto m = ctre::match<R"(tile_(-?\d+)_(-?\d+)\.png)">(filename)) {
            auto ty = std::stoi(std::string{m.get<1>().to_view()});
            auto tx = std::stoi(std::string{m.get<2>().to_view()});
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

    std::mutex mosaicMutex;

    std::vector<std::future<void>> futures;

    for (const auto& [key, path] : tileMap) {
        futures.emplace_back(std::async(std::launch::async, [&mosaic, &mosaicMutex, path, key, minX, minY]() {
            int tx = key.first;
            int ty = key.second;

            cv::Mat tile = cv::imread(path, cv::IMREAD_UNCHANGED);
            if (tile.empty()) {
                std::cerr << "Failed to read tile: " << path << "\n";
                return;
            }

            int x = (tx - minX) * TILE_SIZE;
            int y = (ty - minY) * TILE_SIZE;

            {
                std::lock_guard<std::mutex> lock(mosaicMutex);
                tile.copyTo(mosaic(cv::Rect(x, y, TILE_SIZE, TILE_SIZE)));
            }
        }));
    }

    for (auto& f : futures) { f.get(); }

    return mosaic;
}

// Mosaic from tiles: understandable mosaic (height x width, px) cropping
cv::Mat MosaicBuilder::mosaicFromTiles(const std::string& tileDir, cv::Rect& mosaicBounds, int mosaicWidth, int mosaicHeight, OffsetOrigin offset) {
    std::map<std::pair<int, int>, std::string> tileMap;
    int minX = INT_MAX, minY = INT_MAX, maxX = INT_MIN, maxY = INT_MIN;

    for (const auto& entry : fs::directory_iterator(tileDir)) {
        if (!entry.is_regular_file()) continue;
        std::string filename = entry.path().filename().string();
        std::smatch match;
        if (auto m = ctre::match<R"(tile_(-?\d+)_(-?\d+)\.png)">(filename)) {
            auto ty = std::stoi(std::string{m.get<1>().to_view()});
            auto tx = std::stoi(std::string{m.get<2>().to_view()});
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

    int totalTilesX = maxX - minX + 1;
    int totalTilesY = maxY - minY + 1;
    int tilesInWidth = std::min(totalTilesX, (mosaicWidth + TILE_SIZE - 1) / TILE_SIZE);
    int tilesInHeight = std::min(totalTilesY, (mosaicHeight + TILE_SIZE - 1) / TILE_SIZE);

    int startTileX, startTileY;
    switch (offset) {
        case OffsetOrigin::TOP_LEFT:
            startTileX = minX;
            startTileY = minY;
            break;
        case OffsetOrigin::TOP_RIGHT:
            startTileX = maxX - tilesInWidth + 1;
            startTileY = minY;
            break;
        case OffsetOrigin::BOTTOM_LEFT:
            startTileX = minX;
            startTileY = maxY - tilesInHeight + 1;
            break;
        case OffsetOrigin::BOTTOM_RIGHT:
            startTileX = maxX - tilesInWidth + 1;
            startTileY = maxY - tilesInHeight + 1;
            break;
        case OffsetOrigin::CENTER:
            startTileX = minX + (totalTilesX - tilesInWidth) / 2;
            startTileY = minY + (totalTilesY - tilesInHeight) / 2;
            break;
        default:
            throw std::runtime_error("Unsupported offset origin.");
    }

    cv::Mat mosaic(tilesInHeight * TILE_SIZE, tilesInWidth * TILE_SIZE, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    for (int y = 0; y < tilesInHeight; ++y) {
        for (int x = 0; x < tilesInWidth; ++x) {
            int tx = startTileX + x;
            int ty = startTileY + y;
            auto it = tileMap.find({tx, ty});
            if (it == tileMap.end()) continue;

            cv::Mat tile = cv::imread(it->second, cv::IMREAD_UNCHANGED);
            if (tile.empty()) {
                std::cerr << "Failed to read tile: " << it->second << "\n";
                continue;
            }

            int dx = x * TILE_SIZE;
            int dy = y * TILE_SIZE;
            tile.copyTo(mosaic(cv::Rect(dx, dy, TILE_SIZE, TILE_SIZE)));
        }
    }

    mosaicBounds = cv::Rect(startTileX * TILE_SIZE, startTileY * TILE_SIZE, tilesInWidth * TILE_SIZE, tilesInHeight * TILE_SIZE);

    return mosaic;
}

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
            if (!fs::exists(path)) continue;

            cv::Mat tile = cv::imread(path, cv::IMREAD_UNCHANGED);
            if (tile.empty() || !tiles_.isValidTile(path, tile)) continue; // slight optimizations

            int x = (tx - minX) * TILE_SIZE;
            int y = (ty - minY) * TILE_SIZE;
            tile.copyTo(mosaic(cv::Rect(x, y, TILE_SIZE, TILE_SIZE)));
        }
    }
    return mosaic;
}

bool MosaicBuilder::addImageToMosaic(const std::string& newImagePath) {
    std::vector<std::string> sharedTags = { 
                EXIFTAGS::IMAGE_WIDTH_TAG, 
                EXIFTAGS::IMAGE_HEIGHT_TAG,
                EXIFTAGS::GPS_LATITUDE_TAG,
                EXIFTAGS::GPS_LONGITUDE_TAG
            };

    auto rawExif = exiftool_.getExifTags(newImagePath, sharedTags);
    auto exif = exiftool_.parseExifValuesToNumbers(rawExif);

    double lat = exif.at(EXIFTAGS::GPS_LATITUDE_TAG);
    double lon = exif.at(EXIFTAGS::GPS_LONGITUDE_TAG);

    auto bestTileKeyOpt = tiles_.findClosestTile(lat, lon); // point for optimization. make TileKey& tempClosestKey_ in TileManager?
    if (!bestTileKeyOpt) return false;

    int height = exif.at(EXIFTAGS::IMAGE_HEIGHT_TAG);
    int width = exif.at(EXIFTAGS::IMAGE_WIDTH_TAG);
    int tileRadius = std::ceil(std::max(height, width) / float(TILE_SIZE)) / 2;
    
    TileKey bestKey = *bestTileKeyOpt;
    TileKey localOriginKey{ bestKey.x - tileRadius, bestKey.y - tileRadius };
    
    cv::Rect localBounds;
    cv::Mat localMosaic = getMosaicAroundTile(bestKey, tileRadius, localBounds);
    if (localMosaic.empty()) {
        #ifdef DEBUG
        std::cerr << "Failed to build local mosaic.\n";
        #endif
        return false;
    }
    
    ImageMatrix mosaicMatrix{localMosaic, "Mosaic"};
    ImageMatrix newImageMatrix = toImageMatrix(newImagePath);
    
    cv::Mat H;

    if (!alignImages(newImageMatrix, mosaicMatrix, H)) return false;

    #ifdef DEBUG
    std::cout << "H: \n" << H << "\n";
    std::cout << "det(H): " << cv::determinant(H) << "\n";
    #endif

    H = H.inv();
    #ifdef DEBUG
    std::cout << "det(H) inv: " << cv::determinant(H) << "\n";
    #endif

    H = H / H.at<double>(2, 2);
    #ifdef DEBUG
    std::cout << "det(H) norm: " << cv::determinant(H) << "\n";
    #endif
    cv::Mat withOffset = tiles_.computeGlobalHomography(localOriginKey, H);
    
    tiles_.applyImage(newImagePath, withOffset, true);
    return true;    
}