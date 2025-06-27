#include "../include/mosaic.h"
#include "../include/fmatch.h"
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

        std::smatch match;
        if (std::regex_match(filename, match, TILE_REGEX)) {
            int ty = std::stoi(match[1]);
            int tx = std::stoi(match[2]);
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
            #ifdef DEBUG
            std::cout << "Skipping fully transparent tile: " << tilePath << "\n";
            #endif
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
        #ifdef DEBUG
        std::cout << "Skipping visually uniform tile: " << tilePath << "\n";
        #endif
        return false;
    }
    return true;
}

double MosaicBuilder::findTileDistance(std::string tilePath, double latToCompare, double lonToCompare) {
    double lat = exiftool_.parseExifGPS(exiftool_.getExifTag(tilePath, IMG_GPS_LAT));
    double lon = exiftool_.parseExifGPS(exiftool_.getExifTag(tilePath, IMG_GPS_LON));

    double dLat = (lat - latToCompare) * DEG_TO_RAD;
    double dLon = (lon - lonToCompare) * DEG_TO_RAD;

    // Haversine formula
    double a = std::sin(dLat/2) * std::sin(dLat/2) + 
               std::cos(latToCompare * DEG_TO_RAD) * std::cos(lat * DEG_TO_RAD) * 
               std::sin(dLon/2) * std::sin(dLon/2);

    double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1-a)); // central angle

    return R_M * c; // distance in meters
}

// function for searching best tile for feature matching. works well but quite unnecessary for now
std::optional<TileKey> MosaicBuilder::findBestMatchingTileInRadius(const cv::Mat& image, const TileKey& centerTile, int radius) {
    const int maxThreads = 3; // test with other amounts
    std::vector<std::future<std::pair<TileKey, int>>> futures;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            TileKey candidate{centerTile.x + dx, centerTile.y + dy};
            std::string tilePath = tiles_.getTilePath(candidate);
            #ifdef DEBUG
            std::cout << "Evaluating tile " << tilePath << '\n';
            #endif

            if (!fs::exists(tilePath) || !isValidTile(tilePath)) continue;

            futures.emplace_back(std::async(std::launch::async, [=]() -> std::pair<TileKey, int> {
                FeatureMatcher matcher{exiftool_, "SIFT"};
                cv::Mat H;
                std::vector<cv::DMatch> matches;

                ImageMatrix imgNew{image, "New Image"};
                ImageMatrix imgTile = toImageMatrix(tilePath);
                if (!alignImages(imgNew, imgTile, H)) return {centerTile, 0}; // fix when the best gps tile returns no matches

                bool success = matcher.computeHomography(imgNew, imgTile, H, matches);
                return {candidate, success ? static_cast<int>(matches.size()) : 0};
            }));

            if ((int)futures.size() >= maxThreads * 2) break;
        }
    }

    int bestMatchCount = -1;
    std::optional<TileKey> bestTile;
    for (auto& fut : futures) {
        auto [tile, count] = fut.get();
        if (count > bestMatchCount) {
            bestMatchCount = count;
            bestTile = tile;
        }
    }

    if (bestTile) {
        #ifdef DEBUG
        std::cout << "Best tile based on feature match: " << tiles_.getTilePath(*bestTile) << " with " << bestMatchCount << " good matches.\n";
        #endif
    } else {
        #ifdef DEBUG
        std::cerr << "[WARN] No tile produced valid feature matches.\n";
        #endif
    }

    return bestTile;
}

std::optional<TileKey> MosaicBuilder::findClosestTile(double lat, double lon) {
    if (lat == 0.0 || lon == 0.0) {
        #ifdef DEBUG
        std::cerr << "[WARN] Image has no valid GPS coordinates.\n";
        #endif
        return std::nullopt;
    }

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

        double dist = findTileDistance(tilePath, lat, lon);
        if (dist < bestDist) {
            bestDist = dist;
            bestKey = TileKey{tx, ty};
        }
    }

    if (!bestKey) {
        #ifdef DEBUG
        std::cerr << "[ERROR] No suitable tile found.\n";
        #endif
        return std::nullopt;
    }

    #ifdef DEBUG
    std::cout << "Tile candidate (GPS closest): " << tiles_.getTilePath(*bestKey) << '\n';
    #endif
    return bestKey;
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
    std::vector<std::string> sharedTags = {IMG_GPS_LAT, IMG_GPS_LON, IMG_HEIGHT_TAG, IMG_WIDTH_TAG};

    auto rawExif = exiftool_.getExifTags(newImagePath, sharedTags);
    auto exif = exiftool_.parseExifValuesToNumbers(rawExif);

    double lat = exiftool_.parseExifGPS(rawExif.at(IMG_GPS_LAT));
    double lon = exiftool_.parseExifGPS(rawExif.at(IMG_GPS_LON));

    auto bestTileKeyOpt = findClosestTile(lat, lon);
    if (!bestTileKeyOpt) return false;

    int height = exif.at(IMG_HEIGHT_TAG);
    int width = exif.at(IMG_WIDTH_TAG);
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