#include "../include/mosaic.h"
#include "../include/fmatch.h"
#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>

///////////////////// MosaicTileManager /////////////////////

MosaicTileManager::MosaicTileManager(const std::string& outputDir)
    : outputDirectory_(outputDir)
{
    std::filesystem::create_directories(outputDirectory_);
}

TileKey MosaicTileManager::getTileKeyForPoint(int x, int y) const {
    return { x / TILE_SIZE, y / TILE_SIZE };
}

std::string MosaicTileManager::getTilePath(const TileKey& key) const {
    return outputDirectory_ + "/tile_" + std::to_string(key.y) + "_" + std::to_string(key.x) + ".png";
}

cv::Mat MosaicTileManager::loadTile(const TileKey& key) const {
    std::string path = getTilePath(key);
    if (std::filesystem::exists(path)) {
        cv::Mat tile = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (!tile.empty()) return tile;
    }
    return cv::Mat(TILE_SIZE, TILE_SIZE, CV_8UC4, cv::Scalar(0, 0, 0, 0));
}

void MosaicTileManager::saveTile(const TileKey& key, const cv::Mat& tile) const {
    std::string path = getTilePath(key);
    cv::imwrite(path, tile);
}

cv::Mat MosaicTileManager::warpTileRegion(const cv::Mat& input,
                                          const cv::Mat& H,
                                          const cv::Rect& tileRect) const
{
    cv::Mat Hinv = H.inv();
    std::vector<cv::Point2f> dstPts = {
        {(float)tileRect.x, (float)tileRect.y},
        {(float)(tileRect.x + tileRect.width), (float)tileRect.y},
        {(float)(tileRect.x + tileRect.width), (float)(tileRect.y + tileRect.height)},
        {(float)tileRect.x, (float)(tileRect.y + tileRect.height)}
    };
    std::vector<cv::Point2f> srcPts;
    cv::perspectiveTransform(dstPts, srcPts, Hinv);
    std::vector<cv::Point2f> dstQuad = {
        {0, 0},
        {(float)tileRect.width, 0},
        {(float)tileRect.width, (float)tileRect.height},
        {0, (float)tileRect.height}
    };
    cv::Mat tileH = cv::getPerspectiveTransform(srcPts, dstQuad);
    cv::Mat tile;
    cv::warpPerspective(input, tile, tileH,
                        cv::Size(tileRect.width, tileRect.height),
                        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));
    return tile;
}

void MosaicTileManager::applyImage(const std::string& imagePath,
                                   const cv::Mat& homography)
{
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
            saveTile(key, tile);
        }
    }
}

///////////////////// MosaicBuilder /////////////////////

MosaicBuilder::MosaicBuilder(const std::string& refImagePath,
                             const std::string& targetImagePath,
                             ExifToolPipe& tool,
                             MosaicTileManager& tileManager)
    : refImagePath_(refImagePath),
      targetImagePath_(targetImagePath),
      exiftool_(tool),
      tiles_(tileManager)
{
}

bool MosaicBuilder::loadImages() {
    ref_ = ImageMatrix{cv::imread(refImagePath_, cv::IMREAD_UNCHANGED), refImagePath_};
    target_ = ImageMatrix{cv::imread(targetImagePath_, cv::IMREAD_UNCHANGED), targetImagePath_};
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

    tiles_.applyImage(refImagePath_, cv::Mat::eye(3, 3, CV_64F));

    if (!alignImages(ref_, target_, homography_)) {
        return false;
    }

    tiles_.applyImage(targetImagePath_, homography_);
    return true;
}

// Mosaic from tiles: fixed size
cv::Mat MosaicBuilder::mosaicFromTiles(
            const std::string& tileDir, 
            cv::Rect& mosaicBounds, 
            int startX, 
            int startY, 
            int endX,
            int endY
    ) {
    std::map<std::pair<int, int>, std::string> tileMap;
    std::regex tileRegex(R"(tile_(\-?\d+)\_(\-?\d+)\.png)"); // searches for pattern "tile_y_x.png"

    for (const auto& entry : fs::directory_iterator(tileDir)) {
        if (!entry.is_regular_file()) continue;
        std::string filename = entry.path().filename().string();

        std::smatch match;
        if (std::regex_match(filename, match, tileRegex)) {
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
    std::regex tileRegex(R"(tile_(\-?\d+)\_(\-?\d+)\.png)"); // searches for pattern "tile_y_x.png"

    int minX = INT_MAX, minY = INT_MAX, maxX = INT_MIN, maxY = INT_MIN;

    for (const auto& entry : fs::directory_iterator(tileDir)) {
        if (!entry.is_regular_file()) continue;
        std::string filename = entry.path().filename().string();

        std::smatch match;
        if (std::regex_match(filename, match, tileRegex)) {
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