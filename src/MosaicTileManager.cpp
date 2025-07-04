#include "../include/mosaic.h"
#include "../include/fmatch.h"
#include "../external/ctre.hpp"
#include <cmath>
#include <filesystem>
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

void TileManager::loadGlobalMetadata() {
    const std::string path = outputDirectory_ + "/" + COORDS_METADATA;
    std::ifstream in(path);
    
    if (!in) return;

    std::string key;
    double double_value;
    int int_value;

    // to debug just check the tiles/tiles_x/coords_metadata.txt
    while (in >> key) {
        if (key == "min_x:") {in >> int_value; globalMinX_ = int_value;}
        else if (key == "min_y:") {in >> int_value; globalMinY_ = int_value;}
        else if (key == "max_x:") {in >> int_value; globalMaxX_ = int_value;}
        else if (key == "max_y:") {in >> int_value; globalMaxY_ = int_value;}
        else if (key == "heading:") {in >> double_value; globalHeading_ = double_value;}
        else if (key == "origin_lat:") { in >> double_value; mosaicOriginLat_ = double_value; }
        else if (key == "origin_lon:") { in >> double_value; mosaicOriginLon_ = double_value; }
        else if (key == "center_x:") {in >> int_value; mosaicCenterOrigin_.x = int_value; }
        else if (key == "center_y:") {in >> int_value; mosaicCenterOrigin_.y = int_value; }
    }
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
        << "origin_lon: " << std::fixed << std::setprecision(10) << mosaicOriginLon_ << "\n"
        << "center_x: " << mosaicCenterOrigin_.x << "\n"
        << "center_y: " << mosaicCenterOrigin_.y << "\n";
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

std::pair<double, double> TileManager::calculateTileGPS(const TileKey& tileKey, const TileKey& centerKey, double centerLat, double centerLon, double gsd) const {
    int dxTiles = centerKey.x - tileKey.x;
    int dyTiles = centerKey.y - tileKey.y;

    // convert tile unit offset to meters.
    double offsetXmMosaic = static_cast<double>(dxTiles) * TILE_SIZE * gsd;
    double offsetYmMosaic = static_cast<double>(dyTiles) * TILE_SIZE * gsd;
    
    double headingRad = globalHeading_ * DEG_TO_RAD;

    double cosHeading = std::cos(headingRad);
    double sinHeading = std::sin(headingRad);

    double trueEastOffsetM = (offsetXmMosaic * cosHeading) + (-offsetYmMosaic * sinHeading);
    double trueNorthOffsetM = (offsetXmMosaic * sinHeading) + (-offsetYmMosaic * cosHeading);
    
    double centerLatRad = centerLat * DEG_TO_RAD;
    double N = WGS84_A / std::sqrt(1.0 - WGS84_E_SQUARED * std::sin(centerLatRad) * std::sin(centerLatRad));

    // N * cos(latitude) * (PI / 180)
    double metersAtCenterRefLon = N * std::cos(centerLatRad) * DEG_TO_RAD; 
    
    // M = a(1-e^2) / (1-e^2*sin^2(phi))^(3/2)
    double M = WGS84_A * (1.0 - WGS84_E_SQUARED) / std::pow(1.0 - WGS84_E_SQUARED * std::sin(centerLatRad) * std::sin(centerLatRad), 1.5);
    double metersAtCenterLat = M * DEG_TO_RAD;

    double lat = centerLat + (trueNorthOffsetM / metersAtCenterLat);
    double lon = centerLon + (trueEastOffsetM / metersAtCenterRefLon);

    #ifdef DEBUG
    std::cout << std::fixed << std::setprecision(10); 
    std::cout << "DEBUG: calculateTileGPS for TileKey (" << tileKey.x << "," << tileKey.y << ") with Ref=(" << centerKey.x << "," << centerKey.y << ") and RefGPS=(" << centerLat << "," << centerLon << "):\n"; 
    std::cout << "  dxTiles=" << dxTiles << ", dyTiles=" << dyTiles << "\n";
    std::cout << "  offsetXmMosaic=" << offsetXmMosaic << ", offsetYmMosaic=" << offsetYmMosaic << "\n";
    std::cout << "  trueEastOffsetM=" << trueEastOffsetM << ", trueNorthOffsetM=" << trueNorthOffsetM << "\n";
    std::cout << "  Calculated Lat=" << lat << ", Lon=" << lon << "\n"; 
    #endif

    return {lat, lon}; 
}

double TileManager::estimateGSD(const std::map<std::string, double>& exif) const {
    if (exif.count(EXIFTAGS::FOCAL_LENGTH_TAG) && exif.count(EXIFTAGS::IMAGE_WIDTH_TAG) && exif.at(EXIFTAGS::FOCAL_LENGTH_TAG) != 0.0 && exif.at(EXIFTAGS::IMAGE_WIDTH_TAG) != 0.0) {
        double alt = exif.count(EXIFTAGS::GPS_ALTITUDE_TAG) ? exif.at(EXIFTAGS::GPS_ALTITUDE_TAG) : 0.0;
        double flen = exif.at(EXIFTAGS::FOCAL_LENGTH_TAG);
        double width = exif.at(EXIFTAGS::IMAGE_WIDTH_TAG); // px
        return (3.674 * alt) / (flen * width);
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

void TileManager::assignMetadata(const std::string& imagePath, const double lat, const double lon, const double alt, const double flen, const double gpsDir) const {
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
    
    bool hasExistingGps = false;
    std::map<std::string, double> parsedExistingExif;
    std::vector<std::string> existingTags = {EXIFTAGS::GPS_LATITUDE_TAG, EXIFTAGS::GPS_LONGITUDE_TAG, EXIFTAGS::GPS_ALTITUDE_TAG, EXIFTAGS::FOCAL_LENGTH_TAG, EXIFTAGS::GPS_IMG_DIRECTION_TAG};
    auto rawExistingExif = exiftool_.getExifTags(path, existingTags);
    parsedExistingExif = exiftool_.parseExifValuesToNumbers(rawExistingExif);
    if (parsedExistingExif.count(EXIFTAGS::GPS_LATITUDE_TAG) && parsedExistingExif.count(EXIFTAGS::GPS_LONGITUDE_TAG)) {
        hasExistingGps = true;
    }
    
    cv::imwrite(path, tile);

    if (!hasExistingGps) {
        assignMetadata(path, 
            lat, 
            lon, 
            exif.count(EXIFTAGS::GPS_ALTITUDE_TAG) ? exif.at(EXIFTAGS::GPS_ALTITUDE_TAG) : 0.0, 
            exif.count(EXIFTAGS::FOCAL_LENGTH_TAG) ? exif.at(EXIFTAGS::FOCAL_LENGTH_TAG) : 0.0, 
            exif.count(EXIFTAGS::GPS_IMG_DIRECTION_TAG) ? exif.at(EXIFTAGS::GPS_IMG_DIRECTION_TAG) : 0.0
        );
        #ifdef DEBUG
        std::cout << "DEBUG: Saved tile " << path << " with NEW GPS Lat=" << lat << ", Lon=" << lon 
                  << ", GPSImgDirection=" << (exif.count(EXIFTAGS::GPS_IMG_DIRECTION_TAG) ? exif.at(EXIFTAGS::GPS_IMG_DIRECTION_TAG) : 0.0) << " (image written, metadata assigned)\n";
        #endif
    } else {
        assignMetadata(path, 
            parsedExistingExif.at(EXIFTAGS::GPS_LATITUDE_TAG), 
            parsedExistingExif.at(EXIFTAGS::GPS_LONGITUDE_TAG),
            parsedExistingExif.at(EXIFTAGS::GPS_ALTITUDE_TAG), 
            parsedExistingExif.at(EXIFTAGS::FOCAL_LENGTH_TAG), 
            parsedExistingExif.at(EXIFTAGS::GPS_IMG_DIRECTION_TAG)
        );
    }
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

bool TileManager::isMostlyTransparent(const cv::Mat& tileMat, double threshold = 0.4) const { // way more flexible than isTransparent
    if (tileMat.channels() != 4) return false;

    const int totalPixels = tileMat.rows * tileMat.cols;
    const uchar* pixelData = tileMat.ptr<uchar>(0);

    int transparentCount = 0;
    const uchar alphaThreshold = 10;

    for (int i = 0; i < totalPixels; ++i) {
        if (pixelData[i * 4 + 3] < alphaThreshold) {
            transparentCount++;
            if (static_cast<double>(transparentCount) / totalPixels >= threshold) {
                return true;
            }
        }
    }
    return false;
}

double TileManager::findTileDistance(const std::string& tilePath, double latToCompare, double lonToCompare) {
    double lat = exiftool_.parseExifGPS(exiftool_.getExifTag(tilePath, EXIFTAGS::GPS_LATITUDE_TAG));
    double lon = exiftool_.parseExifGPS(exiftool_.getExifTag(tilePath, EXIFTAGS::GPS_LONGITUDE_TAG));

    double dLat = (lat - latToCompare) * DEG_TO_RAD;
    double dLon = (lon - lonToCompare) * DEG_TO_RAD;

    // Haversine formula
    double a = std::sin(dLat/2) * std::sin(dLat/2) + 
               std::cos(latToCompare * DEG_TO_RAD) * std::cos(lat * DEG_TO_RAD) * 
               std::sin(dLon/2) * std::sin(dLon/2);

    double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1-a)); // central angle

    return WGS84_A * c; // distance in meters
}

std::optional<TileKey> TileManager::findClosestTile(double lat, double lon) { // with BFS now
    if (lat == 0.0 || lon == 0.0) {
        #ifdef DEBUG
        std::cerr << "[WARN] Image has no valid GPS coordinates.\n";
        #endif
        return std::nullopt;
    }

    std::set<TileKey> visited;
    std::queue<TileKey> queue;
    std::optional<TileKey> bestKey;
    double bestDist = DBL_MAX;

    // BFS perimeter
    for (int x = globalMinX_; x <= globalMaxX_; ++x) {
        queue.push(TileKey{x, globalMinY_});
        queue.push(TileKey{x, globalMaxY_});
    }
    for (int y = globalMinY_ + 1; y < globalMaxY_; ++y) {
        queue.push(TileKey{globalMinX_, y});
        queue.push(TileKey{globalMaxX_, y});
    }

    while (!queue.empty()) {
        TileKey key = queue.front();
        queue.pop();

        if (visited.contains(key)) continue;
        visited.insert(key);

        std::string tilePath = getTilePath(key);
        if (!fs::exists(tilePath)) continue;

        cv::Mat tileMat = cv::imread(tilePath, cv::IMREAD_UNCHANGED);
        if (tileMat.empty()) continue;

        if (isMostlyTransparent(tileMat)) continue;

        double dist = findTileDistance(tilePath, lat, lon);
        if (dist < bestDist) {
            bestDist = dist;
            bestKey = key;
        }

        for (const auto& [dx, dy] : std::array<std::pair<int, int>, 4>{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}}) {
            TileKey neighbor{key.x + dx, key.y + dy};
            if (!visited.contains(neighbor)) {
                queue.push(neighbor);
            }
        }
    }

    if (!bestKey) {
        #ifdef DEBUG
        std::cerr << "[ERROR] No suitable tile found.\n";
        #endif
        return std::nullopt;
    }

    #ifdef DEBUG
    std::cout << "Closest tile (BFS frontier): " << getTilePath(*bestKey) << '\n';
    #endif

    tempClosestKey_ = *bestKey;
    return bestKey;
}

void TileManager::blendOntoTile(cv::Mat& tile, const cv::Mat& patch, const cv::Rect& roi, int featheringPx) {
    if (patch.empty() || tile.empty() || patch.channels() != 4 || tile.channels() != 4) {
        #ifdef DEBUG
        std::cerr << "[WARN] blendOntoTile: Invalid input image or patch. Skipping blending.\n";
        #endif
        return;
    }

    cv::Rect validROI = roi & cv::Rect(0, 0, tile.cols, tile.rows);
    if (validROI.empty() || validROI.width != patch.cols || validROI.height != patch.rows) {
        std::vector<cv::Mat> patchChannels;
        cv::split(patch, patchChannels);
        patch.copyTo(tile(roi), patchChannels[3]);
        return;
    }

    std::vector<cv::Mat> tileChannels(4), patchChannels(4);
    cv::split(tile(validROI), tileChannels);
    cv::split(patch, patchChannels);

    cv::Mat patchAlpha = patchChannels[3];
    cv::Mat tileAlpha = tileChannels[3];

    cv::Mat patchOpaqueMask;
    cv::compare(patchAlpha, 0, patchOpaqueMask, cv::CMP_GT);

    cv::Mat distTransform;
    cv::distanceTransform(patchOpaqueMask, distTransform, cv::DIST_L2, 3);

    cv::Mat blendWeightFloat;
    distTransform.convertTo(blendWeightFloat, CV_32F, 1.0f / featheringPx);
    cv::threshold(blendWeightFloat, blendWeightFloat, 1.0, 1.0, cv::THRESH_TRUNC);

    // convertion of float weights [0.0–1.0] to 8-bit [0–255]
    cv::Mat blendWeight;
    blendWeightFloat.convertTo(blendWeight, CV_8U, 255.0);

    const int rows = patch.rows;
    const int cols = patch.cols;

    for (int r = 0; r < rows; ++r) {
        const uchar* wPtr = blendWeight.ptr<uchar>(r);
        const uchar* pAlpha = patchAlpha.ptr<uchar>(r);
        uchar* tAlpha = tileAlpha.ptr<uchar>(r);

        uchar* pR = patchChannels[0].ptr<uchar>(r);
        uchar* pG = patchChannels[1].ptr<uchar>(r);
        uchar* pB = patchChannels[2].ptr<uchar>(r);

        uchar* tR = tileChannels[0].ptr<uchar>(r);
        uchar* tG = tileChannels[1].ptr<uchar>(r);
        uchar* tB = tileChannels[2].ptr<uchar>(r);

        for (int c = 0; c < cols; ++c) {
            float w = wPtr[c] / 255.0f;
            float aNew = (pAlpha[c] / 255.0f) * w;
            float aOld = tAlpha[c] / 255.0f;
            float aFinal = aNew + aOld * (1.0f - aNew);

            if (aFinal > 1e-6f) {
                float invA = 1.0f / aFinal;
                tR[c] = static_cast<uchar>(cv::saturate_cast<uchar>((pR[c] * aNew + tR[c] * aOld * (1.0f - aNew)) * invA));
                tG[c] = static_cast<uchar>(cv::saturate_cast<uchar>((pG[c] * aNew + tG[c] * aOld * (1.0f - aNew)) * invA));
                tB[c] = static_cast<uchar>(cv::saturate_cast<uchar>((pB[c] * aNew + tB[c] * aOld * (1.0f - aNew)) * invA));
                tAlpha[c] = static_cast<uchar>(cv::saturate_cast<uchar>(aFinal * 255.0f));
            } else {
                tR[c] = tG[c] = tB[c] = tAlpha[c] = 0;
            }
        }
    }
    cv::merge(tileChannels, tile(validROI));
}

void TileManager::applyImageWarpOnce(const cv::Mat& img, const cv::Mat& homography, double gsd, std::map<std::string, double>& exif) {
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

    int width = static_cast<int>(std::ceil(maxX - minX));
    int height = static_cast<int>(std::ceil(maxY - minY));

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

    // now image center depends on the tile that was chosen by GPS
    TileKey bestKey;
    if (tempClosestKey_ == TileKey{0, 0}) {
        auto closest = findClosestTile(exif.at(EXIFTAGS::GPS_LATITUDE_TAG), exif.at(EXIFTAGS::GPS_LONGITUDE_TAG));
        bestKey = *closest;
    } else { 
        bestKey.x = tempClosestKey_.x;
        bestKey.y = tempClosestKey_.y;
    }

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

            cv::Rect patchTileROI(0, 0, patch.cols, patch.rows); 
            blendOntoTile(tile, patch, patchTileROI, FEATHERING_SIZE);

            auto [tileLat, tileLon] = calculateTileGPS(key, bestKey, exif.at(EXIFTAGS::GPS_LATITUDE_TAG), exif.at(EXIFTAGS::GPS_LONGITUDE_TAG), gsd);
            saveTile(key, tile, tileLat, tileLon, exif);
            updateGlobalBounds(key);
        }
    }
    tempClosestKey_.x = 0;
    tempClosestKey_.y = 0;
}

void TileManager::applyImagePerTile(const cv::Mat& img, const cv::Mat& homography, double gsd, std::map<std::string, double>& exif) {
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

            cv::Rect patchTileROI(0, 0, patch.cols, patch.rows); 
            blendOntoTile(tile, patch, patchTileROI, FEATHERING_SIZE);

            auto [tileLat, tileLon] = calculateTileGPS(key, mosaicCenterOrigin_, mosaicOriginLat_, mosaicOriginLon_, gsd);
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
        EXIFTAGS::IMAGE_HEIGHT_TAG,
        EXIFTAGS::IMAGE_WIDTH_TAG,
        EXIFTAGS::GPS_IMG_DIRECTION_TAG,
        EXIFTAGS::GPS_LATITUDE_TAG,
        EXIFTAGS::GPS_LONGITUDE_TAG
    };

    auto rawExifMap = exiftool_.getExifTags(imagePath, sharedTags);
    auto parsedExif = exiftool_.parseExifValuesToNumbers(rawExifMap);

    #ifdef DEBUG
    std::cout << "DEBUG: Image path: " << imagePath << "\n";
    std::cout << "  GPS_ALTITUDE_TAG: " << std::fixed << std::setprecision(10) << (parsedExif.count(EXIFTAGS::GPS_ALTITUDE_TAG) ? parsedExif.at(EXIFTAGS::GPS_ALTITUDE_TAG) : 0.0) << "\n";
    std::cout << "  FOCAL_LENGTH_TAG: " << std::fixed << std::setprecision(10) << (parsedExif.count(EXIFTAGS::FOCAL_LENGTH_TAG) ? parsedExif.at(EXIFTAGS::FOCAL_LENGTH_TAG) : 0.0) << "\n";
    std::cout << "  IMAGE_WIDTH_TAG: " << std::fixed << std::setprecision(10) << (parsedExif.count(EXIFTAGS::IMAGE_WIDTH_TAG) ? parsedExif.at(EXIFTAGS::IMAGE_WIDTH_TAG) : 0.0) << "\n";
    std::cout << "DEBUG: Parsed Image Lat: " << std::fixed << std::setprecision(10) << (parsedExif.count(EXIFTAGS::GPS_LATITUDE_TAG) ? parsedExif.at(EXIFTAGS::GPS_LATITUDE_TAG) : 0.0) << "\n";
    std::cout << "DEBUG: Parsed Image Lon: " << std::fixed << std::setprecision(10) << (parsedExif.count(EXIFTAGS::GPS_LONGITUDE_TAG) ? parsedExif.at(EXIFTAGS::GPS_LONGITUDE_TAG) : 0.0) << "\n";
    std::cout << "DEBUG: Parsed GPSImgDirection: " << std::fixed << std::setprecision(2) << (parsedExif.count(EXIFTAGS::GPS_IMG_DIRECTION_TAG) ? parsedExif.at(EXIFTAGS::GPS_IMG_DIRECTION_TAG) : 0.0) << "\n";
    #endif

    if (globalMinX_ == 0 && globalMinY_ == 0 && globalMaxX_ == 0 && globalMaxY_ == 0) {
        if (parsedExif.count(EXIFTAGS::GPS_IMG_DIRECTION_TAG)) {
            globalHeading_ = parsedExif.at(EXIFTAGS::GPS_IMG_DIRECTION_TAG);
            mosaicOriginLat_ = parsedExif.at(EXIFTAGS::GPS_LATITUDE_TAG);
            mosaicOriginLon_ = parsedExif.at(EXIFTAGS::GPS_LONGITUDE_TAG);

            // predicting center tile for correct gps offset
            int centerTileX = static_cast<int>(parsedExif.at(EXIFTAGS::IMAGE_WIDTH_TAG) / 2.0);
            int centerTileY = static_cast<int>(parsedExif.at(EXIFTAGS::IMAGE_HEIGHT_TAG) / 2.0);
    
            mosaicCenterOrigin_.x = static_cast<int>(std::floor(static_cast<double>(centerTileX) / TILE_SIZE));
            mosaicCenterOrigin_.y = static_cast<int>(std::floor(static_cast<double>(centerTileY) / TILE_SIZE));

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