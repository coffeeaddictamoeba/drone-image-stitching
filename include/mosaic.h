#pragma once

#include <optional>
#include <regex>
#include <string>
#include <opencv4/opencv2/opencv.hpp>
#include <filesystem>
#include "metadata.h"
#include "fmatch.h"

constexpr int TILE_SIZE = 512;
const std::regex TILE_REGEX(R"(tile_(\-?\d+)\_(\-?\d+)\.png)"); // searches for pattern "tile_y_x.png"

struct TileKey {
    int x, y;
    bool operator<(const TileKey& other) const {
        return std::tie(y, x) < std::tie(other.y, other.x);
    }
};

class MosaicTileManager {
public:
    MosaicTileManager(const std::string& outputDir, ExifToolPipe& tool);

    TileKey getTileKeyForPoint(int x, int y) const;
    std::string getOutputDirectory() const;
    std::string getTilePath(const TileKey& key) const;
    std::pair<double, double> extractGPS(const std::string& imagePath) const;
    cv::Mat loadTile(const TileKey& key) const;
    void saveTile(const TileKey& key, const cv::Mat& tile, const double lat, const double lon) const;

    void applyImage(const std::string& imagePath, const cv::Mat& homography);

private:
    void blendOntoTile(cv::Mat& tile, const cv::Mat& patch, const cv::Rect& roi);
    cv::Mat warpTileRegion(const cv::Mat& input, const cv::Mat& H, const cv::Rect& tileRect) const;

    std::string outputDirectory_;
    ExifToolPipe& exiftool_;
};

class MosaicBuilder {
public:
    MosaicBuilder(const std::string& refImagePath,
                  const std::string& targetImagePath,
                  ExifToolPipe& tool,
                  MosaicTileManager& tileManager);

    bool loadImages();
    bool alignImages(const ImageMatrix& src, const ImageMatrix& dst, cv::Mat& H);
    bool stitchToTiles();
    cv::Mat mosaicFromTiles(const std::string& tileDir, cv::Rect& mosaicBounds);
    cv::Mat mosaicFromTiles(
        const std::string& tileDir, 
        cv::Rect& mosaicBounds, 
        int startX, 
        int startY, 
        int endX,
        int endY
    );

    std::optional<TileKey> findClosestTile(const std::string& imagePath); // make private

private:
    std::string refImagePath_;
    std::string targetImagePath_;
    ImageMatrix ref_;
    ImageMatrix target_;
    ExifToolPipe& exiftool_;
    MosaicTileManager& tiles_;
    cv::Mat homography_;
};
