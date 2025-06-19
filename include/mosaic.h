#pragma once

#include <string>
#include <opencv4/opencv2/opencv.hpp>
#include <filesystem>
#include "metadata.h"
#include "fmatch.h"

constexpr int TILE_SIZE = 512;

struct TileKey {
    int x, y;
    bool operator<(const TileKey& other) const {
        return std::tie(y, x) < std::tie(other.y, other.x);
    }
};

class MosaicTileManager {
public:
    MosaicTileManager(const std::string& outputDir);

    TileKey getTileKeyForPoint(int x, int y) const;
    std::string getTilePath(const TileKey& key) const;
    cv::Mat loadTile(const TileKey& key) const;
    void saveTile(const TileKey& key, const cv::Mat& tile) const;

    void applyImage(const std::string& imagePath, const cv::Mat& homography);

private:
    void blendOntoTile(cv::Mat& tile, const cv::Mat& patch, const cv::Rect& roi);
    cv::Mat warpTileRegion(const cv::Mat& input, const cv::Mat& H, const cv::Rect& tileRect) const;

    std::string outputDirectory_;
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

private:
    std::string refImagePath_;
    std::string targetImagePath_;
    ImageMatrix ref_;
    ImageMatrix target_;
    ExifToolPipe& exiftool_;
    MosaicTileManager& tiles_;
    cv::Mat homography_;
};
