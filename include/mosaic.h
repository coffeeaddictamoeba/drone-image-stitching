#pragma once

#include <cmath>
#include <optional>
#include <regex>
#include <string>
#include <opencv4/opencv2/opencv.hpp>
#include <filesystem>
#include "metadata.h"
#include "fmatch.h"

constexpr int TILE_SIZE = 512;

constexpr double M_PER_DEGREE_LATITUDE = 111320.0;
constexpr double DEG_TO_RAD = M_PI / 180.0;
constexpr double R_M = 6371000.0; // Earth radius in meters

const std::regex TILE_REGEX(R"(tile_(\-?\d+)\_(\-?\d+)\.png)"); // searches for pattern "tile_y_x.png"

struct TileKey {
    int x, y;

    TileKey() = default;
    TileKey(int x_, int y_) : x(x_), y(y_) {}

    bool operator<(const TileKey& other) const {
        return std::tie(y, x) < std::tie(other.y, other.x);
    }
};

class MosaicTileManager {
    public:
        MosaicTileManager(const std::string& outputDir, ExifToolPipe& tool);

        std::string getOutputDirectory() const;
        std::string getTilePath(const TileKey& key) const;
        TileKey getTileKeyForPoint(int x, int y) const;

        std::pair<double, double> extractGPS(const std::string& imagePath) const;
        double estimateGSD(const std::string& imagePath) const;
        std::pair<double, double> calculateTileGPS(
            const TileKey& tileKey,
            const TileKey& centerTile,
            double centerLat,
            double centerLon,
            double gsd) const;

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
        bool loadImages(std::string refImagePath, std::string targetImagePath);
        bool alignImages(const ImageMatrix& src, const ImageMatrix& dst, cv::Mat& H);
        bool stitchToTiles();
        bool stitchToTiles(std::string refImagePath, std::string targetImagePath);
        cv::Mat mosaicFromTiles(const std::string& tileDir, cv::Rect& mosaicBounds);
        cv::Mat mosaicFromTiles(
            const std::string& tileDir, 
            cv::Rect& mosaicBounds, 
            int startX, 
            int startY, 
            int endX,
            int endY
        );
        bool addImageToMosaic(const std::string& newImagePath);

    private:
        std::string refImagePath_;
        std::string targetImagePath_;
        ImageMatrix ref_;
        ImageMatrix target_;
        ExifToolPipe& exiftool_;
        MosaicTileManager& tiles_;
        cv::Mat homography_;

        ImageMatrix toImageMatrix(std::string imagePath);
        bool isValidTile(std::string tilePath);

        double findTileDistance(std::string tilePath, double latToCompare, double lonToCompare);
        std::optional<TileKey> findClosestTile(const std::string& imagePath);
};
