#pragma once
#include <string>
#include <unordered_map>

struct CameraMetadata {
    double focalLengthMM = 35.0;
    double sensorWidthMM = 6.3;
    double sensorHeightMM = 4.7;
    int imageWidth = 4000;
    int imageHeight = 3000;
    double altitude = 100.0;
    double yawDeg = 0.0;
    double pitchDeg = 0.0;
    double rollDeg = 0.0;
};

class MetadataExtractor {
public:
    explicit MetadataExtractor(const std::string& imagePath);
    CameraMetadata parseMetadata();

private:
    std::string imagePath;
    std::unordered_map<std::string, std::string> exifData;

    void runExifTool();
    double getExifValueAsDouble(const std::string& key, double defaultValue);
    int getExifValueAsInt(const std::string& key, int defaultValue);
};
