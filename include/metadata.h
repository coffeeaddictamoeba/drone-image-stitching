#pragma once

#include <string>
#include <map>
#include <unistd.h>

class ExifToolPipe {
public:
    ExifToolPipe();
    ~ExifToolPipe();

    bool sendCommand(const std::string& imagePath);
    std::map<std::string, std::string> getLastExifData();

private:
    int writeFd = -1;
    int readFd = -1;
    pid_t childPid = -1;

    std::string readResponse();
};

struct CameraMetadata {
    double focalLengthMM = 0.0;
    double sensorWidthMM = 0.0;
    double sensorHeightMM = 0.0;
    int imageWidth = 0;
    int imageHeight = 0;
    double altitude = 0.0;
    double yawDeg = 0.0;
    double pitchDeg = 0.0;
    double rollDeg = 0.0;
};

class MetadataExtractor {
public:
    MetadataExtractor(ExifToolPipe& tool, const std::string& path);
    CameraMetadata parseMetadata();

private:
    ExifToolPipe& exifTool;
    std::string imagePath;
    std::map<std::string, std::string> exifData;

    double getExifValueAsDouble(const std::string& key, double defaultValue);
    int getExifValueAsInt(const std::string& key, int defaultValue);
};
