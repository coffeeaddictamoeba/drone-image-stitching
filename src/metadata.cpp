#include "../include/metadata.h"
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <iostream>

MetadataExtractor::MetadataExtractor(const std::string& path) : imagePath(path) {}

void MetadataExtractor::runExifTool() {
    std::string cmd = "exiftool -n \"" + imagePath + "\" > temp_meta.txt";
    std::system(cmd.c_str());

    std::ifstream infile("temp_meta.txt");
    std::string line;
    while (std::getline(infile, line)) {
        auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);
            key.erase(key.find_last_not_of(" \t") + 1);
            val.erase(0, val.find_first_not_of(" \t"));
            exifData[key] = val;
        }
    }
    std::remove("temp_meta.txt");
}

double MetadataExtractor::getExifValueAsDouble(const std::string& key, double defaultValue) {
    try {
        return exifData.count(key) ? std::stod(exifData[key]) : defaultValue;
    } catch (...) {
        return defaultValue;
    }
}

int MetadataExtractor::getExifValueAsInt(const std::string& key, int defaultValue) {
    try {
        return exifData.count(key) ? std::stoi(exifData[key]) : defaultValue;
    } catch (...) {
        return defaultValue;
    }
}

CameraMetadata MetadataExtractor::parseMetadata() {
    runExifTool();

    CameraMetadata meta;
    meta.focalLengthMM = getExifValueAsDouble("Focal Length", meta.focalLengthMM);
    meta.sensorWidthMM = getExifValueAsDouble("Sensor Width", meta.sensorWidthMM);
    meta.sensorHeightMM = getExifValueAsDouble("Sensor Height", meta.sensorHeightMM);
    meta.imageWidth = getExifValueAsInt("Image Width", meta.imageWidth);
    meta.imageHeight = getExifValueAsInt("Image Height", meta.imageHeight);
    meta.altitude = getExifValueAsDouble("GPS Altitude", meta.altitude);
    meta.yawDeg = getExifValueAsDouble("GPS Img Direction", meta.yawDeg);
    meta.pitchDeg = getExifValueAsDouble("Pitch Angle", meta.pitchDeg);  // Optional custom tag
    meta.rollDeg = getExifValueAsDouble("Roll Angle", meta.rollDeg);    // Optional custom tag
    return meta;
}
