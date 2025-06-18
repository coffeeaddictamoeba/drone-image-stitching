#include "../include/metadata.h"

MetadataExtractor::MetadataExtractor(ExifToolPipe& tool, const std::string& path) 
    : exifTool(tool), imagePath(path) {}

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
    exifTool.sendCommand(imagePath);
    exifData = exifTool.getLastExifData();

    CameraMetadata meta;
    meta.focalLengthMM = getExifValueAsDouble("Focal Length", meta.focalLengthMM);
    meta.sensorWidthMM = getExifValueAsDouble("Sensor Width", meta.sensorWidthMM);
    meta.sensorHeightMM = getExifValueAsDouble("Sensor Height", meta.sensorHeightMM);
    meta.imageWidth = getExifValueAsInt("Image Width", meta.imageWidth);
    meta.imageHeight = getExifValueAsInt("Image Height", meta.imageHeight);
    meta.altitude = getExifValueAsDouble("GPS Altitude", meta.altitude);
    meta.yawDeg = getExifValueAsDouble("GPS Img Direction", meta.yawDeg);
    meta.pitchDeg = getExifValueAsDouble("Pitch Angle", meta.pitchDeg);
    meta.rollDeg = getExifValueAsDouble("Roll Angle", meta.rollDeg);
    return meta;
}
