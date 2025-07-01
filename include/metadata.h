#pragma once

#include <vector>
#ifndef METADATA_H
#define METADATA_H

#include <string>
#include <map>
#include <unistd.h>

namespace EXIFTAGS { // compile-time consts
    constexpr const char* IMAGE_HEIGHT_TAG = "ImageHeight";
    constexpr const char* IMAGE_WIDTH_TAG = "ImageWidth";
    constexpr const char* FOCAL_LENGTH_TAG = "FocalLength";
    constexpr const char* GPS_LATITUDE_TAG = "GPSLatitude";
    constexpr const char* GPS_LONGITUDE_TAG = "GPSLongitude";
    constexpr const char* GPS_ALTITUDE_TAG = "GPSAltitude";
    constexpr const char* GPS_IMG_DIRECTION_TAG = "GPSImgDirection";
}

class ExifToolPipe {
public:
    ExifToolPipe();
    ~ExifToolPipe();

    ExifToolPipe(const ExifToolPipe&) = delete;
    ExifToolPipe& operator=(const ExifToolPipe&) = delete;
    
    ExifToolPipe(ExifToolPipe&& other) noexcept;
    ExifToolPipe& operator=(ExifToolPipe&& other) noexcept;

    bool setExifTag(const std::string& imagePath, const std::string& args);
    bool setExifTags(const std::string& imagePath, const std::map<std::string, std::string>& tags);
    bool setExifTagsBatch(const std::vector<std::string>& imagePaths, const std::map<std::string, std::string>& tags);
    bool hasExifTag(const std::string& imagePath, const std::string& tag);
    std::string getExifTag(const std::string& imagePath, const std::string& tag);
    std::map<std::string, std::string> getExifTags(const std::string& imagePath, const std::vector<std::string>& tags);

    double parseExifNumber(const std::string& value) const;
    std::map<std::string, double> parseExifValuesToNumbers(const std::map<std::string, std::string>& tagMap) const;
    double parseExifGPS(const std::string& value) const;

private:
    pid_t childPid;
    int writeFd;
    int readFd;

    bool sendCommand(const std::string& imagePath);
    std::string readResponse();
    std::map<std::string, std::string> getLastExifData();

    void closeFd(int& fd);
    void terminateChild();
};

#endif