#pragma once

#ifndef METADATA_H
#define METADATA_H

#include <string>
#include <map>
#include <unistd.h>

// Exiftool tags for metadata reference
const std::string IMG_HEIGHT_TAG = "ImageHeight";
const std::string IMG_WIDTH_TAG = "ImageWidth";
const std::string IMG_FOCAL_LEN_TAG = "FocalLength";
const std::string IMG_GPS_LAT = "GPSLatitude";
const std::string IMG_GPS_LON = "GPSLongitude";
const std::string IMG_GPS_ALT = "GPSAltitude";


class ExifToolPipe {
public:
    ExifToolPipe();
    ~ExifToolPipe();

    ExifToolPipe(const ExifToolPipe&) = delete;
    ExifToolPipe& operator=(const ExifToolPipe&) = delete;
    
    ExifToolPipe(ExifToolPipe&& other) noexcept;
    ExifToolPipe& operator=(ExifToolPipe&& other) noexcept;

    bool setExifTag(const std::string& imagePath, const std::string& args);
    bool hasExifTag(const std::string& imagePath, const std::string& tag);
    std::string inExifTag(const std::string& imagePath, const std::string& tag);

    double parseExifNumber(const std::string& value) const;
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