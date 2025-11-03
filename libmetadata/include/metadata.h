#pragma once

#ifndef METADATA_H
#define METADATA_H

#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <array>
#include <unordered_map>
#include <vector>

// NOTE: this list is a metadata minimum required for deblurring/image stitching
constexpr const std::array<std::string_view, 23> EXIFTOOL_TAGS = {
    "CameraModelName", 
    "Make", 
    "Software", 
    "ModifyDate", 
    "ExposureTime", 
    "ISO",
    "DateTimeOriginal", 
    "CreateDate", 
    "FocalLength",
    "GPSVersionID", 
    "GPSLatitude", 
    "GPSLongitude", 
    "GPSAltitude",
    "GPSImgDirection", 
    "GPSImgDirectionRef", 
    "GPSSpeed", 
    "GPSSpeedRef",
    "FlightPitchDegree", 
    "FlightYawDegree", 
    "FlightRollDegree",
    "XMP-drone-dji:FlightXSpeed", 
    "XMP-drone-dji:FlightYSpeed", 
    "XMP-drone-dji:FlightZSpeed"
};

class ExifToolSession {
    public:
        ExifToolSession() {
            proc_ = nullptr;
            #ifdef _WIN32
                proc_ = _popen("exiftool -stay_open True -@ -", "w+");
            #else
                proc_ = popen("exiftool -stay_open True -@ -", "r+");
                if (!proc_) {
                    proc_ = popen("exiftool -stay_open True -@ -", "w+");
                }
            #endif
    
            if (proc_) {
                stayOpen_ = true;
            } else {
                stayOpen_ = false;
            }
        }
    
        ~ExifToolSession() {
            if (stayOpen_ && proc_) {
                fprintf(proc_, "-stay_open\nFalse\n-execute\n");
                fflush(proc_);
                char buf[1024];
                while (fgets(buf, sizeof(buf), proc_)) { /* ignore */ }
                pclose(proc_);
            }
        }
    
        std::string run(const std::string &args) {
            if (stayOpen_) {
                return runStayOpen(args);
            } else {
                return runOneShot(args);
            }
        }
    
    private:
        FILE* proc_ = nullptr;
        bool stayOpen_ = false;
    
        std::string runStayOpen(const std::string &args) {
            if (!proc_) throw std::runtime_error("ExifTool -stay_open not available.");
    
            fprintf(proc_, "%s\n-execute\n", args.c_str());
            fflush(proc_);
    
            std::string output;
            char buf[1024];
            while (fgets(buf, sizeof(buf), proc_)) {
                if (strstr(buf, "{ready}")) {
                    break;
                }
                output += buf;
            }
            return output;
        }
    
        std::string runOneShot(const std::string &args) {
            std::string one = std::string("exiftool ") + args;

            auto pipeDeleter = [](FILE* fp) noexcept {
                if (fp) pclose(fp);
            };
            
            std::unique_ptr<FILE, decltype(pipeDeleter)> pipe(popen(one.c_str(), "r"), pipeDeleter);            
            if (!pipe) throw std::runtime_error("Failed to run exiftool one-shot command.");
    
            std::string out;
            char buf[1024];
            while (fgets(buf, sizeof(buf), pipe.get())) {
                out += buf;
            }
            return out;
        }
};

// info
void listMetadata();

// metadata operations
std::unordered_map<std::string, std::string> extractImageMetadata(const std::string& imagePath);
std::string extractExifTagValue(const std::string& imagePath, const std::string& tagName);

void copyMetadata(const std::string& sourceImagePath, const std::string& destImagePath);
void assignMetadata(const std::string& imagePath, const std::unordered_map<std::string, std::string>& tags);

void getPitchRollYaw(const std::unordered_map<std::string, std::string>& metadata, float &pitch, float &roll, float &yaw);
void getPitchRollYawRad(const std::unordered_map<std::string, std::string> &metadata, float &pitchRad, float &rollRad, float &yawRad);
void getSpeedXYZ(const std::unordered_map<std::string, std::string> &metadata, float &speedX, float &speedY, float &speedZ);
float getGPSImgDirection(const std::unordered_map<std::string, std::string> &metadata);
float getGPSImgDirectionRad(const std::unordered_map<std::string, std::string> &metadata);

float findGSD(float altitude, float focalLength, int imageWidth, int imageHeight, float sensorWidth, float sensorHeight);
float findGSD(const std::unordered_map<std::string, std::string> &metadata, float sensorWidth, float sensorHeight);

// parsers
inline bool parseFloatFromMetadata(const std::unordered_map<std::string,std::string>& metadata, const std::string& key, float& value) {
    auto it = metadata.find(key);
    if (it == metadata.end()) return false;
    try {
        value = std::stof(it->second);
        return true;
    } catch (...) {
        return false;
    }
}

inline bool parseIntFromMetadata(const std::unordered_map<std::string,std::string>& metadata, const std::string& key, int& value) {
    auto it = metadata.find(key);
    if (it == metadata.end()) return false;
    try {
        value = std::stoi(it->second);
        return true;
    } catch (...) {
        return false;
    }
}

float parseExifExposureTime(const std::string &exposure_str);
float parseExifGPSSpeed(const std::string &gpsspeed_str, const std::string &gpsspeedref_str);

inline constexpr std::size_t TAGS_ARGS_LEN = []() consteval {
    std::size_t sum = 0;
    for (auto tag : EXIFTOOL_TAGS) sum += 2 + tag.size(); // " -" + tag
    return sum;
}();

template <std::size_t M>
consteval auto tagsToArgs() {
    std::array<char, M + 1> out{}; // +1 for '\0'
    std::size_t pos = 0;
    for (auto tag : EXIFTOOL_TAGS) {
        out[pos++] = ' ';
        out[pos++] = '-';
        for (char c : tag) out[pos++] = c;
    }
    out[pos] = '\0';
    return out;
}

inline constexpr auto TAG_ARGS_BUF = tagsToArgs<TAGS_ARGS_LEN>();
inline constexpr std::string_view EXIFTOOL_TAGS_ARGS(
    TAG_ARGS_BUF.data(),
    TAG_ARGS_BUF.size() - 1 // drop '\0'
);

#endif