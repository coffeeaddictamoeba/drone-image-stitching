#include <iostream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <array>
#include "../include/helpers.h"
#include "../include/metadata.h"

#ifdef _WIN32
#include <process.h>
#define popen _popen
#define pclose _pclose
#endif

#define RESET   "\033[0m"
#define RED     "\033[31m"      // Errors
#define YELLOW  "\033[33m"      // Warnings
#define GREEN   "\033[32m"      // Success

bool isExifToolAvailable() {
    #ifdef _WIN32
        const char* cmd = "where exiftool >nul 2>&1";
    #else
        const char* cmd = "command -v exiftool >/dev/null 2>&1";
    #endif
        return (system(cmd) == 0);
}
    
const bool EXIFTOOL_IS_AVAILABLE = isExifToolAvailable();
ExifToolSession exif;

bool isAllowedTag(std::string_view tag) {
    return std::find(EXIFTOOL_TAGS.begin(), EXIFTOOL_TAGS.end(), tag) != EXIFTOOL_TAGS.end();
}

std::unordered_map<std::string, std::string> strtomap(const std::string &out) {
    std::unordered_map<std::string, std::string> metadata;
    std::istringstream iss(out);
    std::string line;
    while (std::getline(iss, line)) {
        line = trim(line);
        if (line.empty()) continue;
        auto colonPos = line.find(':');
        if (colonPos == std::string::npos) continue;
        std::string key = trim(line.substr(0, colonPos));
        std::string value = trim(line.substr(colonPos + 1));
        metadata[key] = value;
    }
    return metadata;
}

std::unordered_map<std::string, std::string> extractImageMetadata(const std::string& imagePath) {
    if (!EXIFTOOL_IS_AVAILABLE) {
        std::cerr << RED << "ExifTool not found in PATH." << RESET << "\n";
        return {};
    }

    std::string args = "\"" + imagePath + "\"";
    std::string out = exif.run(args);
    return strtomap(out);
}

std::string extractExifTagValue(const std::string& imagePath, const std::string& tagName) {
    if (!EXIFTOOL_IS_AVAILABLE) {
        throw std::runtime_error("ExifTool not found in PATH or not executable.");
    }

    std::string args = "-" + trim(tagName) + " -s3 -q \"" + imagePath + "\"";
    std::string out = exif.run(args);

    std::istringstream iss(out);
    std::string line;
    while (std::getline(iss, line)) {
        line = trim(line);
        if (!line.empty()) return line;
    }
    return std::string{};
}

// assigns metadata to an image from other image's metadata
void copyMetadata(const std::string& sourceImagePath, const std::string& destImagePath) {
        if (!EXIFTOOL_IS_AVAILABLE) {
            std::cerr << RED << "ExifTool not found." << RESET << "\n";
            return;
        }
    
        std::ostringstream ss;
        ss << "-overwrite_original -tagsFromFile \"" << sourceImagePath << "\"";
        for (const auto &tag : EXIFTOOL_TAGS) {
            ss << " -" << tag;
        }
        ss << " \"" << destImagePath << "\"";
    
        std::string out = exif.run(ss.str());
    
    #ifdef DEBUG
        std::cout << "[DEBUG] copyMetadata output:\n" << out << std::endl;
    #endif
}

// assigns metadata to an image from list of tags
void assignMetadata(const std::string& imagePath, const std::unordered_map<std::string, std::string>& tags) {
    if (!EXIFTOOL_IS_AVAILABLE) {
        std::cerr << RED << "ExifTool not found." << RESET << "\n";
        return;
    }

    std::ostringstream ss;
    ss << "-overwrite_original";

    for (const auto &kv : tags) {
        std::string strippedKey;
        strippedKey.reserve(kv.first.size());
        for (char ch : kv.first) if (ch != ' ') strippedKey.push_back(ch);

        if (isAllowedTag(std::string_view(strippedKey))) {
            ss << " -" << strippedKey << "=\"" << escapeQuotes(kv.second) << "\"";
        }
    }
    ss << " \"" << imagePath << "\"";

    std::string out = exif.run(ss.str());

    #ifdef DEBUG
        std::cout << "[DEBUG] assignMetadata output:\n" << out << std::endl;
    #endif
}

float parseExifExposureTime(const std::string &exposure_str) {
    float exposure = 0.0f;
    size_t slash_pos = exposure_str.find('/');
    if (slash_pos != std::string::npos) {
        float numerator = std::stof(exposure_str.substr(0, slash_pos));
        float denominator = std::stof(exposure_str.substr(slash_pos + 1));
        exposure = (denominator != 0) ? numerator / denominator : 1.0f;
    } else {
        exposure = std::stof(exposure_str);
    }
    return exposure;
}

float parseExifGPSSpeed(const std::string &gpsspeed_str, const std::string &gpsspeedref_str) {
    float speed = std::stof(gpsspeed_str);
    if (gpsspeedref_str == "km/h") {
        return speed * 1000.0f / 3600.0f;
    } else if (gpsspeedref_str == "mph") {
        return speed * 1609.34f / 3600.0f;
    } else {
        std::cout << YELLOW << "[Warn] GPS speed assumed in m/s: " << speed << RESET <<"\n";
        return speed;
    }
}

void listMetadata() {
    std::cerr << RED << "[Error] Image requires essential metadata listed below:\n"
                << "    - Flight Yaw Degree\n"
                << "    - Flight Pitch Degree\n"
                << "    - Flight Roll Degree\n"
                << "    - GPS Altitude\n"
                << "    - Exposure Time\n"
                << " If you are using overall speed parameters, check:\n"
                << "    - GPS Speed\n"
                << "    - GPS Speed Ref\n"
                << "    - GPS Img Direction\n"
                << " If you are using 3D speed parameters, check:\n"
                << "    - Flight X Speed\n" 
                << "    - Flight Y Speed\n" 
                << "    - Flight Z Speed" << RESET << std::endl;
}