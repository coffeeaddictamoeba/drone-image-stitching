#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <array>
#include <cmath>

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

namespace metadata {
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

    float parseExposureTime(const std::string &extm) {
        size_t sp = extm.find('/');
        if (sp != std::string::npos) {
            float numerator = std::stof(extm.substr(0, sp));
            float denominator = std::stof(extm.substr(sp + 1));
            return (denominator != 0) ? numerator / denominator : 1.0f;
        } else {
            return std::stof(extm);
        }
    }

    float parseGPSSpeed(const std::string &gpsspeed, const std::string &gpsspeedref) {
        float speed = std::stof(gpsspeed);
        if (std::strcmp(gpsspeedref.c_str(), "km/h") == 0 || std::strcmp(gpsspeedref.c_str(), "K") == 0) return speed * 1000.0f / 3600.0f;
        else if (std::strcmp(gpsspeedref.c_str(), "mph") == 0 || std::strcmp(gpsspeedref.c_str(), "M") == 0) return speed * 1609.34f / 3600.0f;
        else {
            fprintf(
                stdout, 
                YELLOW "[WARN] GPS speed assumed in m/s: %.02f \n" RESET, speed
            );
            return speed;
        }
    }

    bool tagAsFloat(const std::unordered_map<std::string,std::string>& md, const std::string& key, float& value) {
        auto it = md.find(key);
        if (it == md.end()) return false;
        try {
            if (std::strcmp(key.c_str(), "Exposure Time") == 0) { value = parseExposureTime(it->second); return true; }
            else if (std::strcmp(key.c_str(), "GPS Speed") == 0) {
                auto tmpit = md.find("GPS Speed Ref");
                value = parseGPSSpeed(it->second, tmpit->second);
                return true;
            } else {
                value = std::stof(it->second);
                std::cout << "Key: " << it->first << " Value: " << it->second << "\n";
                return true;
            }
        } catch (...) {
            return false;
        }
    }

    bool tagAsInt(const std::unordered_map<std::string,std::string>& md, const std::string& key, int& value) {
        auto it = md.find(key);
        if (it == md.end()) return false;
        try {
            if (std::strcmp(key.c_str(), "Exposure Time") == 0) { value = static_cast<int>(parseExposureTime(it->second)); return true; }
            else if (std::strcmp(key.c_str(), "GPS Speed") == 0) {
                auto tmpit = md.find("GPS Speed Ref");
                value =  static_cast<int>(parseGPSSpeed(it->second, tmpit->second));
                return true;
            } else {
                value = std::stof(it->second);
                return true;
            }
        } catch (...) {
            return false;
        }
    }

    bool isValidTag(std::string_view tag) {
        if (std::find(EXIFTOOL_TAGS.begin(), EXIFTOOL_TAGS.end(), tag) != EXIFTOOL_TAGS.end()) { return true; }

        auto mappedTag = XMP_TAGS_MAP.find(std::string(tag));
        if (mappedTag != XMP_TAGS_MAP.end()) { return true; }

        return false;
    }

    std::unordered_map<std::string, std::string> strtomap(const std::string &out) {
        std::unordered_map<std::string, std::string> md;
        md.reserve(64);
        std::istringstream iss(out);
        std::string line;
        while (std::getline(iss, line)) {
            line = trim(line);
            if (line.empty()) continue;
            auto col = line.find(':');
            if (col != std::string::npos) {
                std::string key = trim(line.substr(0, col));
                std::string value = trim(line.substr(col + 1));
                md[key] = value;
            }
        }
        return md;
    }

    std::unordered_map<std::string, std::string> extractAll(const std::string& imgpath) {
        MEASURE_FUNCTION();
        if (!EXIFTOOL_IS_AVAILABLE) {
            fputs(
                RED "ExifTool not found in PATH." RESET,
                stderr
            );
            return {};
        }

        std::string args = "\"" + imgpath + "\"";
        std::string out = exif.run(args);
        return strtomap(out);
    }

    std::string extract(const std::string& imgpath, const std::string& tagname) {
        MEASURE_FUNCTION();
        if (!EXIFTOOL_IS_AVAILABLE) {
            fputs(
                RED "ExifTool not found in PATH." RESET,
                stderr
            );
        }

        std::string args = "-" + trim(tagname) + " -s3 -q \"" + imgpath + "\"";
        std::string out = exif.run(args);

        std::istringstream iss(out);
        std::string line;
        while (std::getline(iss, line)) {
            line = trim(line);
            if (!line.empty()) return line;
        }
        return std::string{};
    }

    void copyAll(const std::string& srcpath, const std::string& dstpath) {
        MEASURE_FUNCTION();
        if (!EXIFTOOL_IS_AVAILABLE) {
            fputs(
                RED "ExifTool not found. \n" RESET,
                stderr
            );
            return;
        }

        std::string cmd;
        cmd.reserve(
            (sizeof("-overwrite_original -tagsFromFile \"\" \"\"") - 1) + srcpath.size() + dstpath.size() + EXIFTOOL_TAGS_ARGS.size()
        );

        cmd.append("-overwrite_original -tagsFromFile \"");
        cmd.append(srcpath);
        cmd.push_back('"');

        cmd.append(EXIFTOOL_TAGS_ARGS);

        cmd.append(" \"");
        cmd.append(dstpath);
        cmd.push_back('"');

        std::string out = exif.run(cmd);
    }

    void copyAll(const std::unordered_map<std::string, std::string>& md, const std::string& imgpath) {
        MEASURE_FUNCTION();
        if (!EXIFTOOL_IS_AVAILABLE) {
            fputs(
                RED "ExifTool not found in PATH." RESET,
                stderr
            );
            return;
        }

        std::ostringstream ss;
        ss << " -overwrite_original ";

        for (const auto &kv : md) {
            std::string strippedKey;
            strippedKey.reserve(kv.first.size());
            for (char ch : kv.first) if (ch != ' ') strippedKey.push_back(ch);

            if (isValidTag(std::string_view(strippedKey))) {
                ss << " -" << strippedKey << "=\"" << escapeQuotes(kv.second) << "\"";
            }
        }
        ss << " \"" << imgpath << "\"";

        std::string out = exif.run(ss.str());
    }

    // finds Pitch, Roll and Yaw from metadata
    void getPitchRollYaw(const std::unordered_map<std::string, std::string>& md, float &pitch, float &roll, float &yaw) {
        yaw = pitch = roll = 0.0f;
        tagAsFloat(md, "Flight Yaw Degree", yaw);
        tagAsFloat(md, "Flight Pitch Degree", pitch);
        tagAsFloat(md, "Flight Roll Degree", roll);
    }

    // finds Pitch, Roll and Yaw from metadata in radians
    void getPitchRollYawRad(const std::unordered_map<std::string, std::string>& md, float &pitchRad, float &rollRad, float &yawRad) {
        float yaw = 0.0f, pitch = 0.0f, roll = 0.0f;
        tagAsFloat(md, "Flight Yaw Degree", yaw);
        tagAsFloat(md, "Flight Pitch Degree", pitch);
        tagAsFloat(md, "Flight Roll Degree", roll);

        yawRad = yaw * static_cast<float>(M_PI / 180.0f);
        pitchRad = pitch * static_cast<float>(M_PI / 180.0f);
        rollRad = roll * static_cast<float>(M_PI / 180.0f);
    }

    // Finds Drone Speed (both 3D Speed, m/s and Overall Speed, m/s are valid)
    void getSpeedXYZ(const std::unordered_map<std::string, std::string>& md, float &speedX, float &speedY, float &speedZ) { // m/s
        bool hasX = tagAsFloat(md, "Flight X Speed", speedX); // m/s
        bool hasY = tagAsFloat(md, "Flight Y Speed", speedY); // m/s
        bool hasZ = tagAsFloat(md, "Flight Z Speed", speedZ); // m/s

        if (!hasX || !hasY || !hasZ) {
            speedX = speedY = speedZ = 0.0f;
            tagAsFloat(md, "GPS Speed", speedX);

            fprintf(
                stdout,
                YELLOW "[WARN] Using GPS speed fallback: %.02f m/s \r\n" RESET, speedX
            );
        }
    }

    // calculate ground sample distance (GSD)
    float findGSD(float altitude, float focalLength, int imageWidth, int imageHeight, float sensorWidth = 3.68f, float sensorHeight = 2.76f) {
        altitude *= 1000.0f; // m -> mm

        float gsdWidth = (altitude * sensorWidth) / (focalLength * imageWidth);    // mm/px
        float gsdHeight = (altitude * sensorHeight) / (focalLength * imageHeight); // mm/px

        float gsd = std::max(gsdWidth, gsdHeight); // mm/px

        fprintf(
            stdout, 
            "[INFO] Calculated GSD = %.02f mm/px \r\n", gsd
        );

        return gsd;
    }

    // Finds GSD from given metadata (mm/px)
    float findGSD(const std::unordered_map<std::string, std::string>& metadata, float sensorWidth = 3.68f, float sensorHeight = 2.76f) {
        float alt = 0, flen = 0;
        int width = 0, height = 0;

        if (!tagAsFloat(metadata, "GPS Altitude", alt) || !tagAsFloat(metadata, "Focal Length", flen) ||
            !tagAsInt(metadata, "Image Width", width) || !tagAsInt(metadata, "Image Height", height)) {
                fputs(
                    RED "[ERROR] Missing essential metadata for GSD computation." RESET,
                    stderr
                );
                return 1.0f; // fallback (1 mm/px)
        }

        return findGSD(alt, flen, width, height, sensorWidth, sensorHeight);
    }

    // Finds GPS Image Direction (in case of Overall Speed)
    float getGPSImgDirection(const std::unordered_map<std::string, std::string> &metadata) {
        float gpsImgDirection = 0.0f;
        if (!tagAsFloat(metadata, "GPS Img Direction", gpsImgDirection)) {
            fputs(
                RED "[Error] Missing essential metadata for GSD computation." RESET,
                stderr
            );
        }
        return gpsImgDirection;
    }

    // Finds GPS Image Direction (in case of Overall Speed) in radians
    float getGPSImgDirectionRad(const std::unordered_map<std::string, std::string> &metadata) {
        return getGPSImgDirection(metadata) * static_cast<float>(M_PI) / 180.0f;
    }

    void listMetadata() {
        std::cerr << RED << "[ERROR] Image requires essential metadata listed below:\n"
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
                    << "    - Flight Z Speed" << RESET << "\n";
    }
}

