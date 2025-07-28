#include <iostream>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

bool isExifToolAvailable() {
#ifdef _WIN32
    const char* cmd = "where exiftool >nul 2>&1";
#else
    const char* cmd = "command -v exiftool >/dev/null 2>&1";
#endif
    return (system(cmd) == 0);
}

int runExifToolCommand(const std::string& cmd) {
#ifdef _WIN32
    std::string fullCmd = cmd + " >nul 2>&1";
#else
    std::string fullCmd = cmd + " >/dev/null 2>&1";
#endif
    return system(fullCmd.c_str());
}

std::unordered_map<std::string, std::string> extractImageMetadata(const std::string& imagePath) {
    std::unordered_map<std::string, std::string> metadata;
    if (!isExifToolAvailable()) {
        std::cerr << "ExifTool not found in PATH.\n";
        return metadata;
    }

    std::string cmd = "exiftool \"" + imagePath + "\"";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "Failed to run exiftool.\n";
        return metadata;
    }

    char buffer[512];
    std::regex kvRegex(R"(^\s*([^:]+?)\s*:\s*(.*)\s*$)");
    std::smatch match;

    while (fgets(buffer, sizeof(buffer), pipe)) {
        std::string line(buffer);
        if (std::regex_match(line, match, kvRegex)) {
            std::string key = match[1].str();
            std::string value = match[2].str();
            metadata[key] = value;

            if (key == "User Comment") {
                std::regex innerRegex(R"(([^=,]+)=([^=,]+))");
                for (auto it = std::sregex_iterator(value.begin(), value.end(), innerRegex); it != std::sregex_iterator(); ++it) {
                    metadata[(*it)[1].str()] = (*it)[2].str();
                }
            }
        }
    }
    pclose(pipe);
    return metadata;
}

void copyMetadata(const std::string& sourceImagePath, const std::string& destImagePath, const std::unordered_map<std::string, std::string>& customTags) {
    if (!isExifToolAvailable()) {
        std::cerr << "ExifTool not found.\n";
        return;
    }

    std::ostringstream cmdStream;
    cmdStream << "exiftool -overwrite_original -tagsFromFile \"" << sourceImagePath << "\"";

    std::vector<std::string> tagsToCopy = {
        "-CameraModelName", "-Make", "-Software", "-ModifyDate", "-ExposureTime", "-ISO",
        "-DateTimeOriginal", "-CreateDate", "-FocalLength",
        "-GPSVersionID", "-GPSLatitude", "-GPSLongitude", "-GPSAltitude", "-GPSImgDirection",
        "-GPSSpeed", "-GPSSpeedRef", "-FlightPitchDegree", "-FlightYawDegree", "-FlightRollDegree"
    };

    for (const auto& tag : tagsToCopy) {
        cmdStream << " " << tag;
    }

    if (!customTags.empty()) {
        cmdStream << " -UserComment=\"";
        bool first = true;
        for (const auto& [key, value] : customTags) {
            if (!first) cmdStream << ",";
            cmdStream << key << "=" << value;
            first = false;
        }
        cmdStream << "\"";
    }

    cmdStream << " \"" << destImagePath << "\"";
    if (runExifToolCommand(cmdStream.str()) != 0) {
        std::cerr << "Failed to copy metadata with ExifTool.\n";
    }
}

void assignMetadata(const std::string& imagePath, const std::unordered_map<std::string, std::string>& tags) {
    if (!isExifToolAvailable()) {
        std::cerr << "ExifTool not found.\n";
        return;
    }

    std::ostringstream cmdStream;
    cmdStream << "exiftool -overwrite_original";

    std::vector<std::string> standardTags = {
        "GPSAltitude", "FocalLength", "GPSLatitude", "GPSLongitude", "GPSImgDirection",
        "GPSSpeed", "GPSSpeedRef", "ExposureTime", "ISO", "CreateDate", 
        "DateTimeOriginal", "Make", "Model", "FlightPitchDegree", "FlightYawDegree", 
        "FlightRollDegree"
    };

    std::vector<std::string> userCommentParts;
    for (const auto& [key, value] : tags) {
        std::string strippedKey;
        for (char ch : key) if (ch != ' ') strippedKey += ch;

        if (std::find(standardTags.begin(), standardTags.end(), strippedKey) != standardTags.end()) {
            cmdStream << " -" << strippedKey << "=\"" << value << "\"";
        } else {
            userCommentParts.emplace_back(key + "=" + value);
        }
    }

    if (!userCommentParts.empty()) {
        cmdStream << " -UserComment=\"";
        for (size_t i = 0; i < userCommentParts.size(); ++i) {
            if (i > 0) cmdStream << ",";
            cmdStream << userCommentParts[i];
        }
        cmdStream << "\"";
    }

    cmdStream << " \"" << imagePath << "\"";

    if (runExifToolCommand(cmdStream.str()) != 0) {
        std::cerr << "Failed to assign metadata.\n";
    }
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
        std::cout << "[Warn] GPS speed assumed in m/s: " << speed << "\n";
        return speed;
    }
}