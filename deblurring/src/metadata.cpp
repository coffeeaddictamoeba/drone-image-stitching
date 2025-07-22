#include "../include/metadata.h"
#include <iostream>
#include <regex>
#include <sstream>
#include <cstdio>
#include <unordered_map>

std::unordered_map<std::string, std::string> extractImageMetadata(const std::string& imagePath) {
    std::unordered_map<std::string, std::string> metadata;
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
                auto words_begin = std::sregex_iterator(value.begin(), value.end(), innerRegex);
                auto words_end = std::sregex_iterator();
                for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
                    std::string innerKey = (*i)[1].str();
                    std::string innerVal = (*i)[2].str();
                    metadata[innerKey] = innerVal;
                }
            }
        }
    }
    pclose(pipe);
    return metadata;
}

void copyMetadata(const std::string& sourceImagePath, const std::string& destImagePath, const std::unordered_map<std::string, std::string>& customTags) {
    std::ostringstream cmdStream;
    cmdStream << "exiftool -overwrite_original -tagsFromFile \"" << sourceImagePath << "\"";

    std::vector<std::string> tagsToCopy = {
        "-CameraModelName", "-Make", "-Software", "-ModifyDate", "-ExposureTime", "-ISO",
        "-DateTimeOriginal", "-CreateDate", "-FocalLength",
        "-GPSVersionID", "-GPSLatitude", "-GPSLongitude", "-GPSAltitude", "-GPSImgDirection"
    };

    for (const auto& tag : tagsToCopy) {
        cmdStream << " " << tag;
    }

    if (!customTags.empty()) {
        cmdStream << " -UserComment=\"";
        bool first = true;
        for (const auto& pair : customTags) {
            if (!first) {
                cmdStream << ",";
            }
            cmdStream << pair.first << "=" << pair.second;
            first = false;
        }
        cmdStream << "\"";
    }

    cmdStream << " \"" << destImagePath << "\"";

    std::string cmd = cmdStream.str();
    int result = system(cmd.c_str());
    if (result != 0) {
        std::cerr << "ExifTool command failed with exit code " << result << ": " << cmd << "\n";
    }
}

void assignMetadata(const std::string& imagePath, const std::unordered_map<std::string, std::string>& tags) {
    std::ostringstream cmdStream;
    cmdStream << "exiftool -overwrite_original";

    std::vector<std::string> standardTags = {
        "GPSAltitude", "FocalLength", "GPSLatitude", "GPSLongitude", "GPSImgDirection",
        "ExposureTime", "ISO", "CreateDate", "DateTimeOriginal", "Make", "Model"
    };

    std::vector<std::string> userCommentParts;

    for (const auto& [key, value] : tags) {
        std::string normalizedKey = key;
        std::string strippedKey;
        for (char ch : key) {
            if (ch != ' ') strippedKey += ch;
        }

        if (std::find(standardTags.begin(), standardTags.end(), strippedKey) != standardTags.end()) {
            cmdStream << " -" << strippedKey << "=\"" << value << "\"";
        } else {
            userCommentParts.push_back(key + "=" + value);
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

    std::string cmd = cmdStream.str();
    int result = system(cmd.c_str());
    if (result != 0) {
        std::cerr << "Failed to assign metadata. Command:\n" << cmd << "\nExit code: " << result << "\n";
    }
}