#include "../include/metadata.h"
#include "../include/strutils.hpp"
#include "../external/ctre.hpp"
#include <iostream>
#include <regex>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sstream>
#include <string.h>
#include <poll.h>

void ExifToolPipe::closeFd(int& fd) {
    if (fd != -1) {
        close(fd);
        fd = -1;
    }
}

void ExifToolPipe::terminateChild() {
    if (childPid > 0) {
        kill(childPid, SIGTERM);
        waitpid(childPid, nullptr, 0);
        childPid = -1;
    }
}

ExifToolPipe::ExifToolPipe() : childPid(-1), writeFd(-1), readFd(-1) {
    int pipeToChild[2], pipeFromChild[2];
    
    if (pipe(pipeToChild) == -1 || pipe(pipeFromChild) == -1) {
        throw std::runtime_error("ExifToolPipe: Failed to create pipes");
    }

    childPid = fork();
    if (childPid == -1) {
        throw std::runtime_error("ExifToolPipe: Failed to fork");
    }

    if (childPid == 0) {
        dup2(pipeToChild[0], STDIN_FILENO);
        dup2(pipeFromChild[1], STDOUT_FILENO);

        close(pipeToChild[0]); close(pipeToChild[1]);
        close(pipeFromChild[0]); close(pipeFromChild[1]);

        execlp("exiftool", "exiftool", "-stay_open", "True", "-@", "-", nullptr);
        perror("execlp failed");
        _exit(1);
    }

    writeFd = pipeToChild[1];
    readFd = pipeFromChild[0];

    close(pipeToChild[0]);
    close(pipeFromChild[1]);
}

ExifToolPipe::~ExifToolPipe() {
    if (writeFd != -1) {
        std::string quit_cmd = "-stay_open\nFalse\n";
        write(writeFd, quit_cmd.c_str(), quit_cmd.size());
    }
    closeFd(writeFd);
    closeFd(readFd);
    terminateChild();
}

std::string ExifToolPipe::readResponse() {
    std::string result;
    char buffer[256];
    struct pollfd pfd = { readFd, POLLIN, 0 };

    while (true) {
        int ret = poll(&pfd, 1, 3000);
        if (ret <= 0) break;

        ssize_t bytes = read(readFd, buffer, sizeof(buffer) - 1);
        if (bytes > 0) {
            result.append(buffer, bytes);
            if (result.find("{ready}") != std::string::npos) break;
        } else {
            break;
        }
    }
    //std::cout << "ExifTool read the data successfully\n";

    return result;
}

bool ExifToolPipe::sendCommand(const std::string& command) {
    if (writeFd == -1) return false;
    std::string fullCmd = command + "\n-execute\n";
    return write(writeFd, fullCmd.c_str(), fullCmd.size()) == (ssize_t)fullCmd.size();
}

std::map<std::string, std::string> ExifToolPipe::getLastExifData() {
    std::string response = readResponse();
    std::map<std::string, std::string> data;
    std::istringstream stream(response);
    std::string line;

    while (std::getline(stream, line)) {
        auto pos = line.find(':');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);

            // trim spaces
            key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
            value = std::regex_replace(value, std::regex("^ +| +$"), "");

            data[key] = value;
            //std::cout << "[DEBUG] ExifTool got: " << key << " : " << value << '\n';
        }
    }
    return data;
}

double ExifToolPipe::parseExifNumber(const std::string& value) const {
    if (auto m = ctre::search<R"((\+|-)?\d*(\.\d+)?)">(value)) {
        return utils::to_double(m.to_view());
    }
    throw std::runtime_error("Failed to parse EXIF numeric value: " + value);
}

double ExifToolPipe::parseExifGPS(const std::string& dmsStr) const {
    double deg = 0.0;
    double min = 0.0;
    double sec = 0.0;
    int sign = 1;

    std::string s = utils::trim(dmsStr);

    if (!s.empty()) {
        char lastChar = s.back();
        if (lastChar == 'S' || lastChar == 'W') {
            sign = -1;
            s.pop_back();
            s = utils::trim(s);
        } else if (lastChar == 'N' || lastChar == 'E') {
            s.pop_back();
            s = utils::trim(s);
        }
    }

    size_t degPos = s.find("deg");
    size_t minPos = s.find("'");
    size_t secPos = s.find("\"");

    if (degPos == std::string::npos && minPos == std::string::npos) {
        try {
            return sign * std::stod(s);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing as direct decimal: '" << s << "' - " << e.what() << "\n";
            return 0.0;
        }
    }

    try {
        deg = std::stod(utils::trim(s.substr(0, degPos)));

        std::string minStr = utils::trim(s.substr(degPos + 3, minPos - (degPos + 3)));
        min = std::stod(minStr);

        if (secPos != std::string::npos) {
            std::string secStr = utils::trim(s.substr(minPos + 1, secPos - (minPos + 1)));
            sec = std::stod(secStr);
        } else {
            std::string remainingStr = utils::trim(s.substr(minPos + 1));
            if (!remainingStr.empty() && std::isdigit(remainingStr[0])) {
                sec = std::stod(remainingStr);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing DMS string: '" << dmsStr << "' - " << e.what() << "\n";
        return 0.0;
    }

    double decimalDeg = deg + (min / 60.0) + (sec / 3600.0);
    return sign * decimalDeg;
}

bool ExifToolPipe::setExifTag(const std::string& imagePath, const std::string& args) {
    std::ostringstream cmd;
    cmd << args << "\n" << imagePath << "\n" << "-overwrite_original_in_place";
    if (!sendCommand(cmd.str())) return false;

    std::string response = readResponse();
    return response.find("image files updated") != std::string::npos ||
           response.find("image files created") != std::string::npos;
}

bool ExifToolPipe::setExifTags(const std::string& imagePath, const std::map<std::string, std::string>& tags) {
    std::ostringstream cmd;
    for (const auto& [tag, value] : tags) {
        cmd << "-" << tag << "=" << value << "\n";
    }
    cmd << imagePath << "\n-overwrite_original_in_place";

    if (!sendCommand(cmd.str())) return false;

    std::string response = readResponse();
    return response.find("image files updated") != std::string::npos || response.find("image files created") != std::string::npos;
}

bool ExifToolPipe::hasExifTag(const std::string& imagePath, const std::string& tag) {
    std::ostringstream cmd;
    cmd << "-" << tag << "\n" << imagePath;
    if (!sendCommand(cmd.str())) return false;

    auto data = getLastExifData();
    return data.find(tag) != data.end();
}

std::string ExifToolPipe::getExifTag(const std::string& imagePath, const std::string& tag) {
    std::ostringstream cmd;
    cmd << "-" << tag << "\n" << imagePath;
    //std::cout << "[DEBUG]: Running " << "-" << tag << "\n" << imagePath << '\n'; 
    if (!sendCommand(cmd.str())) return "";

    auto data = getLastExifData();
    auto it = data.find(tag);
    return (it != data.end()) ? it->second : "";
}

std::map<std::string, std::string> ExifToolPipe::getExifTags(const std::string& imagePath, const std::vector<std::string>& tags) {
    std::ostringstream cmd;
    for (const auto& tag : tags) {
        cmd << "-" << tag << "\n";
    }
    cmd << imagePath;

    if (!sendCommand(cmd.str())) return {};

    return getLastExifData();
}

bool ExifToolPipe::setExifTagsBatch(const std::vector<std::string>& imagePaths, const std::map<std::string, std::string>& tags) {
    if (imagePaths.empty() || tags.empty()) return false;

    std::ostringstream cmd;
    for (const auto& [tag, value] : tags) {
        cmd << "-" << tag << "=" << value << "\n";
    }

    for (const auto& path : imagePaths) {
        cmd << path << "\n";
    }

    cmd << "-overwrite_original_in_place";

    if (!sendCommand(cmd.str())) return false;

    std::string response = readResponse();
    return response.find("image files updated") != std::string::npos || response.find("image files created") != std::string::npos;
}

std::map<std::string, double> ExifToolPipe::parseExifValuesToNumbers(const std::map<std::string, std::string>& tagMap) const {
    std::map<std::string, double> result;

    for (const auto& [key, value] : tagMap) {
        try {
            if (key == EXIFTAGS::GPS_LATITUDE_TAG || key == EXIFTAGS::GPS_LONGITUDE_TAG) {
                result[key] = parseExifGPS(value);
            } else {
                result[key] = std::stod(value);
            }
        } catch (const std::invalid_argument& e) {
            std::cerr << "Warning: Invalid argument for stod/parseExifGPS for tag '" << key << "' with value '" << value << "': " << e.what() << "\n";
            result[key] = 0.0;
        } catch (const std::out_of_range& e) {
            std::cerr << "Warning: Out of range for stod/parseExifGPS for tag '" << key << "' with value '" << value << "': " << e.what() << "\n";
            result[key] = 0.0;
        }
    }
    return result;
}