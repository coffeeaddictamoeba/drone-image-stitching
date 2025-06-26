#include "../include/metadata.h"
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
#include <chrono>
#include <thread>

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
    std::smatch match;
    std::regex numberRegex(R"([-+]?\d*\.?\d+)");
    if (std::regex_search(value, match, numberRegex)) {
        return std::stod(match.str());
    }
    throw std::runtime_error("Failed to parse EXIF numeric value: " + value);
}

double ExifToolPipe::parseExifGPS(const std::string& value) const {
    std::string str = std::regex_replace(value, std::regex("^ +| +$|( ) +"), "$1");
    std::smatch match;
    std::regex dmsRegex(R"((\d+)[^\d]+(\d+)[^\d]+([\d.]+))"); // DMS format: 54 deg 54' 19.67"
    if (std::regex_search(str, match, dmsRegex) && match.size() == 4) {
        double degrees = std::stod(match[1]);
        double minutes = std::stod(match[2]);
        double seconds = std::stod(match[3]);
        return degrees + minutes / 60.0 + seconds / 3600.0;
    }
    throw std::runtime_error("Failed to parse EXIF GPS value: " + value);
}

bool ExifToolPipe::setExifTag(const std::string& imagePath, const std::string& args) {
    std::ostringstream cmd;
    cmd << args << "\n" << imagePath << "\n" << "-overwrite_original_in_place";
    if (!sendCommand(cmd.str())) return false;

    std::string response = readResponse();
    return response.find("image files updated") != std::string::npos ||
           response.find("image files created") != std::string::npos;
}

bool ExifToolPipe::hasExifTag(const std::string& imagePath, const std::string& tag) {
    std::ostringstream cmd;
    cmd << "-" << tag << "\n" << imagePath;
    if (!sendCommand(cmd.str())) return false;

    auto data = getLastExifData();
    return data.find(tag) != data.end();
}

std::string ExifToolPipe::inExifTag(const std::string& imagePath, const std::string& tag) {
    std::ostringstream cmd;
    cmd << "-" << tag << "\n" << imagePath;
    //std::cout << "[DEBUG]: Running " << "-" << tag << "\n" << imagePath << '\n'; 
    if (!sendCommand(cmd.str())) return "";

    auto data = getLastExifData();
    auto it = data.find(tag);
    return (it != data.end()) ? it->second : "";
}