#include "../include/metadata.h"
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sstream>

ExifToolPipe::ExifToolPipe() {
    int pipeToChild[2], pipeFromChild[2];
    pipe(pipeToChild);
    pipe(pipeFromChild);

    childPid = fork();
    if (childPid == 0) {
        dup2(pipeToChild[0], STDIN_FILENO);
        dup2(pipeFromChild[1], STDOUT_FILENO);
        close(pipeToChild[1]);
        close(pipeFromChild[0]);
        execlp("exiftool", "exiftool", "-stay_open", "True", "-@", "-", nullptr);
        exit(1);
    }

    writeFd = pipeToChild[1];
    readFd = pipeFromChild[0];
    close(pipeToChild[0]);
    close(pipeFromChild[1]);
}

ExifToolPipe::~ExifToolPipe() {
    std::string quit = "-stay_open\nFalse\n";
    write(writeFd, quit.c_str(), quit.size());
    close(writeFd);
    close(readFd);
    waitpid(childPid, nullptr, 0);
}

bool ExifToolPipe::sendCommand(const std::string& imagePath) {
    std::ostringstream cmd;
    cmd << imagePath << "\n-n\n-execute\n";
    std::string commandStr = cmd.str();
    return write(writeFd, commandStr.c_str(), commandStr.size()) >= 0;
}

std::string ExifToolPipe::readResponse() {
    std::string result;
    char buffer[256];
    ssize_t bytes;
    while ((bytes = read(readFd, buffer, sizeof(buffer) - 1)) > 0) {
        buffer[bytes] = 0;
        result += buffer;
        if (result.find("{ready}") != std::string::npos)
            break;
    }
    return result;
}

bool ExifToolPipe::setExifTag(const std::string& imagePath, const std::string& args) {
    if (imagePath.empty()) return false;

    std::ostringstream cmd;
    cmd << args << " \"" << imagePath << "\""
        << "\n-n\n-execute\n";

    std::string commandStr = cmd.str();
    ssize_t bytesWritten = write(writeFd, commandStr.c_str(), commandStr.size());

    if (bytesWritten < 0) {
        std::cerr << "[ERROR] Failed to write to ExifTool pipe.\n";
        return false;
    }

    return true;
}

bool ExifToolPipe::hasExifTag(const std::string& imagePath, const std::string& tag) {
    if (imagePath.empty()) {
        std::cerr << "[ERROR] hasExifTag: empty image path.\n";
        return false;
    }

    sendCommand(imagePath);
    auto imageData = getLastExifData();

    auto it = imageData.find(tag);
    if (it == imageData.end()) {
        std::cerr << "[WARN] Tag '" << tag << "' not found in metadata of " << imagePath << ".\n";
        return false;
    }

    return !it->second.empty();
}

std::string ExifToolPipe::inExifTag(const std::string& imagePath, const std::string& tag) {
    sendCommand(imagePath);
    auto imageData = getLastExifData();

    auto it = imageData.find(tag);
    if (it != imageData.end()) {
        std::cout << tag << " of " << imagePath << ": " << it->second << '\n';
        return it->second;
    } else {
        std::cerr << "[Warning] Tag \"" << tag << "\" not found in EXIF data of " << imagePath << '\n';
        return "";
    }
}

std::map<std::string, std::string> ExifToolPipe::getLastExifData() {
    std::string response = readResponse();
    std::map<std::string, std::string> data;
    std::istringstream stream(response);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.find("{ready}") != std::string::npos) break;
        auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);
            key.erase(key.find_last_not_of(" \t\r\n") + 1);
            val.erase(0, val.find_first_not_of(" \t\r\n"));
            data[key] = val;
        }
    }
    return data;
}