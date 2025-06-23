#pragma once

#ifndef METADATA_H
#define METADATA_H

#include <string>
#include <map>
#include <unistd.h>

class ExifToolPipe {
public:
    ExifToolPipe();
    ~ExifToolPipe();

    ExifToolPipe(const ExifToolPipe&) = delete;
    ExifToolPipe& operator=(const ExifToolPipe&) = delete;
    
    ExifToolPipe(ExifToolPipe&& other) noexcept;
    ExifToolPipe& operator=(ExifToolPipe&& other) noexcept;

    bool sendCommand(const std::string& imagePath);
    bool setExifTag(const std::string& imagePath, const std::string& args);
    bool hasExifTag(const std::string& imagePath, const std::string& tag);
    std::string inExifTag(const std::string& imagePath, const std::string& tag);
    std::map<std::string, std::string> getLastExifData();
    double parseExifNumber(const std::string& value) const;

private:
    pid_t childPid;
    int writeFd;
    int readFd;

    std::string readResponse();
    void closeFd(int& fd);
    void terminateChild();
};

#endif