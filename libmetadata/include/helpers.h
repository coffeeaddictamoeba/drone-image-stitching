#ifndef HELPERS_H
#define HELPERS_H

#include <ctime>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstddef>
#include <variant>

inline std::string constructPathWithPrefix(const std::string &originalPath, const std::string &prefix) {
    size_t extensionStart = originalPath.find_last_of('.');
    std::string imageName = originalPath.substr(0, extensionStart);
    std::string imageExtension = originalPath.substr(extensionStart, originalPath.size());
    return imageName + prefix + imageExtension;
}

inline std::string constructPathWithNewDir(const std::string &originalPath, const std::string &newDirPath) {
    size_t filenameStart = originalPath.find_last_of('/');

    #ifdef _WIN32
        filenameStart = originalPath.find_last_of('\\');
    #endif

    std::string filename = "";
    if (filenameStart != std::variant_npos) {
        filename = originalPath.substr(filenameStart, originalPath.size());
    } else {
        #ifdef _WIN32
            if (newDirPath.find_last_of('\\') != std::variant_npos) filename = originalPath;
            else filename = "\\" + originalPath;
        #else
            if (newDirPath.find_last_of('/') != std::variant_npos) filename = originalPath;
            else filename = "/" + originalPath;
        #endif
    }
    
    return newDirPath + filename;
}

inline std::string trim(const std::string &s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e-1]))) --e;
    return s.substr(b, e-b);
}

inline std::string escapeQuotes(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else out += c;
    }
    return out;
}

inline std::string getExtension(const std::string &filepath) {
    size_t extensionStart = filepath.find_last_of('.');
    std::string imageName = filepath.substr(0, extensionStart);
    return filepath.substr(extensionStart, filepath.size());
}

inline bool isJPG(const std::string &filepath) {
    auto ext = getExtension(filepath);
    for (auto& c : ext) c = std::tolower(c);
    return ext == ".jpg" || ext == ".jpeg";
}

inline std::string getTimestamp() {
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream tss;
    tss << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");
    return tss.str();
}

#endif