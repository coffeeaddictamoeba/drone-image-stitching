#ifndef HELPERS_H
#define HELPERS_H

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

#endif