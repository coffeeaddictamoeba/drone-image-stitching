#ifndef HELPERS_H
#define HELPERS_H

#include <string>
#include <cstddef>


inline std::string constructPathWithPrefix(std::string &originalPath, std::string &prefix) {
    size_t extensionStart = originalPath.find('.');
    std::string imageName = originalPath.substr(0, extensionStart);
    std::string imageExtension = originalPath.substr(extensionStart, originalPath.size());
    return imageName + prefix + imageExtension;
}

inline std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r\f\v");
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(first, (last - first + 1));
}

#endif