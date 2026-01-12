#ifndef HELPERS_H
#define HELPERS_H

#include <ctime>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstddef>
#include <variant>
#include <filesystem>

namespace fs = std::filesystem;

inline fs::path constructPathWithPrefix(const fs::path& originalPath, const std::string& prefix) {
    fs::path parent = originalPath.parent_path();
    std::string stem = originalPath.stem().string();      // "image"
    std::string ext  = originalPath.extension().string(); // ".jpg"

    return parent / (stem + prefix + ext);
}

inline fs::path constructPathWithNewDir(const fs::path& originalPath, const fs::path& newDirPath) {
    if (!fs::exists(newDirPath)) fs::create_directories(newDirPath);
    return newDirPath / originalPath.filename();
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

class Timer {
public:
    Timer(const std::string& name): name_(name), start_(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start_;
        std::cout << name_ << " took " << elapsed.count() << " ms\n";
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

#define MEASURE_FUNCTION() Timer timer(__FUNCTION__);

#endif