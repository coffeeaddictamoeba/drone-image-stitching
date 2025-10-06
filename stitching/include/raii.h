#ifndef RAII_H
#define RAII_H

#include <gdal_priv.h>
#include <string>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

class GDALDatasetRAII {
public:
    explicit GDALDatasetRAII(GDALDataset* ds = nullptr) : dataset_(ds) {}

    GDALDatasetRAII(const GDALDatasetRAII&) = delete;
    GDALDatasetRAII& operator=(const GDALDatasetRAII&) = delete;

    GDALDatasetRAII(GDALDatasetRAII&& other) noexcept : dataset_(other.dataset_) {
        other.dataset_ = nullptr;
    }
    GDALDatasetRAII& operator=(GDALDatasetRAII&& other) noexcept {
        if (this != &other) {
            if (dataset_) GDALClose(dataset_);
            dataset_ = other.dataset_;
            other.dataset_ = nullptr;
        }
        return *this;
    }

    ~GDALDatasetRAII() {
        if (dataset_) {
            GDALClose(dataset_);
            dataset_ = nullptr;
        }
    }

    GDALDataset* get() const { return dataset_; }
    GDALDataset* operator->() const { return dataset_; }
    operator bool() const { return dataset_ != nullptr; }
    bool operator!() const { return dataset_ == nullptr; }

private:
    GDALDataset* dataset_;
};

class CPLStringRAII {
    public:
        explicit CPLStringRAII(const char* s = nullptr) : str_(nullptr) {
            if (s) {
                str_ = CPLStrdup(s);
            }
        }
    
        ~CPLStringRAII() {
            if (str_) {
                CPLFree(str_);
                str_ = nullptr;
            }
        }
    
        CPLStringRAII(const CPLStringRAII&) = delete;
        CPLStringRAII& operator=(const CPLStringRAII&) = delete;
    
        CPLStringRAII(CPLStringRAII&& other) noexcept : str_(other.str_) {
            other.str_ = nullptr;
        }
        CPLStringRAII& operator=(CPLStringRAII&& other) noexcept {
            if (this != &other) {
                if (str_) CPLFree(str_);
                str_ = other.str_;
                other.str_ = nullptr;
            }
            return *this;
        }
    
        const char* get() const { return str_; }
        operator const char*() const { return str_; }
        operator std::string() const { return str_ ? std::string(str_) : ""; }
        bool empty() const { return str_ == nullptr || str_[0] == '\0'; }
    
    private:
        char* str_;
    };

class TemporaryPath {
    public:
        TemporaryPath(const fs::path& p, bool is_dir = false) : path_(p), is_directory_(is_dir) {}
        
        TemporaryPath(const TemporaryPath&) = delete;
        TemporaryPath& operator=(const TemporaryPath&) = delete;
        
        TemporaryPath(TemporaryPath&& other) noexcept : path_(std::move(other.path_)), is_directory_(other.is_directory_) {
            other.path_ = "";
        }
        TemporaryPath& operator=(TemporaryPath&& other) noexcept {
            if (this != &other) {
                cleanup();
                path_ = std::move(other.path_);
                is_directory_ = other.is_directory_;
                other.path_ = "";
            }
            return *this;
        }
        
        ~TemporaryPath() {
            cleanup();
        }
        
        const fs::path& get_path() const { return path_; }
        operator const fs::path&() const { return path_; }
        
        fs::path release() {
            fs::path p = std::move(path_);
            path_ = "";
            return p;
        }
        
    private:
        fs::path path_;
        bool is_directory_;
        
        void cleanup() {
            if (!path_.empty() && fs::exists(path_)) {
                try {
                    if (is_directory_) {
                        fs::remove_all(path_);
                    } else {
                        fs::remove(path_);
                    }
                } catch (const fs::filesystem_error& e) {
                    std::cerr << "[WARN] Failed to clean up temporary path " << path_ << ": " << e.what() << std::endl;
                }
            }
        }
    };

#endif