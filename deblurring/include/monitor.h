#include <iostream>
#include <filesystem>
#include <thread>
#include <chrono>
#include <unordered_set>
#include <functional>
#include <atomic>

#define RESET   "\033[0m"
#define RED     "\033[31m"      // Errors
#define YELLOW  "\033[33m"      // Warnings
#define GREEN   "\033[32m"      // Success

namespace fs = std::filesystem;

std::atomic<bool> stopFlag{false}; // ^C handler
void signalHandler(int) {
    std::cout << YELLOW << "\n[Info] Stopping monitor..." << RESET << std::endl;
    stopFlag = true;
}

class DirectoryMonitor {
    public:
        using Callback = std::function<void(const std::string&)>;

        DirectoryMonitor(const std::string& path, Callback cb, int intervalMs = 1000): dirPath_(path), callback_(std::move(cb)), interval_(intervalMs), running_(false) {}

        ~DirectoryMonitor() {
            stop();
        }

        void start() {
            running_ = true;
            monitorThread_ = std::thread([this]() { this->run(); });
        }

        void stop() {
            running_ = false;
            if (monitorThread_.joinable())
                monitorThread_.join();
        }

    private:
        std::string dirPath_;
        Callback callback_;
        int interval_;
        std::unordered_set<std::string> seenFiles_;
        std::atomic<bool> running_;
        std::thread monitorThread_;

        inline bool isJpg(const fs::path& p) {
            auto ext = p.extension().string();
            for (auto& c : ext) c = std::tolower(c);
            return ext == ".jpg" || ext == ".jpeg";
        }

        void run() {
            std::cout << "[Monitor] Start monitoring " << dirPath_ << std::endl;
            while (running_ && !stopFlag) {
                for (auto& entry : fs::directory_iterator(dirPath_)) {
                    if (!running_ || stopFlag) break;
        
                    if (entry.is_regular_file()) {
                        fs::path filePath = entry.path();
        
                        if (isJpg(filePath)) {
                            std::string absPath = fs::absolute(filePath).string();
        
                            if (seenFiles_.insert(absPath).second) {
                                std::cout << YELLOW << "[Monitor] New file detected: " << absPath << RESET << std::endl;
                                callback_(absPath);
                            }
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_));
            }
        }
        
};
