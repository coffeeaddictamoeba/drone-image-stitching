#ifndef STITCHER_H
#define STITCHER_H

#include "config.h"
#include "fwatcher.h"
#include "batchproc.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <csignal>

class MosaicStitcher {
public:
    MosaicStitcher();
    void run(int argc, char* argv[]);

private:
    Config config_;
    std::atomic<bool> stopSignal_;
    std::mutex queueMutex_;
    std::condition_variable queueCV_;
    std::queue<BatchTask> batchQueue_;

    std::unique_ptr<FolderWatcher> watcher_;
    std::unique_ptr<BatchProcessor> processor_;

    std::thread watcherThread_;
    std::thread processorThread_;

    static MosaicStitcher* instance_;
    static void signalHandler(int signum);

    void parseArgs(int argc, char* argv[]);
    void isMySetupOkay();
    void initGDAL();
    void cleanupGDAL();
};

#endif