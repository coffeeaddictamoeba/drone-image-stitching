#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <cstddef>

struct Config {
    std::string incomingDir = "incoming";
    std::size_t blockSize = 256;
    std::size_t batchSize = 10; // optimal, batches of less size result in bad quality images
    std::size_t retries = 3;
    int batchTimeoutSec = 5;
    int retryTimeoutSec = 3;
    bool useBigTIFF = true;
    bool compress = true;
    bool retry = true;
    bool waitForBatchSize = false;
    bool savePreviousOrthophoto = false;
    double rgbValidationThreshold = 25.0;
    double alphaValidationThreshold = 0.1;
};

#endif