#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <cstddef>

struct Config {
    std::string dataDir = "images";
    std::string incomingDir = "incoming";
    std::string batchDir = "processing";
    std::string stitchedDir = "stitched";
    std::string stitchedFileName = "final_orthophoto.tif";
    std::string stitchedFile = stitchedDir + '/' + stitchedFileName;
    std::size_t blockSize = 256;
    std::size_t batchSize = 10; // optimal, batches of less size result in bad quality images
    std::size_t retries = 3;
    int batchTimeoutSec = 5;
    int retryTimeoutSec = 3;
    bool useBigTIFF = true;
    bool compress = true;
    bool retry = true;
    bool checkMySetup = false;
    bool isMySetupOkay = false;
    bool waitForBatchSize = false;
    bool savePreviousOrthophoto = false;
    double rgbValidationThreshold = 25.0;
    double alphaValidationThreshold = 0.1;
};

#endif