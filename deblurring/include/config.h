#ifndef DEBLUR_CONFIG_H
#define DEBLUR_CONFIG_H

#include <string>
#include <cstddef>
#include <filesystem>

namespace fs = std::filesystem;

struct DeblurConfig {
    fs::path testImagePath = "synthetic.jpg";
    fs::path originalImagePath;
    fs::path sourceDir;
    fs::path targetDir;
    
    float snr = 1500.0;
    float blurThreshold = 100.0;
    float sensorWidth = 3.68f;
    float sensorHeight = 2.76f;

    bool generateTest = false;
    bool overwriteMetadata = false;
    bool blur = false;
    bool denoise = false;
    bool forceDeblurring = false;
    bool monitorDir = false;
};

#endif