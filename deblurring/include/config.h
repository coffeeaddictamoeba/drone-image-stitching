#ifndef DEBLUR_CONFIG_H
#define DEBLUR_CONFIG_H

#include <string>
#include <cstddef>

struct DeblurConfig {
    std::string testImagePath = "synthetic.jpg";
    std::string originalImagePath = "";
    
    float snr = 500.0;
    bool generateTest = false;
    bool overwriteMetadata = false;
    bool blur = false;
    bool denoise = false;
    bool forceDeblurring = false;
};

inline std::string constructPathWithPrefix(std::string &originalPath, std::string &prefix) {
    size_t extensionStart = originalPath.find('.');
    std::string imageName = originalPath.substr(0, extensionStart);
    std::string imageExtension = originalPath.substr(extensionStart, originalPath.size());
    return imageName + prefix + imageExtension;
}

#endif