#include "../include/deblur.h"
#include "../include/monitor.h"
#include "helpers.h"
#include <csignal>
#include <cstring>
#include <exception>
#include <string>

#define RESET   "\033[0m"
#define RED     "\033[31m"      // Errors
#define YELLOW  "\033[33m"      // Warnings
#define GREEN   "\033[32m"      // Success


DeblurConfig config;

void parseArgs(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--generate-test") {
            config.generateTest = true;
            if (i + 1 < argc)
                config.testImagePath = argv[++i];
            break;
        } else if (arg == "--blur" && i + 1 < argc) {
            config.blur = true;
            if (std::strcmp(argv[i+1], "--source-dir") != 0) // allow to blur all images in specified directory
                config.originalImagePath = argv[++i]; // only in case of single input image for blurring
        } else if (arg == "--source-dir" && i + 1 < argc) {
            config.monitorDir = true;
            config.sourceDir = argv[++i];
        } else if (arg == "--target-dir" && i + 1 < argc) 
            config.targetDir = argv[++i];
        else if (arg == "--overwrite-metadata") 
            config.overwriteMetadata = true;
        else if (arg == "--denoise") 
            config.denoise = true;
        else if (arg == "--force") 
            config.forceDeblurring = true;
        else if (arg == "--snr" && i + 1 < argc)
            config.snr = std::stof(argv[++i]);
        else if (arg == "--blur-threshold" && i + 1 < argc)
            config.blurThreshold = std::stof(argv[++i]);
        else if (arg == "--sensor-width" && i + 1 < argc)
            config.sensorWidth = std::stof(argv[++i]);
        else if (arg == "--sensor-height" && i + 1 < argc)
            config.sensorHeight = std::stof(argv[++i]);
        else {
            continue;
        }
    }
}

int main(int argc, char** argv) {
    parseArgs(argc, argv);

    if (config.generateTest) { // generate test
        Deblurrer deblurrer(config);
        deblurrer.generateTest();
        return 0;
    } else if (config.blur) { // blur image
        if (!config.monitorDir) {
            std::string imageToBlur = config.originalImagePath;
    
            Deblurrer deblurrer(config);
            deblurrer.blurImage(imageToBlur, false);
            return 0;
        } else { // blur all images from specified directory
            std::signal(SIGINT, signalHandler);
            std::string sourceDir = config.sourceDir;

            Deblurrer deblurrer(config);

            DirectoryMonitor monitor(sourceDir, std::bind(&Deblurrer::blurImage, &deblurrer, std::placeholders::_1, false));
            
            monitor.start();
            while (!stopFlag) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            monitor.stop();
            return 0;
        }
    } else if (config.monitorDir) { // deblur all images from specified directory
        std::signal(SIGINT, signalHandler);
        std::string sourceDir = config.sourceDir;

        Deblurrer deblurrer(config);

        DirectoryMonitor monitor(sourceDir, std::bind(&Deblurrer::deblurImage, &deblurrer, std::placeholders::_1, config.snr));
        monitor.start();
        while (!stopFlag) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        monitor.stop();
        return 0;
    } else { // deblur single image
        try {
            std::string blurredImage = argv[1];

            Deblurrer deblurrer(config);
            deblurrer.deblurImage(blurredImage, config.snr);
            return 0;
        } catch (std::exception &e) {
            std::cerr << RED <<"Error: " << e.what() << RESET << "\n";
            return 1;
        }
    }
}