#include "../include/deblur.h"

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
            config.originalImagePath = argv[++i];
        }
        else if (arg == "--overwrite-metadata") config.overwriteMetadata = true;
        else if (arg == "--denoise") config.denoise = true;
        else if (arg == "--snr" && i + 1 < argc)
            config.snr = std::stof(argv[++i]);
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
        std::string imageToBlur = config.originalImagePath;
        std::string prefix = "_blurred";
        std::string blurredImage = constructPathWithPrefix(imageToBlur, prefix);

        Deblurrer deblurrer(config);
        deblurrer.blurImage(imageToBlur, blurredImage, false);
        return 0;
    } else { // deblur image
        try {
            std::string blurredImage = argv[1]; // not very flexible
            std::string prefix = "_deblurred";
            std::string deblurredImage = constructPathWithPrefix(blurredImage, prefix);

            Deblurrer deblurrer(config);
            deblurrer.deblurImage(blurredImage, deblurredImage, config.snr);
            return 0;
        } catch (...) {
            std::cerr << "Wrong arguments.\n";
            return 1;
        }
    }
}