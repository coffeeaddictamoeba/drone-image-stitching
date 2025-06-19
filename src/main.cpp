#include "../include/mosaic.h"
#include "../include/metadata.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <ref_image> <target_image> <output_dir> <exiftool_path>\n";
        return 1;
    }

    std::string refImage = argv[1];
    std::string targetImage = argv[2];
    std::string outputDir = argv[3];

    try {
        ExifToolPipe exiftool;
        MosaicTileManager tileManager(outputDir);
        MosaicBuilder builder(refImage, targetImage, exiftool, tileManager);

        if (!builder.stitchToTiles()) {
            std::cerr << "Mosaic stitching failed.\n";
            return 1;
        }

        std::cout << "Mosaic stitching completed successfully.\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
