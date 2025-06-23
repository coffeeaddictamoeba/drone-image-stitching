#include "../include/mosaic.h"
#include "../include/metadata.h"
#include <iostream>
#include <string>

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
        MosaicTileManager tileManager(outputDir, exiftool);
        MosaicBuilder builder(refImage, targetImage, exiftool, tileManager);

        if (!builder.stitchToTiles()) {
            std::cerr << "Mosaic stitching failed.\n";
            return 1;
        }

        std::cout << "Mosaic stitching completed successfully.\n";

        

        if (!builder.addImageToMosaic("incoming/waypoint_22_20250521_105937.jpg")) {
            std::cout << "Image was not aligned.";
        }

        cv::Rect bounds;
        cv::Mat mosaic = builder.mosaicFromTiles(outputDir, bounds);
        if (mosaic.empty()) {
            std::cerr << "Failed to reconstruct mosaic.\n";
            return 1;
        }

        std::cout << "Mosaic bounds: " << bounds << "\n";
        cv::imwrite("result.png", mosaic);
        std::cout << "Saved reconstructed mosaic to: " << "result.png" << "\n";
        
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
