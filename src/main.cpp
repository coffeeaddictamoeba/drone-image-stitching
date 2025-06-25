#include "../include/mosaic.h"
#include "../include/metadata.h"
#include <iostream>
#include <string>

void add(const std::string imagePath, std::string outputDir, MosaicBuilder builder, int idx) {
    if (!builder.addImageToMosaic(imagePath)) {
        std::cout << "Image was not aligned.";
    }

    cv::Rect bounds;
    cv::Mat mosaic = builder.mosaicFromTiles(outputDir, bounds);
    if (mosaic.empty()) {
        std::cerr << "Failed to reconstruct mosaic.\n";
        exit(1);
    }

    std::string res = "result_" + std::to_string(idx) + ".png";
    std::cout << "Mosaic bounds: " << bounds << "\n";
    cv::imwrite(res, mosaic);
    std::cout << "Saved reconstructed mosaic to: " << res << "\n";
}

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

        cv::Rect bounds;
        cv::Mat mosaic = builder.mosaicFromTiles(outputDir, bounds);
        if (mosaic.empty()) {
            std::cerr << "Failed to reconstruct mosaic.\n";
            return 1;
        }

        std::cout << "Mosaic bounds: " << bounds << "\n";
        cv::imwrite("result.png", mosaic);
        std::cout << "Saved reconstructed mosaic to: " << "result.png" << "\n";
        int idx = 0;

        while (true) {
            std::string newImagePath = "";
            std::cout << "Enter a path to an image you want to add: ";
            std::cin >> newImagePath;

            add(newImagePath, outputDir, builder, idx);
            idx++;
        }

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
