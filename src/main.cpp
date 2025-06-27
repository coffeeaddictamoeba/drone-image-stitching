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

void buildMosaic(const std::string refImage, const std::string targetImage, const std::string outputDir) {
    try {
        ExifToolPipe exiftool;
        TileManager tileManager(outputDir, exiftool);
        MosaicBuilder builder(exiftool, tileManager);

        if (!builder.stitchToTiles(refImage, targetImage)) {
            std::cerr << "Mosaic stitching failed.\n";
            return exit(1);
        }

        std::cout << "Mosaic stitching completed successfully.\n";

        cv::Rect bounds;
        cv::Mat mosaic = builder.mosaicFromTiles(outputDir, bounds);
        if (mosaic.empty()) {
            std::cerr << "Failed to reconstruct mosaic.\n";
            exit(1);
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
        exit(1);
    }
}

void useReadyMosaic(const std::string outputDir) {
    try {
        ExifToolPipe exiftool;
        TileManager tileManager(outputDir, exiftool);
        MosaicBuilder builder(exiftool, tileManager);

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
        exit(1);
    }
    
}

// for testing purposes the mosaic from tiles is saved in a full size. In prod it is better to limit its size
int main(int argc, char** argv) {
    if (argc == 4) {
        std::string refImage = argv[1];
        std::string targetImage = argv[2];
        std::string outputDir = argv[3];
        std::cout << "Building new mosaic...";
        buildMosaic(refImage, targetImage, outputDir); // in future this can be used in case image does not match the existing mosaic not to lose the data
        return 0;
    } else if (argc == 2) {
        std::string outputDir = argv[1];
        useReadyMosaic(outputDir);
        return 0;
    } else {
        std::cerr << "Usage: " << argv[0] << " <ref_image> <target_image> <output_dir> <exiftool_path>\n";
        return 1;
    }
}
