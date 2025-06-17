#include "../include/mosaic.h"
#include <cstdio>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Wrong arguments" << '\n';
        return -1;
    }

    std::string path1 = argv[1];
    std::cout << "Source Image: " << path1 << '\n';
    std::string path2 = argv[2];
    std::cout << "Destination Image: " << path2 << '\n';
    //std::string path2 = "incoming/waypoint_22_20250521_105937.jpg";
    // std::string path1 = "incoming/waypoint_20_20250521_105913.jpg";
    // std::string path2 = "incoming/waypoint_21_20250521_105925.jpg";

    MosaicBuilder mosaic(path1, path2);
    cv::Mat stitched;

    if (mosaic.stitchImages(stitched)) {
        std::cout << "Stitching successful.\n";
        cv::imwrite("incoming/stitched_result.jpg", stitched);
    } else {
        std::cerr << "Stitching failed.\n";
    }

    return 0;
}
