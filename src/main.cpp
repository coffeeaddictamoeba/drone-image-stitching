#include "../include/mosaic.h"
#include <opencv4/opencv2/opencv.hpp>

int main() {
    std::string path1 = "incoming/waypoint_20_20250521_105913.jpg";
    std::string path2 = "incoming/waypoint_21_20250521_105925.jpg";

    MosaicBuilder mosaic(path1, path2);
    cv::Mat stitched;

    if (mosaic.stitchImages(stitched)) {
        std::cout << "Stitching successful.\n";
        cv::imwrite("stitched_result.jpg", stitched);
    } else {
        std::cerr << "Stitching failed.\n";
    }

    return 0;
}
