#include "../include/homography.h"
#include "../include/fmatch.h"
#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <ostream>
#include <string>

int main() {
    std::string imagePath = "incoming/waypoint_20_20250521_105913.jpg";

    cv::Mat input = cv::imread(imagePath);
    if (input.empty()) {
        std::cerr << "Image load failed\n";
        return -1;
    }

    HomographyEstimator estimator(imagePath);

    cv::Mat warped = estimator.warpImage(input, cv::Size(5000, 5000));
    cv::imwrite("warped_output.jpg", warped);

    FeatureMatcher matcher;
    cv::Mat img2 = cv::imread("incoming/waypoint_21_20250521_105925.jpg");
    if (!img2.empty()) {
        cv::Mat H;
        std::vector<cv::DMatch> matches;
        if (matcher.computeHomography(input, img2, H, matches)) {
            std::cout << "Feature-based homography computed successfully.\n";
        }
        std::cout << H << std::endl;
    }

    return 0;
}