#pragma once
#include "metadata.h"
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <string>

struct ImageMatrix {
    cv::Mat imageMatrix;
    std::string imagePath;
};

class FeatureMatcher {
public:
    FeatureMatcher(ExifToolPipe& tool, const std::string& detectorType = "SIFT");
    bool setWarped(const std::string& firstPath, const std::string& secondPath);
    bool computeHomography(
        const ImageMatrix img1, 
        const ImageMatrix img2, 
        cv::Mat& H, 
        std::vector<cv::DMatch>& goodMatches
    );

private:
    cv::Ptr<cv::Feature2D> detector;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    ExifToolPipe& exiftool;

    void detectAndDescribe(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
};
