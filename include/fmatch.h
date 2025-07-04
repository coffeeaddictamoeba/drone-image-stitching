#pragma once
#include "metadata.h"
#include <optional>
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
    bool computeHomography(const ImageMatrix& src, const ImageMatrix& dst, cv::Mat& H, std::vector<cv::DMatch>& goodMatches);

private:
    cv::Ptr<cv::Feature2D> detector;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    ExifToolPipe& exiftool;

    void detectAndDescribe(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
};

class Deblurrer {
    public:
        Deblurrer();
        // methods will be added soon
    private:
};