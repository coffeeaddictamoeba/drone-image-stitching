#pragma once
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <string>

class FeatureMatcher {
public:
    FeatureMatcher(const std::string& detectorType = "SIFT");
    bool computeHomography(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& H, std::vector<cv::DMatch>& goodMatches);

private:
    cv::Ptr<cv::Feature2D> detector;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    void detectAndDescribe(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
};
