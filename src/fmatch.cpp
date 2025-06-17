#include "../include/fmatch.h"
#include <stdexcept>

FeatureMatcher::FeatureMatcher(const std::string& detectorType) {
    if (detectorType == "SIFT")
        detector = cv::SIFT::create();
    else if (detectorType == "ORB")
        detector = cv::ORB::create();
    else
        throw std::runtime_error("Unsupported detector type.");

    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
}

void FeatureMatcher::detectAndDescribe(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}

bool FeatureMatcher::computeHomography(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& H, std::vector<cv::DMatch>& goodMatches) {
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;

    detectAndDescribe(img1, kp1, desc1);
    detectAndDescribe(img2, kp2, desc2);

    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(desc1, desc2, knnMatches, 2);

    for (const auto& pair : knnMatches) {
        if (pair.size() == 2 && pair[0].distance < 0.75 * pair[1].distance)
            goodMatches.push_back(pair[0]);
    }

    if (goodMatches.size() < 10)
        return false;

    std::vector<cv::Point2f> pts1, pts2;
    for (auto& match : goodMatches) {
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }

    H = cv::findHomography(pts2, pts1, cv::RANSAC);
    std::cout << "Homography Matrix (Feature Matching):\n";
    std::cout << H << '\n';
    return !H.empty();
}
