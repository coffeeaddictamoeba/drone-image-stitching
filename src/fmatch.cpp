#include "../include/fmatch.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <vector>

FeatureMatcher::FeatureMatcher(ExifToolPipe& tool, const std::string& detectorType) 
    : exiftool(tool) {
    if (detectorType == "SIFT") {
        detector = cv::SIFT::create();
        cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>();
        matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams);
    } else if (detectorType == "ORB") {
        detector = cv::ORB::create();
        cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1);
        matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams);
    } else {
        throw std::runtime_error("Unsupported detector type: " + detectorType);
    }
}

void FeatureMatcher::detectAndDescribe(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    if (image.empty()) {
        std::cerr << "[WARN] FeatureMatcher::detectAndDescribe called with empty image.\n";
        keypoints.clear();
        descriptors.release();
        return;
    }
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}

bool FeatureMatcher::setWarped(const std::string& refPath, const std::string& targetPath) {
    bool alreadyWarped = exiftool.hasExifTag(targetPath, "XMP:Warped");
    if (!alreadyWarped) {
        exiftool.setExifTag(targetPath, "-XMPWarped=True -XMPWarpedFrom=" + refPath);
    }
    return alreadyWarped;
}

bool FeatureMatcher::computeHomography(const ImageMatrix img1, const ImageMatrix img2, cv::Mat& H, std::vector<cv::DMatch>& goodMatches) {
    goodMatches.clear(); 

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;

    detectAndDescribe(img1.imageMatrix, kp1, desc1);
    detectAndDescribe(img2.imageMatrix, kp2, desc2);

    if (desc1.empty() || desc2.empty()) {
        std::cerr << "[ERROR] Descriptors are empty. Cannot compute homography.\n";
        H = cv::Mat();
        return false;
    }

    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(desc1, desc2, knnMatches, 2);

    for (const auto& pair : knnMatches) {
        if (pair.size() >= 2) { 
            if (pair[0].distance < 0.75 * pair[1].distance) {
                goodMatches.push_back(pair[0]);
            }
        }
    }

    std::cout << "Good matches found: " << goodMatches.size() << '\n';

    if (goodMatches.size() < 10) {
        std::cerr << "[WARN] Not enough good matches (" << goodMatches.size() << ") found to compute homography. Minimum required: " << 10 << ".\n";
        H = cv::Mat();
        return false;
    }

    std::vector<cv::Point2f> pts1, pts2;
    pts1.reserve(goodMatches.size());
    pts2.reserve(goodMatches.size());

    for (const auto& match : goodMatches) {
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }

    setWarped(img1.imagePath, img2.imagePath);

    H = cv::findHomography(pts2, pts1, cv::RANSAC);
    
    std::cout << "Homography Matrix (Feature Matching):\n";
    if (!H.empty()) {
        std::cout << H << '\n';
    } else {
        std::cerr << "[ERROR] Homography computation returned an empty matrix.\n";
    }

    return !H.empty();
}