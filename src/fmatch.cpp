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

bool FeatureMatcher::computeHomography(const ImageMatrix src, const ImageMatrix dst, cv::Mat& H, std::vector<cv::DMatch>& goodMatches) {
    goodMatches.clear(); 

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    std::string srcPath, dstPath;

    srcPath = src.imagePath;
    dstPath = dst.imagePath;

    detectAndDescribe(src.imageMatrix, kp1, desc1);
    detectAndDescribe(dst.imageMatrix, kp2, desc2);

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

    std::cout << "[" << dstPath << "] Good matches found: " << goodMatches.size() << '\n';

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

    cv::Mat inlierMask;
    cv::Mat affine = cv::estimateAffinePartial2D(pts2, pts1, inlierMask, cv::RANSAC, 3.0);

    auto inliers = cv::countNonZero(inlierMask);
    if (inliers < 100) {
        std::cerr << "[ERROR] Too few inliers after affine estimation (" << inliers << "). Skipping this image (" << dstPath << ").\n";
        return false;
    } else {
        std::cout << "[" << dstPath << "] Inliers after affine estimation: " << inliers << "\n";
    }

    if (!affine.empty()) {
        cv::Mat H_affine = cv::Mat::eye(3, 3, affine.type());
        affine.copyTo(H_affine(cv::Rect(0, 0, 3, 2)));
        H = H_affine;

        double detH = cv::determinant(H);
        if (std::abs(detH) < 0.05 || std::abs(1.0 / detH) > 20.0) {
            std::cerr << "[ERROR] Suspicious homography (det = " << detH << "). Skipping image.\n";
            return false;
        }
    }

    return !H.empty();
}