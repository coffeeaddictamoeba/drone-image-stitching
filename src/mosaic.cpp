#include "../include/mosaic.h"
#include "../include/fmatch.h"

#include <limits>
#include <iostream>

MosaicBuilder::MosaicBuilder(const std::string& refImagePath, const std::string& targetImagePath)
    : refPath(refImagePath), targetPath(targetImagePath) {}

bool MosaicBuilder::loadImages() {
    refImage = cv::imread(refPath);
    targetImage = cv::imread(targetPath);
    if (refImage.empty() || targetImage.empty()) {
        std::cerr << "Error loading input images.\n";
        return false;
    }
    return true;
}

bool MosaicBuilder::alignImages() {
    FeatureMatcher matcher("SIFT");
    std::vector<cv::DMatch> matches;
    return matcher.computeHomography(refImage, targetImage, homography, matches);
}

void MosaicBuilder::computeBoundingBox(const std::vector<cv::Point2f>& warpedCorners,
                                       int& width, int& height, cv::Mat& translation) {
    std::vector<cv::Point2f> refCorners = {
        {0, 0},
        {static_cast<float>(refImage.cols), 0},
        {static_cast<float>(refImage.cols), static_cast<float>(refImage.rows)},
        {0, static_cast<float>(refImage.rows)}
    };

    std::vector<cv::Point2f> allCorners = refCorners;
    allCorners.insert(allCorners.end(), warpedCorners.begin(), warpedCorners.end());

    float minX = std::numeric_limits<float>::max(), minY = minX;
    float maxX = std::numeric_limits<float>::lowest(), maxY = maxX;

    for (const auto& pt : allCorners) {
        minX = std::min(minX, pt.x);
        minY = std::min(minY, pt.y);
        maxX = std::max(maxX, pt.x);
        maxY = std::max(maxY, pt.y);
    }

    width = static_cast<int>(std::ceil(maxX - minX));
    height = static_cast<int>(std::ceil(maxY - minY));

    translation = (cv::Mat_<double>(3, 3) <<
        1, 0, -minX,
        0, 1, -minY,
        0, 0, 1);
}

bool MosaicBuilder::stitchImages(cv::Mat& outputMosaic) {
    if (!loadImages()) return false;
    if (!alignImages()) {
        std::cerr << "Homography estimation failed.\n";
        return false;
    }

    std::vector<cv::Point2f> corners = {
        {0, 0},
        {static_cast<float>(targetImage.cols), 0},
        {static_cast<float>(targetImage.cols), static_cast<float>(targetImage.rows)},
        {0, static_cast<float>(targetImage.rows)}
    };
    std::vector<cv::Point2f> warpedCorners;
    cv::perspectiveTransform(corners, warpedCorners, homography);

    int width, height;
    cv::Mat translation;
    computeBoundingBox(warpedCorners, width, height, translation);

    cv::Mat warped;
    cv::warpPerspective(targetImage, warped, translation * homography, cv::Size(width, height));

    outputMosaic = warped.clone();
    int x = static_cast<int>(translation.at<double>(0, 2));
    int y = static_cast<int>(translation.at<double>(1, 2));
    std::cout << "x: " << x << ", y: " << y << '\n';

    if (x >= 0 && y >= 0 &&
        x + refImage.cols <= outputMosaic.cols &&
        y + refImage.rows <= outputMosaic.rows) {
        cv::Mat roi = outputMosaic(cv::Rect(x, y, refImage.cols, refImage.rows));
        refImage.copyTo(roi);
    } else {
        std::cerr << "ERROR: Reference image ROI is out of bounds. Adjust bounding box or translation.\n";
        return false;
    }
  // naive overlay — optionally add blending

    return true;
}
