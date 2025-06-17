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

    // transparent bg
    cv::Mat refImageRGBA, targetImageRGBA;
    cv::cvtColor(refImage, refImageRGBA, cv::COLOR_BGR2BGRA);
    cv::cvtColor(targetImage, targetImageRGBA, cv::COLOR_BGR2BGRA);

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

    cv::Mat transparentMosaic(height, width, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    cv::Mat warpedRGBA;
cv::warpPerspective(targetImageRGBA, warpedRGBA, translation * homography, cv::Size(width, height),
                    cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));  // Transparent border

    std::vector<cv::Mat> channels;
    cv::split(warpedRGBA, channels); // channels[3] is alpha
    cv::Mat mask = channels[3] > 0;
    warpedRGBA.copyTo(transparentMosaic, mask);

    int x = static_cast<int>(translation.at<double>(0, 2));
    int y = static_cast<int>(translation.at<double>(1, 2));
    std::cout << "x: " << x << ", y: " << y << '\n';

    if (x >= 0 && y >= 0 &&
        x + refImage.cols <= transparentMosaic.cols &&
        y + refImage.rows <= transparentMosaic.rows) {
        cv::Mat roi = transparentMosaic(cv::Rect(x, y, refImageRGBA.cols, refImageRGBA.rows));
            for (int r = 0; r < refImageRGBA.rows; ++r) {
                for (int c = 0; c < refImageRGBA.cols; ++c) {
                    cv::Vec4b srcPixel = refImageRGBA.at<cv::Vec4b>(r, c);
                    if (srcPixel[3] > 0) {
                        roi.at<cv::Vec4b>(r, c) = srcPixel;
                    }
                }
            }
            
    } else {
        std::cerr << "ERROR: Reference image ROI is out of bounds. Adjust bounding box or translation.\n";
        return false;
    }

    outputMosaic = transparentMosaic;
    return true;
}
