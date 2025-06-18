#include "../include/mosaic.h"
#include "../include/fmatch.h"
#include <limits>
#include <iostream>
#include <filesystem>

MosaicBuilder::MosaicBuilder(const std::string& refImagePath, const std::string& targetImagePath, ExifToolPipe& tool)
    : ref{cv::imread(refImagePath, cv::IMREAD_UNCHANGED), refImagePath},
      target{cv::imread(targetImagePath, cv::IMREAD_UNCHANGED), targetImagePath},
      exiftool(tool)
{
    if (ref.imageMatrix.empty()) throw std::runtime_error("Failed to load: " + refImagePath);
    if (target.imageMatrix.empty()) throw std::runtime_error("Failed to load: " + targetImagePath);
}

bool MosaicBuilder::alignImages() {
    FeatureMatcher matcher(exiftool, "SIFT");
    std::vector<cv::DMatch> matches;
    return matcher.computeHomography(ref, target, homography, matches);
}

void MosaicBuilder::computeBoundingBox(const std::vector<cv::Point2f>& warpedCorners,
                                       int& width, int& height, cv::Mat& translation) {
    std::vector<cv::Point2f> refCorners = {
        {0, 0},
        {static_cast<float>(ref.imageMatrix.cols), 0},
        {static_cast<float>(ref.imageMatrix.cols), static_cast<float>(ref.imageMatrix.rows)},
        {0, static_cast<float>(ref.imageMatrix.rows)}
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

void MosaicBuilder::computeExpandedCanvas(
    const cv::Mat& H,
    const cv::Size& refSize,
    const cv::Size& targetSize,
    int& width,
    int& height,
    cv::Mat& translation)
{
    std::vector<cv::Point2f> refCorners = {
        {0, 0},
        {static_cast<float>(refSize.width), 0},
        {static_cast<float>(refSize.width), static_cast<float>(refSize.height)},
        {0, static_cast<float>(refSize.height)}
    };

    std::vector<cv::Point2f> targetCorners = {
        {0, 0},
        {static_cast<float>(targetSize.width), 0},
        {static_cast<float>(targetSize.width), static_cast<float>(targetSize.height)},
        {0, static_cast<float>(targetSize.height)}
    };

    std::vector<cv::Point2f> warpedCorners;
    cv::perspectiveTransform(targetCorners, warpedCorners, H);

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

cv::Mat MosaicBuilder::warpPartial(const cv::Mat& input, const cv::Mat& H, const cv::Rect& mosaicROI, const cv::Size& mosaicSize) {
    cv::Mat Hinv = H.inv();

    std::vector<cv::Point2f> mosaicCorners = {
        {static_cast<float>(mosaicROI.x), static_cast<float>(mosaicROI.y)},
        {static_cast<float>(mosaicROI.x + mosaicROI.width), static_cast<float>(mosaicROI.y)},
        {static_cast<float>(mosaicROI.x + mosaicROI.width), static_cast<float>(mosaicROI.y + mosaicROI.height)},
        {static_cast<float>(mosaicROI.x), static_cast<float>(mosaicROI.y + mosaicROI.height)}
    };

    std::vector<cv::Point2f> sourceROI;
    cv::perspectiveTransform(mosaicCorners, sourceROI, Hinv);

    cv::Rect2f floatBounds = cv::boundingRect(sourceROI);
    cv::Rect srcBounds(
        std::max(0, static_cast<int>(std::floor(floatBounds.x))),
        std::max(0, static_cast<int>(std::floor(floatBounds.y))),
        std::min(input.cols, static_cast<int>(std::ceil(floatBounds.width))),
        std::min(input.rows, static_cast<int>(std::ceil(floatBounds.height)))
    );

    if (srcBounds.width <= 0 || srcBounds.height <= 0) return cv::Mat();

    cv::Mat inputROI = input(srcBounds);

    cv::Mat T = (cv::Mat_<double>(3,3) <<
        1, 0, srcBounds.x,
        0, 1, srcBounds.y,
        0, 0, 1);
    cv::Mat adjustedH = H * T;

    cv::Mat result(mosaicSize, CV_8UC4, cv::Scalar(0,0,0,0));
    cv::warpPerspective(inputROI, result, adjustedH, mosaicSize,
        cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

    return result;
}

bool MosaicBuilder::stitchImages(cv::Mat& outputMosaic) {
    if (!std::filesystem::exists(ref.imagePath) || !std::filesystem::exists(target.imagePath)) {
        std::cerr << "[ERROR] One of the input image paths does not exist.\n";
        return false;
    }    

    std::cout << "[INFO] Starting image stitching.\n";

    cv::Mat refRGBA, targetRGBA;
    cv::cvtColor(ref.imageMatrix, refRGBA, cv::COLOR_BGR2BGRA);
    cv::cvtColor(target.imageMatrix, targetRGBA, cv::COLOR_BGR2BGRA);

    FeatureMatcher matcher(exiftool, "SIFT");
    std::vector<cv::DMatch> matches;
    cv::Mat H;

    bool targetAlreadyWarped = exiftool.hasExifTag(target.imagePath, "XMPWarped");

    if (!targetAlreadyWarped) {
        std::cout << "[INFO] Estimating homography...\n";
        if (!matcher.computeHomography(ref, target, H, matches)) {
            std::cerr << "[ERROR] Homography estimation failed.\n";
            return false;
        }
    } else {
        std::cout << "[INFO] Target image already warped. Using identity homography.\n";
        H = cv::Mat::eye(3, 3, CV_64F);
    }

    int mosaicWidth, mosaicHeight;
    cv::Mat translation;
    computeExpandedCanvas(H, refRGBA.size(), targetRGBA.size(), mosaicWidth, mosaicHeight, translation);

    std::cout << "[INFO] Computed mosaic canvas size: " << mosaicWidth << "x" << mosaicHeight << "\n";

    cv::Mat mosaic(mosaicHeight, mosaicWidth, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    if (!targetAlreadyWarped) {
        std::cout << "[INFO] Warping target image (partial)...\n";
        cv::Mat warpedPartial = warpPartial(targetRGBA, translation * H, cv::Rect(0, 0, mosaicWidth, mosaicHeight), cv::Size(mosaicWidth, mosaicHeight));

        if (warpedPartial.empty()) {
            std::cerr << "[ERROR] Partial warp failed. Empty result.\n";
            return false;
        }

        std::vector<cv::Mat> channels;
        cv::split(warpedPartial, channels);
        cv::Mat mask = channels[3] > 0;
        warpedPartial.copyTo(mosaic, mask);
    } else {
        std::cout << "[INFO] Copying pre-warped target into mosaic...\n";
        int tx = static_cast<int>(translation.at<double>(0, 2));
        int ty = static_cast<int>(translation.at<double>(1, 2));

        if (tx < 0 || ty < 0 || tx + targetRGBA.cols > mosaic.cols || ty + targetRGBA.rows > mosaic.rows) {
            std::cerr << "[ERROR] Target image is out of mosaic bounds.\n";
            return false;
        }

        cv::Mat roi = mosaic(cv::Rect(tx, ty, targetRGBA.cols, targetRGBA.rows));
        for (int y = 0; y < targetRGBA.rows; ++y) {
            for (int x = 0; x < targetRGBA.cols; ++x) {
                cv::Vec4b px = targetRGBA.at<cv::Vec4b>(y, x);
                if (px[3] > 0)
                    roi.at<cv::Vec4b>(y, x) = px;
            }
        }
    }

    std::cout << "[INFO] Blending reference image...\n";
    int refX = static_cast<int>(translation.at<double>(0, 2));
    int refY = static_cast<int>(translation.at<double>(1, 2));

    if (refX < 0 || refY < 0 || refX + refRGBA.cols > mosaic.cols || refY + refRGBA.rows > mosaic.rows) {
        std::cerr << "[ERROR] Reference image out of bounds.\n";
        return false;
    }

    cv::Mat refROI = mosaic(cv::Rect(refX, refY, refRGBA.cols, refRGBA.rows));
    for (int y = 0; y < refRGBA.rows; ++y) {
        for (int x = 0; x < refRGBA.cols; ++x) {
            cv::Vec4b px = refRGBA.at<cv::Vec4b>(y, x);
            if (px[3] > 0)
                refROI.at<cv::Vec4b>(y, x) = px;
        }
    }

    outputMosaic = mosaic;
    std::cout << "[INFO] Stitching complete.\n";
    return true;
}