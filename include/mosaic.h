#pragma once
#include "fmatch.h"
#include "metadata.h"
#include <opencv4/opencv2/opencv.hpp>
#include <string>

class MosaicBuilder {
public:
    MosaicBuilder(const std::string& refImagePath, const std::string& targetImagePath, ExifToolPipe& tool);
    bool stitchImages(cv::Mat& outputMosaic);

private:
    ImageMatrix ref, target;
    cv::Mat homography;
    ExifToolPipe& exiftool;

    bool loadImages();
    bool alignImages();
    void computeExpandedCanvas(const cv::Mat& H,
        const cv::Size& refSize,
        const cv::Size& targetSize,
        int& width,
        int& height,
        cv::Mat& translation
    );
    cv::Rect computeAlphaBoundingBox(const cv::Mat& image);
    cv::Mat warpPartial(const cv::Mat& input, const cv::Mat& H, const cv::Rect& mosaicROI, const cv::Size& mosaicSize);
    void computeBoundingBox(const std::vector<cv::Point2f>& warpedCorners,
                            int& width, int& height, cv::Mat& translation);
};
