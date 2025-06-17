#pragma once
#include <opencv4/opencv2/opencv.hpp>
#include <string>

class MosaicBuilder {
public:
    MosaicBuilder(const std::string& refImagePath, const std::string& targetImagePath);
    bool stitchImages(cv::Mat& outputMosaic);

private:
    std::string refPath, targetPath;
    cv::Mat refImage, targetImage;
    cv::Mat homography;

    bool loadImages();
    bool alignImages();
    void computeBoundingBox(const std::vector<cv::Point2f>& warpedCorners,
                            int& width, int& height, cv::Mat& translation);
};
