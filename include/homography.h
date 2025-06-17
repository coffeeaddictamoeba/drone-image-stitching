#pragma once
#include <opencv4/opencv2/opencv.hpp>
#include "../include/metadata.h"

class HomographyEstimator {
public:
    explicit HomographyEstimator(const std::string& imagePath);

    cv::Mat warpImage(const cv::Mat& inputImage, const cv::Size& outputSize);

private:
    CameraMetadata metadata;
    void computeCameraMatrix(cv::Mat& K) const;
    void computeRotationMatrix(cv::Mat& R) const;
};
