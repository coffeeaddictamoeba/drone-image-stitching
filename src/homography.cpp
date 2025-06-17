#include "../include/homography.h"
#include <cmath>

HomographyEstimator::HomographyEstimator(const std::string& imagePath) {
    MetadataExtractor extractor(imagePath);
    metadata = extractor.parseMetadata();
}

void HomographyEstimator::computeCameraMatrix(cv::Mat& K) const {
    double fx = (metadata.focalLengthMM / metadata.sensorWidthMM) * metadata.imageWidth;
    double fy = (metadata.focalLengthMM / metadata.sensorHeightMM) * metadata.imageHeight;
    double cx = metadata.imageWidth / 2.0;
    double cy = metadata.imageHeight / 2.0;

    K = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                                   0, fy, cy,
                                   0, 0, 1);
}

void HomographyEstimator::computeRotationMatrix(cv::Mat& R) const {
    double yaw = metadata.yawDeg * CV_PI / 180.0;
    double pitch = metadata.pitchDeg * CV_PI / 180.0;
    double roll = metadata.rollDeg * CV_PI / 180.0;

    cv::Mat Rx = (cv::Mat_<double>(3, 3) <<
        1, 0, 0,
        0, cos(pitch), -sin(pitch),
        0, sin(pitch), cos(pitch));

    cv::Mat Ry = (cv::Mat_<double>(3, 3) <<
        cos(roll), 0, sin(roll),
        0, 1, 0,
        -sin(roll), 0, cos(roll));

    cv::Mat Rz = (cv::Mat_<double>(3, 3) <<
        cos(yaw), -sin(yaw), 0,
        sin(yaw), cos(yaw), 0,
        0, 0, 1);

    R = Rz * Ry * Rx;
}

cv::Mat HomographyEstimator::warpImage(const cv::Mat& inputImage, const cv::Size& outputSize) {
    cv::Mat K, R;
    computeCameraMatrix(K);
    computeRotationMatrix(R);

    // Use projection: H = K * R * K.inv()
    cv::Mat H = K * R * K.inv();

    cv::Mat warped;
    cv::warpPerspective(inputImage, warped, H, outputSize, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return warped;
}
